"""Experiment bookkeeping for budgeted remote training services.

Two small pieces of infrastructure that every budgeted study needs and that are easy to get
wrong under deadline pressure:

* ``HTTPClient`` -- a dependency-free client for a /budget /submit /experiments service with
  retries on transient errors (it never retries a 4xx: a 400 means "out of budget", a 409 means
  "duplicate", and retrying those wastes budget or hides bugs).
* ``Ledger`` -- an append-only JSONL file that records every submission and every result,
  keyed by a canonical hash of the request. Re-running your notebook or script does not
  resubmit (and re-pay for) runs you already have, and a crash mid-study loses nothing.

Nothing here decides *which* runs to do -- that is the study design, and it is yours.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Optional


def request_key(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


class HTTPClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 retries: int = 4, backoff: float = 0.5, timeout: float = 30.0):
        self.base = (base_url or os.environ.get("A3_BASE_URL", "http://127.0.0.1:8765")).rstrip("/")
        self.key = api_key if api_key is not None else os.environ.get("A3_API_KEY", "")
        self.retries, self.backoff, self.timeout = retries, backoff, timeout

    def _call(self, method: str, path: str, body: Optional[dict] = None):
        data = None if body is None else json.dumps(body).encode()
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"X-API-Key": self.key,
                                                  "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500:            # client errors are final
                    raise RuntimeError(f"{e.code} {e.read().decode(errors='replace')}") from None
                err = e
            except (urllib.error.URLError, TimeoutError) as e:
                err = e
            time.sleep(self.backoff * 2 ** attempt)
        raise RuntimeError(f"giving up on {method} {path}: {err}")

    def budget(self):
        return self._call("GET", "/budget")

    def submit(self, body: dict):
        return self._call("POST", "/submit", body)

    def experiments(self):
        return self._call("GET", "/experiments")

    def experiment(self, eid: int):
        return self._call("GET", f"/experiment/{eid}")

    def final_submission(self, body: dict):
        return self._call("POST", "/final_submission", body)


class Ledger:
    """Append-only record of submissions and results (one JSON object per line)."""

    def __init__(self, path: str):
        self.path = path
        self.submitted: Dict[str, dict] = {}     # key -> {"experiment_id", "body", "t"}
        self.results: Dict[int, dict] = {}       # experiment_id -> final status record
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec["kind"] == "submit":
                        self.submitted[rec["key"]] = rec
                    elif rec["kind"] == "result":
                        self.results[rec["experiment_id"]] = rec

    def _append(self, rec: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def submit_once(self, client: HTTPClient, body: dict, tag: str = "") -> int:
        """Submit ``body`` unless an identical request is already in the ledger."""
        key = request_key(body)
        if key in self.submitted:
            return self.submitted[key]["experiment_id"]
        resp = client.submit(body)
        rec = {"kind": "submit", "key": key, "experiment_id": resp["experiment_id"],
               "tag": tag, "body": body, "t": time.time(), "budget": resp.get("budget_summary")}
        self._append(rec)
        self.submitted[key] = rec
        return rec["experiment_id"]

    def poll(self, client: HTTPClient, ids: Iterable[int], interval: float = 30.0,
             max_interval: float = 300.0, deadline: Optional[float] = None) -> List[dict]:
        """Poll until every id has finished; record each final status exactly once."""
        pending = [i for i in ids if i not in self.results]
        wait = interval
        while pending:
            still = []
            for eid in pending:
                e = client.experiment(eid)
                st = e["status"]["status_type"]
                if st in ("completed", "failed"):
                    rec = {"kind": "result", "experiment_id": eid, "status": e["status"],
                           "t": time.time()}
                    self._append(rec)
                    self.results[eid] = rec
                else:
                    still.append(eid)
            pending = still
            if pending:
                if deadline is not None and time.time() > deadline:
                    break
                time.sleep(wait)
                wait = min(max_interval, wait * 1.5)
        return [self.results[i] for i in ids if i in self.results]

    def table(self) -> List[dict]:
        """One row per finished run: tag, request body, status, final loss (or None)."""
        rows = []
        for key, sub in self.submitted.items():
            res = self.results.get(sub["experiment_id"])
            if res is None:
                continue
            losses = res["status"].get("val_losses") or []
            final = losses[-1] if res["status"]["status_type"] == "completed" and losses else None
            rows.append({"experiment_id": sub["experiment_id"], "tag": sub.get("tag", ""),
                         "body": sub["body"], "status": res["status"]["status_type"],
                         "final_loss": final,
                         "used_seconds": res["status"].get("used_runtime_seconds")})
        return sorted(rows, key=lambda r: r["experiment_id"])
