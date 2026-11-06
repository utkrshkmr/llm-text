"""The self-hosted training service for Project 3 (Chapter 27).

A small HTTP service that "trains" a model config and returns validation losses, charging a
wall-clock budget. Behind it sits a hidden loss surface from ``llmsys.scaling`` (randomised by
``--truth-seed`` so that the constants in the source code do not give the answer away).
Endpoints (all take an ``X-API-Key`` header):

    GET  /budget                  used / remaining / total seconds
    POST /submit                  queue a run; reserves max_runtime_seconds
    GET  /experiments             every run you submitted, with status and losses
    GET  /experiment/{id}         one run
    POST /final_submission        your chosen config and predicted loss
    GET  /final_submission        what you submitted
    GET  /score?hours=48          after a final submission: regret against the hidden optimum

    python training_service.py --port 8765 --budget-hours 12 --time-scale 3600 --truth-seed 7
    export TRAIN_API_URL=http://127.0.0.1:8765

``--time-scale 3600`` makes one simulated hour pass per real second, so queued runs complete
while your poller sleeps. Use ``--time-scale 60`` for a more realistic pace.

Simplifications: grouped-query attention (``num_key_value_heads`` < heads) is accepted but
priced as full multi-head attention, and the optimizer config only uses the peak LR.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from llmsys.scaling import APIError, GroundTruth, RunConfig, SyntheticTrainingAPI  # noqa: E402

SEQ_LEN = 512


def config_from_request(body: dict) -> tuple[RunConfig, float, int, int]:
    arch = body["architecture_config"]
    opt = body.get("optimizer_config", {})
    lr = opt.get("lr_scheduler", {}).get("peak_value", opt.get("lr", 3e-4))
    hidden, heads = int(arch["hidden_size"]), int(arch["num_attention_heads"])
    if hidden != heads * int(arch.get("head_dim", hidden // heads)):
        raise APIError(422, "hidden_size must equal num_attention_heads * head_dim")
    if heads % int(arch.get("num_key_value_heads", heads)):
        raise APIError(422, "num_attention_heads must be divisible by num_key_value_heads")
    cfg = RunConfig(n_layer=int(arch["num_hidden_layers"]), d_model=hidden, n_head=heads,
                    d_ff=int(arch["intermediate_size"]), tokens=int(body["total_train_tokens"]),
                    batch_size=int(body["train_batch_size"]), lr=float(lr), seq_len=SEQ_LEN,
                    vocab=int(arch.get("vocab_size", 32_000)),
                    tie_embeddings=bool(arch.get("tie_word_embeddings", False)))
    return cfg, float(body["max_runtime_seconds"]), int(body.get("n_evals", 8)), int(body.get("model_seed", 0))


class MockService:
    def __init__(self, budget_hours=12.0, time_scale=3600.0, api_key=None, seed=0,
                 truth_seed=None):
        truth = GroundTruth() if truth_seed is None else GroundTruth.sample(truth_seed)
        self.api = SyntheticTrainingAPI(total_budget_seconds=budget_hours * 3600, truth=truth,
                                        seed=seed)
        self.time_scale = time_scale
        self.api_key = api_key
        self.lock = threading.Lock()
        self.t_real = time.monotonic()
        self.requests = {}           # eid -> original request body
        self.final = None

    def tick(self):
        now = time.monotonic()
        self.api.advance((now - self.t_real) * self.time_scale)
        self.t_real = now

    def status(self, eid):
        e = self.api.get(eid)
        st = {"status_type": e["status"], "val_losses": e["val_losses"]}
        if e["status"] in ("completed", "failed"):
            st["used_runtime_seconds"] = e["used_runtime_seconds"]
        if e["status"] == "failed":
            st["reason"] = {"reason": e.get("reason", "unexpected"),
                            "partial_val_losses": e["val_losses"]}
        return {"experiment_id": eid, "training_config": self.requests[eid], "status": st}


def make_handler(svc: MockService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep output quiet
            pass

        def _send(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _auth(self):
            if svc.api_key and self.headers.get("X-API-Key") != svc.api_key:
                self._send(401, {"detail": "bad api key"})
                return False
            return True

        def do_GET(self):
            if not self._auth():
                return
            with svc.lock:
                svc.tick()
                if self.path == "/budget":
                    return self._send(200, svc.api.budget())
                if self.path == "/experiments":
                    return self._send(200, [svc.status(e["experiment_id"]) for e in svc.api.experiments()])
                if self.path.startswith("/experiment/"):
                    try:
                        return self._send(200, svc.status(int(self.path.rsplit("/", 1)[1])))
                    except (APIError, ValueError, KeyError):
                        return self._send(404, {"detail": "not found"})
                if self.path == "/final_submission":
                    return self._send(200, svc.final)
                if self.path.startswith("/score"):
                    if svc.final is None:
                        return self._send(400, {"detail": "submit a final config first"})
                    hours = 48.0
                    if "hours=" in self.path:
                        hours = float(self.path.split("hours=", 1)[1].split("&")[0])
                    return self._send(200, svc.api.score(hours * 3600))
            self._send(404, {"detail": "unknown path"})

        def do_POST(self):
            if not self._auth():
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with svc.lock:
                svc.tick()
                try:
                    if self.path == "/submit":
                        cfg, max_rt, n_evals, seed = config_from_request(body)
                        eid = svc.api.submit(cfg, max_rt, n_evals=n_evals, model_seed=seed)
                        svc.requests[eid] = body
                        return self._send(200, {"experiment_id": eid, "budget_summary": svc.api.budget()})
                    if self.path == "/final_submission":
                        cfg, _, _, _ = config_from_request(body["training_config"])
                        svc.api.submit_final(cfg, float(body["predicted_final_loss"]))
                        svc.final = {**body, "submitted_at": time.time()}
                        return self._send(200, svc.final)
                except APIError as err:
                    return self._send(err.status, {"detail": str(err)})
                except (KeyError, ValueError, TypeError) as err:
                    return self._send(422, {"detail": f"bad request: {err!r}"})
            self._send(404, {"detail": "unknown path"})

    return Handler


def serve(port=8765, **kw) -> tuple[ThreadingHTTPServer, MockService]:
    svc = MockService(**kw)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(svc))
    return srv, svc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--budget-hours", type=float, default=12.0)
    p.add_argument("--time-scale", type=float, default=3600.0)
    p.add_argument("--api-key", default=None)
    p.add_argument("--seed", type=int, default=0, help="noise seed")
    p.add_argument("--truth-seed", type=int, default=None,
                   help="randomise the hidden loss surface (recommended)")
    a = p.parse_args()
    srv, _ = serve(a.port, budget_hours=a.budget_hours, time_scale=a.time_scale,
                   api_key=a.api_key, seed=a.seed, truth_seed=a.truth_seed)
    print(f"training service on http://127.0.0.1:{a.port}  (Ctrl-C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
