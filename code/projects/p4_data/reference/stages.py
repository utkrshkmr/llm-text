"""The pipeline stages of Project 4 as resumable, deterministic shard jobs (Chapter 45).

Every stage is a function ``fn(in_path, out_path) -> counters`` run through ``shard_runner.run``:
one input shard, one same-named output shard written atomically, counters summed over shards into
the funnel. A stage reads its configuration from ``stage.json`` in its output directory, which is
content-addressed (``config.stage_dir``), so

* a rerun with the same configuration and inputs finds every shard in the manifest and does nothing;
* a changed threshold creates a new directory instead of silently overwriting the old one;
* a crash loses the shard that was in flight and nothing else.

Global stages (exact dedup, near dedup) are two shard passes around one small global step: pass 1
emits per-shard summaries (line hashes, signatures), the global step reduces them to a decision
(duplicate line hashes, documents to drop) stored next to ``stage.json``, and pass 2 applies the
decision shard by shard. The global step is the only place where the whole corpus is in one
process, and what it holds is hashes and signatures, never text.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
for p in (KIT, os.path.dirname(os.path.dirname(KIT))):
    if p not in sys.path:
        sys.path.insert(0, p)

import shard_runner  # noqa: E402  (code/projects/p4_data/shard_runner.py)
from llmsys.datapipe.classify import HashedNgramClassifier, NaiveBayesLangID  # noqa: E402
from llmsys.datapipe.decontam import Decontaminator  # noqa: E402
from llmsys.datapipe.dedup import hash64  # noqa: E402
from llmsys.datapipe.normalize import normalize_for_dedup  # noqa: E402
from llmsys.datapipe.pii import PIIMasker  # noqa: E402
from llmsys.datapipe.pipeline import Doc  # noqa: E402
from llmsys.datapipe.rules import RuleConfig, gopher_report  # noqa: E402
from llmsys.datapipe.synth import language_training_set, quality_training_set  # noqa: E402
from llmsys.ops import minhash as M  # noqa: E402

from . import dedup as D  # noqa: E402
from .config import content_hash, stage_dir  # noqa: E402
from .wet import iter_wet_docs  # noqa: E402

PKG = __name__.rsplit(".", 1)[0]          # the package this module was imported as


# ----------------------------------------------------------------------------- shard I/O helpers
def read_docs(path: str) -> List[Doc]:
    with open(path, encoding="utf-8") as f:
        return [Doc(**json.loads(line)) for line in f]


def write_docs(docs: Iterable[Doc], out_path: str) -> int:
    """Write JSONL atomically (temp file + rename): a crash never leaves a half-written shard."""
    tmp = out_path + ".tmp"
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"url": d.url, "text": d.text, "meta": d.meta}, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, out_path)
    return n


_STAGE_CACHE: Dict[str, Dict] = {}


def stage_config(out_path: str) -> Dict:
    """The ``stage.json`` of the directory ``out_path`` is written to (cached per process)."""
    path = os.path.join(os.path.dirname(os.path.abspath(out_path)), "stage.json")
    if path not in _STAGE_CACHE:
        with open(path) as f:
            _STAGE_CACHE[path] = json.load(f)
    return _STAGE_CACHE[path]


def _maybe_fail(out_path: str) -> None:
    """Test hook: ``P4_FAIL_AFTER=k`` kills the job (an exception) after k shards of a stage dir, so
    the acceptance tests can check that a restart resumes from the manifest."""
    k = int(os.environ.get("P4_FAIL_AFTER", "0"))
    if k:
        done = shard_runner.load_manifest(os.path.dirname(out_path))
        if len(done) >= k:
            raise RuntimeError(f"P4_FAIL_AFTER: simulated crash after {k} shards")


# ----------------------------------------------------------------------------- stage 1: extract
def extract_shard(in_path: str, out_path: str) -> Dict[str, int]:
    _maybe_fail(out_path)
    docs = list(iter_wet_docs(in_path))
    kept = [d for d in docs if d.text.strip()]
    for d in kept:
        d.meta = {"chars": len(d.text), "src": os.path.basename(in_path)}
    write_docs(kept, out_path)
    return {"records": len(docs), "nonempty": len(kept), "chars": sum(len(d.text) for d in kept)}


# ----------------------------------------------------------------------------- stage 2: filter
_MODELS: Dict[str, object] = {}


class Models:
    """Per-process classifiers, trained or loaded once (the `load once per worker` rule)."""

    def __init__(self, filters: Dict):
        self.cfg = filters
        self.lang_model = None
        if filters.get("lang_model"):
            import fasttext  # pragma: no cover - production path
            self.lang_model = fasttext.load_model(filters["lang_model"])
        else:
            self.nb = NaiveBayesLangID().fit(language_training_set(200, seed=1))
        self.quality_model = None
        if filters.get("quality"):
            if filters.get("quality_model"):
                import fasttext  # pragma: no cover - production path
                self.quality_model = fasttext.load_model(filters["quality_model"])
            else:
                data = quality_training_set(400, seed=2)
                self.hq = HashedNgramClassifier(buckets=2 ** 16, seed=0)
                self.hq.fit([t for t, _ in data], [y for _, y in data], epochs=4)
        self.pii = PIIMasker()
        self.rules = RuleConfig(min_words=filters["min_words"], max_words=filters["max_words"])

    def language(self, text: str) -> Tuple[str, float]:
        if self.lang_model is not None:  # pragma: no cover
            labels, probs = self.lang_model.predict(text.replace("\n", " ")[:4000])
            return labels[0].replace("__label__", ""), float(probs[0])
        return self.nb.predict(text[:2000])

    def quality(self, text: str) -> float:
        if self.quality_model is not None:  # pragma: no cover
            labels, probs = self.quality_model.predict(text.replace("\n", " ")[:4000])
            p = float(probs[0])
            return p if labels[0].endswith("good") else 1.0 - p
        return self.hq.predict_proba(text)


def get_models(filters: Dict) -> Models:
    key = content_hash(filters)
    if key not in _MODELS:
        _MODELS[key] = Models(filters)
    return _MODELS[key]   # type: ignore[return-value]


def filter_doc(d: Doc, m: Models, f: Dict) -> Tuple[Optional[Doc], str]:
    """Apply the per-document filters in cost order; return (doc or None, reason)."""
    lang, p = m.language(d.text)
    d.meta["lang"], d.meta["lang_p"] = lang, round(p, 4)
    if f.get("language") and (lang != f["language"] or p < f["lang_threshold"]):
        return None, "language"
    if f.get("pii"):
        r = m.pii.mask(d.text)
        d.text = r.text
        d.meta["pii"] = r.counts
    if f.get("rules"):
        rep = gopher_report(d.text, m.rules)
        d.meta["rules_failed"] = rep.failed
        if not rep.passed:
            return None, "rule_" + rep.failed[0]
    if f.get("quality"):
        q = m.quality(d.text)
        d.meta["quality"] = round(q, 4)
        if q < f["quality_threshold"]:
            return None, "quality"
    return d, "kept"


def filter_shard(in_path: str, out_path: str) -> Dict[str, int]:
    _maybe_fail(out_path)
    f = stage_config(out_path)["config"]["filters"]
    m = get_models(f)
    counters: Counter = Counter()
    kept = []
    for d in read_docs(in_path):
        counters["seen"] += 1
        out, reason = filter_doc(d, m, f)
        counters[reason] += 1
        if out is not None:
            kept.append(out)
            for k, v in out.meta.get("pii", {}).items():
                counters[f"pii_{k}"] += v
    write_docs(kept, out_path)
    counters["kept"] = len(kept)
    return dict(counters)


# ----------------------------------------------------------------------------- stage 3: exact dedup
def line_hashes_shard(in_path: str, out_path: str) -> Dict[str, int]:
    """Pass 1: per-shard line-hash counts and normalised document hashes (as .npz, not text)."""
    _maybe_fail(out_path)
    docs = read_docs(in_path)
    c: Counter = Counter()
    for d in docs:
        c.update(hash64(ln) for ln in d.text.splitlines())
    doc_h = np.array([hash64(normalize_for_dedup(d.text)) for d in docs], dtype=np.uint64)
    keys = np.fromiter(c.keys(), dtype=np.uint64, count=len(c))
    vals = np.fromiter(c.values(), dtype=np.int64, count=len(c))
    tmp = out_path + ".tmp.npz"
    np.savez(tmp, keys=keys, vals=vals, doc_hashes=doc_h)
    os.replace(tmp, out_path)
    return {"docs": len(docs), "lines": int(vals.sum()), "distinct_lines": len(c)}


def exact_reduce(pass1_dir: str, shard_order: Sequence[str], out_dir: str) -> Dict[str, int]:
    """Global step: duplicated line hashes across the corpus; first occurrence of each document."""
    total: Counter = Counter()
    seen_docs: Dict[int, Tuple[str, int]] = {}
    drop: Dict[str, List[int]] = {}
    for name in shard_order:
        z = np.load(os.path.join(pass1_dir, name + ".npz"))
        total.update(dict(zip(z["keys"].tolist(), z["vals"].tolist())))
        for pos, h in enumerate(z["doc_hashes"].tolist()):
            if h in seen_docs:
                drop.setdefault(name, []).append(pos)
            else:
                seen_docs[h] = (name, pos)
    dup = np.array(sorted(h for h, n in total.items() if n > 1), dtype=np.uint64)
    np.save(os.path.join(out_dir, "dup_lines.npy"), dup)
    with open(os.path.join(out_dir, "drop_docs.json"), "w") as f:
        json.dump(drop, f)
    return {"distinct_lines": len(total), "duplicated_lines": int(len(dup)),
            "exact_duplicate_docs": sum(len(v) for v in drop.values())}


def exact_apply_shard(in_path: str, out_path: str) -> Dict[str, int]:
    """Pass 2: drop duplicated lines (``unique_only``) or later repeats (``keep_first`` within the
    shard) and the exact-duplicate documents chosen by the global step."""
    _maybe_fail(out_path)
    st = stage_config(out_path)
    d = os.path.dirname(os.path.abspath(out_path))
    dup = np.load(os.path.join(d, "dup_lines.npy"))
    with open(os.path.join(d, "drop_docs.json")) as f:
        drop = set(json.load(f).get(os.path.basename(in_path)[:-len(".jsonl")], []))
    policy = st["config"]["exact_dedup"]["line_policy"]
    dup_set = set(dup.tolist())
    seen = set()
    kept, c = [], Counter()
    for pos, doc in enumerate(read_docs(in_path)):
        c["seen"] += 1
        if pos in drop:
            c["exact_duplicate"] += 1
            continue
        lines = []
        for ln in doc.text.splitlines():
            h = hash64(ln)
            if policy == "unique_only" and h in dup_set:
                c["lines_removed"] += 1
                continue
            if policy == "keep_first":
                if h in seen:
                    c["lines_removed"] += 1
                    continue
                seen.add(h)
            lines.append(ln)
        text = "\n".join(lines).strip()
        if not text:
            c["emptied"] += 1
            continue
        doc.text = text
        kept.append(doc)
    write_docs(kept, out_path)
    c["kept"] = len(kept)
    return dict(c)


# ----------------------------------------------------------------------------- stage 4: near dedup
def signatures_shard(in_path: str, out_path: str) -> Dict[str, int]:
    """Pass 1 (map): MinHash signatures and band keys of one shard, through llmsys.ops.minhash."""
    _maybe_fail(out_path)
    nd = stage_config(out_path)["config"]["near_dedup"]
    texts = [d.text for d in read_docs(in_path)]
    t0 = time.perf_counter()
    sigs = M.signatures(texts, nd["num_perm"], nd["ngram"], nd.get("seed", 0), backend=nd.get("backend"))
    keys = M.band_keys(sigs, nd["bands"], backend=nd.get("backend"))
    tmp = out_path + ".tmp.npz"
    np.savez(tmp, sigs=sigs.numpy(), keys=keys.numpy())
    os.replace(tmp, out_path)
    return {"docs": len(texts), "signature_seconds": time.perf_counter() - t0}


def near_reduce(pass1_dir: str, filtered_dir: str, shard_order: Sequence[str], out_dir: str,
                nd: Dict) -> Dict[str, int]:
    """Global step (shuffle + reduce): candidates from band keys, verification, clustering, and a
    per-shard drop list. Text is read only for the documents that are candidates."""
    import torch
    sig_parts, key_parts, index = [], [], []
    for name in shard_order:
        z = np.load(os.path.join(pass1_dir, name + ".npz"))
        sig_parts.append(z["sigs"])
        key_parts.append(z["keys"])
        index.extend((name, pos) for pos in range(len(z["sigs"])))
    sigs = torch.from_numpy(np.concatenate(sig_parts)) if sig_parts else torch.zeros((0, nd["num_perm"]), dtype=torch.int64)
    keys = torch.from_numpy(np.concatenate(key_parts)) if key_parts else torch.zeros((0, nd["bands"]), dtype=torch.int64)
    cands = M.candidates_from_band_keys(keys, max_bucket=nd.get("max_bucket", 0))
    texts: Dict[int, str] = {}
    if nd["verify"] == "exact" and cands:
        need: Dict[str, List[int]] = {}
        for i, j in cands:
            for x in (i, j):
                need.setdefault(index[x][0], []).append(x)
        for name, ids in need.items():
            docs = read_docs(os.path.join(filtered_dir, name + ".jsonl"))
            for x in ids:
                texts[x] = docs[index[x][1]].text
    confirmed = []
    cache: Dict[int, set] = {}
    for i, j in sorted(cands):
        if nd["verify"] == "exact":
            for x in (i, j):
                if x not in cache:
                    cache[x] = D.shingle_set(texts[x], nd["ngram"])
            ok = D.jaccard(cache[i], cache[j]) >= nd["threshold"]
        elif nd["verify"] == "estimate":
            ok = M.estimate_jaccard(sigs, i, j) >= nd["threshold"]
        else:
            ok = True
        if ok:
            confirmed.append((i, j))
    drop, clusters = D.cluster_pairs(len(index), confirmed)
    per_shard: Dict[str, List[int]] = {}
    for x in drop:
        per_shard.setdefault(index[x][0], []).append(index[x][1])
    with open(os.path.join(out_dir, "drop_docs.json"), "w") as f:
        json.dump(per_shard, f)
    with open(os.path.join(out_dir, "clusters.json"), "w") as f:
        json.dump([[list(index[x]) for x in c] for c in clusters], f)
    return {"docs": len(index), "candidates": len(cands), "confirmed": len(confirmed),
            "clusters": len(clusters), "near_duplicates": len(drop)}


def drop_apply_shard(in_path: str, out_path: str) -> Dict[str, int]:
    """Pass 2: drop the documents listed for this shard in ``drop_docs.json``."""
    _maybe_fail(out_path)
    d = os.path.dirname(os.path.abspath(out_path))
    with open(os.path.join(d, "drop_docs.json")) as f:
        drop = set(json.load(f).get(os.path.basename(in_path)[:-len(".jsonl")], []))
    docs = read_docs(in_path)
    kept = [doc for pos, doc in enumerate(docs) if pos not in drop]
    write_docs(kept, out_path)
    return {"seen": len(docs), "dropped": len(docs) - len(kept), "kept": len(kept)}


# ----------------------------------------------------------------------------- stage 5: decontam
_DECON: Dict[str, Decontaminator] = {}


def read_eval_texts(paths: Sequence[str]) -> List[str]:
    out = []
    for p in paths:
        for q in sorted(glob.glob(p)) or [p]:
            if not os.path.exists(q):
                continue
            with open(q, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        out.append(r["text"] if isinstance(r, dict) else str(r))
    return out


def get_decontaminator(paths: Sequence[str], n: int) -> Decontaminator:
    key = content_hash(list(paths), n)
    if key not in _DECON:
        _DECON[key] = Decontaminator(read_eval_texts(paths), n=n)
    return _DECON[key]


def decontam_shard(in_path: str, out_path: str) -> Dict[str, int]:
    _maybe_fail(out_path)
    st = stage_config(out_path)
    dc = st["config"]["decontam"]
    dec = get_decontaminator(st["eval_paths"], dc["ngram"])
    kept, c = [], Counter()
    for doc in read_docs(in_path):
        c["seen"] += 1
        hits = dec.hits(doc.text)
        if hits >= dc["min_hits"]:
            c["contaminated"] += 1
            c["ngram_hits"] += hits
        else:
            kept.append(doc)
    write_docs(kept, out_path)
    c["kept"] = len(kept)
    return dict(c)


# ----------------------------------------------------------------------------- stage 6: tokenise
class ByteTokenizer:
    """ids = UTF-8 bytes, EOT = 256, vocab 512: the tokenizer of the ``tiny`` recipe."""
    vocab_size, eot = 512, 256

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))


class GPT2Tokenizer:
    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size, self.eot = self.enc.n_vocab, self.enc.eot_token

    def encode(self, text: str) -> List[int]:
        return self.enc.encode_ordinary(text)


_TOK: Dict[str, object] = {}


def get_tokenizer(name: str):
    if name not in _TOK:
        _TOK[name] = GPT2Tokenizer() if name == "gpt2" else ByteTokenizer()
    return _TOK[name]


def tokenize_texts(texts: Iterable[str], tok, out_path: str) -> int:
    dtype = np.uint16 if tok.vocab_size <= 2 ** 16 else np.uint32
    n = 0
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        for t in texts:
            ids = tok.encode(t) + [tok.eot]
            np.asarray(ids, dtype=dtype).tofile(f)
            n += len(ids)
    os.replace(tmp, out_path)
    return n


def tokenize_shard(in_path: str, out_path: str) -> Dict[str, int]:
    _maybe_fail(out_path)
    tok = get_tokenizer(stage_config(out_path)["config"]["tokenizer"])
    docs = read_docs(in_path)
    n = tokenize_texts((d.text for d in docs), tok, out_path)
    return {"docs": len(docs), "tokens": n}


def concat_bins(parts: Sequence[str], out_path: str) -> int:
    tmp = out_path + ".tmp"
    n = 0
    with open(tmp, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                while chunk := f.read(1 << 24):
                    out.write(chunk)
                    n += len(chunk)
    os.replace(tmp, out_path)
    return n


# ----------------------------------------------------------------------------- orchestration
def _prepare(out_dir: str, stage: str, cfg: Dict, extra: Optional[Dict] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stage.json")
    if not os.path.exists(path):
        with open(path + ".tmp", "w") as f:
            json.dump({"stage": stage, "config": cfg, **(extra or {})}, f, indent=1, sort_keys=True)
        os.replace(path + ".tmp", path)


def _run_stage(fn_name: str, inputs: Sequence[str], out_dir: str, workers: int, log) -> Counter:
    total, _, wall = shard_runner.run(f"{PKG}.stages:{fn_name}", list(inputs), out_dir, workers,
                                      log=lambda s: None)
    total["seconds_wall"] = round(wall, 3)
    log(f"  {fn_name:<22} {dict(total)}")
    return total


def _outputs(inputs: Sequence[str], out_dir: str, suffix: str = ".jsonl") -> List[str]:
    return [shard_runner.output_path(p, out_dir, suffix) for p in inputs]


def _names(inputs: Sequence[str]) -> List[str]:
    return [os.path.basename(shard_runner.output_path(p, "", ""))[1:] if shard_runner.output_path(p, "", "").startswith("/")
            else os.path.basename(shard_runner.output_path(p, "", "")) for p in inputs]


def corpus_hash(paths: Sequence[str]) -> str:
    """SHA-256 over the sorted output files: the identity of a corpus version."""
    h = hashlib.sha256()
    for p in sorted(paths, key=os.path.basename):
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


def run_pipeline(cfg: Dict, inputs: Sequence[str], root: str, workers: Optional[int] = None,
                 log: Callable[[str], None] = print, eval_paths: Optional[Sequence[str]] = None) -> Dict:
    """Run every enabled stage over ``inputs`` (WET/WARC shards) under ``root``. Returns the funnel:
    per-stage counters, the final shard paths, the token file and the corpus hash."""
    workers = cfg.get("workers", 1) if workers is None else workers
    funnel: Dict[str, Dict] = {}
    t_start = time.perf_counter()
    current = list(inputs)

    d = stage_dir(root, "01_extract", {}, current)
    _prepare(d, "extract", cfg)
    funnel["extract"] = _run_stage("extract_shard", current, d, workers, log)
    current = _outputs(current, d)

    d = stage_dir(root, "02_filter", cfg["filters"], current)
    _prepare(d, "filter", cfg)
    funnel["filter"] = _run_stage("filter_shard", current, d, workers, log)
    current = _outputs(current, d)

    if cfg["exact_dedup"]["enabled"]:
        d1 = stage_dir(root, "03_exact_pass1", cfg["exact_dedup"], current)
        _prepare(d1, "exact_pass1", cfg)
        funnel["exact_pass1"] = _run_stage("line_hashes_shard", current, d1, workers, log)
        names = [os.path.basename(p)[:-len(".jsonl")] for p in current]
        d2 = stage_dir(root, "03_exact_pass2", cfg["exact_dedup"], current)
        _prepare(d2, "exact_pass2", cfg)
        if not os.path.exists(os.path.join(d2, "drop_docs.json")):
            exact_reduce(d1, names, d2)
        funnel["exact_dedup"] = _run_stage("exact_apply_shard", current, d2, workers, log)
        current = _outputs(current, d2)

    if cfg["near_dedup"]["enabled"]:
        nd = cfg["near_dedup"]
        d1 = stage_dir(root, "04_near_pass1", nd, current)
        _prepare(d1, "near_pass1", cfg)
        funnel["near_pass1"] = _run_stage("signatures_shard", current, d1, workers, log)
        names = [os.path.basename(p)[:-len(".jsonl")] for p in current]
        d2 = stage_dir(root, "04_near_pass2", nd, current)
        _prepare(d2, "near_pass2", cfg)
        if not os.path.exists(os.path.join(d2, "drop_docs.json")):
            red = near_reduce(d1, os.path.dirname(current[0]) if current else root, names, d2, nd)
            with open(os.path.join(d2, "reduce.json"), "w") as f:
                json.dump(red, f)
        with open(os.path.join(d2, "reduce.json")) as f:
            funnel["near_reduce"] = json.load(f)
        funnel["near_dedup"] = _run_stage("drop_apply_shard", current, d2, workers, log)
        current = _outputs(current, d2)

    if cfg["decontam"]["enabled"]:
        eval_paths = list(eval_paths or [])
        if not eval_paths:
            eval_paths = list(cfg["data"]["benchmark_jsonl"]) + [cfg["data"]["target_jsonl"]]
        eval_paths = [p for p in eval_paths if os.path.exists(p) or glob.glob(p)]
        d = stage_dir(root, "05_decontam", {**cfg["decontam"], "eval": eval_paths}, current)
        _prepare(d, "decontam", cfg, {"eval_paths": eval_paths})
        funnel["decontam"] = _run_stage("decontam_shard", current, d, workers, log)
        current = _outputs(current, d)

    final_docs = list(current)
    d = stage_dir(root, "06_tokens", {"tokenizer": cfg["tokenizer"]}, current)
    _prepare(d, "tokenize", cfg)
    funnel["tokenize"] = _run_stage("tokenize_shard", current, d, workers, log)
    bins = _outputs(current, d, ".bin")
    train_bin = os.path.join(d, "train.bin")
    if not os.path.exists(train_bin):
        concat_bins(bins, train_bin)

    result = {"name": cfg["name"], "funnel": {k: dict(v) for k, v in funnel.items()},
              "final_docs": final_docs, "train_bin": train_bin,
              "tokens": int(funnel["tokenize"].get("tokens", 0)),
              "corpus_hash": corpus_hash(final_docs), "seconds": round(time.perf_counter() - t_start, 3)}
    with open(os.path.join(root, f"funnel-{cfg['name']}.json"), "w") as f:
        json.dump(result, f, indent=1)
    return result


def tokenize_target(target_jsonl: str, tokenizer: str, out_path: str) -> int:
    """The held-out target set as a token file for ``train_on_bin.py --val-bin``."""
    tok = get_tokenizer(tokenizer)
    with open(target_jsonl, encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f if line.strip()]
    return tokenize_texts(texts, tok, out_path)


def funnel_table(funnel: Dict[str, Dict]) -> str:
    rows = [f"{'stage':<14}{'seen':>9}{'kept':>9}{'drop%':>8}   details"]
    for name, c in funnel.items():
        seen = c.get("seen", c.get("records", c.get("docs", 0)))
        kept = c.get("kept", c.get("nonempty", seen))
        drop = 100 * (seen - kept) / seen if seen else 0.0
        rest = {k: v for k, v in c.items() if k not in ("seen", "kept", "records", "nonempty", "docs")}
        rows.append(f"{name:<14}{seen:>9}{kept:>9}{drop:>8.1f}   {rest}")
    return "\n".join(rows)
