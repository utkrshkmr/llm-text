import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import inspect_samples  # noqa: E402
import shard_runner  # noqa: E402


def _toy_job(in_path, out_path):
    with open(in_path) as f:
        lines = f.read().splitlines()
    kept = [ln for ln in lines if len(ln) > 3]
    with open(out_path, "w") as f:
        f.write("\n".join(kept))
    return {"seen": len(lines), "kept": len(kept)}


def test_runner_aggregates_and_resumes():
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i in range(3):
            p = os.path.join(d, f"in{i}.txt")
            with open(p, "w") as f:
                f.write("ab\nabcd\nabcdef\n")
            paths.append(p)
        out = os.path.join(d, "out")
        spec = "test_proj_a4:_toy_job"
        total, results, _ = shard_runner.run(spec, paths[:2], out, workers=1, log=lambda s: None)
        assert total == {"seen": 6, "kept": 4}
        total, results, _ = shard_runner.run(spec, paths, out, workers=1, log=lambda s: None)
        assert total == {"seen": 9, "kept": 6} and len(results) == 3
        with open(os.path.join(out, "_manifest.jsonl")) as f:
            assert len([json.loads(x) for x in f]) == 3      # earlier files were not re-run


def test_reservoir_is_uniform_size():
    s = inspect_samples.reservoir(iter(range(1000)), 25, seed=1)
    assert len(s) == 25 and len(set(s)) == 25


def test_fixed_recipe_trainer_learns_on_cpu():
    import numpy as np
    import train_on_bin
    rng = np.random.default_rng(0)
    # a learnable stream: repeating "documents" with an EOT separator (id 511)
    doc = rng.integers(0, 500, size=40)
    stream = np.concatenate([np.append(doc, 511) for _ in range(300)]).astype(np.uint16)
    with tempfile.TemporaryDirectory() as d:
        tr, va = os.path.join(d, "train.bin"), os.path.join(d, "val.bin")
        stream.tofile(tr); stream[:4000].tofile(va)
        res = train_on_bin.train("tiny", tr, va, os.path.join(d, "run"), eval_every=30, log=lambda s: None)
        logs = [json.loads(x) for x in open(os.path.join(d, "run", "log.jsonl"))]
        assert len(logs) == 2 and logs[-1]["val_loss"] < logs[0]["val_loss"] < 6.3
        assert os.path.exists(os.path.join(d, "run", "best.pt")) and res["best_val_loss"] == logs[-1]["val_loss"]
