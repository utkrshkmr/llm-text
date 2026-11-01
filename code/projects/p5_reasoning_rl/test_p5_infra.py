import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_check  # noqa: E402
import run_matrix  # noqa: E402
import seeds_ci  # noqa: E402


def test_expand_grid_names_and_count():
    grid = {"script": "x.py", "fixed": {"lr": 1e-5},
            "vary": {"variant": ["a", "b"], "seed": [0, 1, 2]}}
    runs = run_matrix.expand(grid)
    assert len(runs) == 6
    assert runs[0]["run_group"] == "variant=a" and runs[0]["run_name"] == "variant=a-seed=0"
    cmd = run_matrix.to_command("x.py", runs[-1])
    assert "--variant b" in cmd and "--seed 2" in cmd and "--lr 1e-05" in cmd


def test_seed_aggregation(tmp_path):
    for s, vals in enumerate([[0.1, 0.3], [0.2, 0.5], [0.3, 0.4]]):
        p = tmp_path / f"variant=a-seed={s}.jsonl"
        p.write_text("\n".join(json.dumps({"step": i * 10, "val_reward": v}) for i, v in enumerate(vals)))
    out = seeds_ci.aggregate([str(p) for p in tmp_path.iterdir()], "val_reward")
    rows = out["variant=a"]
    assert [r["step"] for r in rows] == [0, 10]
    assert abs(rows[1]["mean"] - 0.4) < 1e-9 and rows[1]["n"] == 3
    assert abs(rows[1]["hi"] - (0.4 + 1.96 * 0.1 / math.sqrt(3))) < 1e-9


def test_env_check_reports_missing_gpus():
    r = env_check.check(need_gpus=99, min_disk_gb=0.0)
    assert any("GPUs" in p for p in r["problems"])
    assert "torch" in r
