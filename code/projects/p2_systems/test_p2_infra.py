import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep  # noqa: E402
import tables  # noqa: E402


def test_expand_and_render():
    g = {"size": ["s", "m"], "ctx": [128, 512]}
    settings = list(sweep.expand(g))
    assert len(settings) == 4 and settings[0] == {"size": "s", "ctx": 128}
    assert sweep.render(["run", "--ctx", "{ctx}"], settings[1]) == ["run", "--ctx", "512"]


def test_sweep_runs_and_resumes(tmp_path):
    grid = tmp_path / "g.json"
    grid.write_text(json.dumps({"x": [1, 2]}))
    out = tmp_path / "runs.jsonl"
    cmd = [sys.executable, "-c", "import json,sys; print(json.dumps({'y': int(sys.argv[1])**2}))", "{x}"]
    sys.argv = ["sweep.py", "--grid", str(grid), "--out", str(out), "--"] + cmd
    sweep.main()
    recs = tables.load(str(out))
    assert [r["result"]["y"] for r in recs] == [1, 4]
    sweep.main()                                   # resumable: nothing is re-run
    assert len(tables.load(str(out))) == 2


def test_pivot_marks_failures():
    recs = [dict(setting=dict(size="s", dtype="fp32"), status="ok", result=dict(ms=1.234)),
            dict(setting=dict(size="s", dtype="bf16"), status="oom")]
    t = tables.pivot(recs, ["size"], ["dtype"], "ms")
    md = tables.render(t, ["size"], ["dtype"], latex=False)
    assert "oom" in md and "1.23" in md
    assert r"\toprule" in tables.render(t, ["size"], ["dtype"], latex=True)
