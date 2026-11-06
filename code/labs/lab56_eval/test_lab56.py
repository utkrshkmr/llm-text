import json
import os

import pytest
import torch

from _labutil import load_impl
from llmsys.qwen3 import QWEN3_CONFIG_TINY, Qwen3Model

lab = load_impl(__file__)
DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "math_sample.jsonl")


def test_last_boxed():
    assert lab.last_boxed(r"a \boxed{1} then \boxed{\frac{2}{3}}") == r"\frac{2}{3}"
    assert lab.last_boxed(r"\boxed {x^{2}}") == "x^{2}"
    assert lab.last_boxed("no box here") is None
    assert lab.last_boxed(r"\boxed{unbalanced") is None


def test_extract_fallbacks():
    assert lab.extract_final_candidate(r"so $\boxed{ 14/3 }$") == "14/3"
    assert lab.extract_final_candidate("we get 3 then 7.5 apples") == "7.5"
    assert lab.extract_final_candidate("no numbers", fallback="number_only") == ""
    assert lab.extract_final_candidate("no numbers") == "no numbers"
    assert lab.extract_final_candidate("x = 4", fallback="none") == ""


@pytest.mark.parametrize("pred,gt,ok", [
    (r"\dfrac{14}{3}", "14/3", True), ("(14/3, 2/3)", "(14/3, 4/6)", True),
    ("x = 5", "5", True), ("1,000", "1000", True), (r"2\sqrt{3}", r"\sqrt{12}", True),
    ("(1, 2)", "(2, 1)", False), ("[2, 5)", "[2, 5]", False), ("10", "11", False),
    ("", "5", False), (r"\frac{\pi}{2}", r"\pi/2", True),
])
def test_grade(pred, gt, ok):
    assert lab.grade_answer(pred, gt) is ok


def test_evaluate_with_cache(tmp_path):
    exs = [json.loads(l) for l in open(DATA)][:6]
    calls = []

    def fake_generate(prompts):
        calls.append(len(prompts))
        # correct on even-indexed problems, wrong otherwise
        out = []
        for p in prompts:
            ex = next(e for e in exs if e["problem"] in p)
            i = exs.index(ex)
            out.append(f"... \\boxed{{{ex['answer']}}}" if i % 2 == 0 else "\\boxed{-999}")
        return out

    cache = tmp_path / "res.jsonl"
    r1 = lab.evaluate(exs, fake_generate, lambda q: "Q: " + q, batch_size=4, cache_path=str(cache))
    assert abs(r1["accuracy"] - 0.5) < 1e-9 and calls == [4, 2]
    assert [r["problem"] for r in r1["records"]] == [e["problem"] for e in exs]
    r2 = lab.evaluate(exs, fake_generate, lambda q: "Q: " + q, batch_size=4, cache_path=str(cache))
    assert calls == [4, 2] and r2["accuracy"] == r1["accuracy"]   # nothing regenerated


def test_wilson():
    lo, hi = lab.wilson_interval(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    lo, hi = lab.wilson_interval(0, 10)
    assert lo < 1e-12 and 0.25 < hi < 0.35


def test_mcq_letters():
    assert lab.first_choice_letter(" B") == "B"
    assert lab.first_choice_letter(" (c).") == "C"
    assert lab.first_choice_letter("Because the answer is D") == "D"
    assert lab.first_choice_letter("Because") is None


def test_predict_by_logits():
    torch.manual_seed(0)
    m = Qwen3Model(QWEN3_CONFIG_TINY).eval()
    ids = [10, 11, 12]
    letters = {"A": 1, "B": 2, "C": 3, "D": 4}
    with torch.no_grad():
        logits = m(torch.tensor([ids]))[0, -1]
    expect = max(letters, key=lambda L: float(logits[letters[L]]))
    assert lab.predict_by_logits(m, ids, letters) == expect


def test_elo_and_bt():
    votes = [("gpt", "tiny")] * 8 + [("tiny", "gpt")] * 2 + [("mid", "tiny")] * 6 + [("gpt", "mid")] * 5
    elo = lab.elo_ratings(votes)
    assert elo["gpt"] > elo["tiny"]
    assert abs(sum(elo.values()) - 3000) < 1e-6          # Elo is zero-sum
    bt = lab.bradley_terry(votes)
    assert bt["gpt"] > bt["mid"] > bt["tiny"]
    # order invariance: BT does not depend on the order of votes, Elo does
    bt2 = lab.bradley_terry(list(reversed(votes)))
    assert all(abs(bt[m] - bt2[m]) < 1e-6 for m in bt)
