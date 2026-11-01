"""Turn a sweep's JSONL into writeup-ready tables (Markdown or LaTeX).

    python projects/p2_systems/tables.py runs.jsonl --rows size ctx --cols dtype --value fwd_ms
    python projects/p2_systems/tables.py runs.jsonl --rows size --cols ctx --value peak_GiB --latex

Cells show the requested metric, or the run status ("oom", "error", "timeout") when the run failed.
Write the generated table to a file that your writeup \\input{}s, so no number is ever hand-copied.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Sequence


def load(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def cell(rec: dict, value: str, fmt: str) -> str:
    if rec.get("status") != "ok":
        return rec.get("status", "?")
    v = rec.get("result", {}).get(value)
    if isinstance(v, float):
        return fmt.format(v)
    return "-" if v is None else str(v)


def pivot(records: Sequence[dict], rows: Sequence[str], cols: Sequence[str], value: str,
          fmt: str = "{:.3g}") -> Dict[str, object]:
    """Return dict(row_keys, col_keys, cells) where cells[(r, c)] is a formatted string."""
    rk = lambda r: tuple(r["setting"].get(k) for k in rows)
    ck = lambda r: tuple(r["setting"].get(k) for k in cols)
    row_keys = sorted({rk(r) for r in records}, key=str)
    col_keys = sorted({ck(r) for r in records}, key=str)
    cells = {(rk(r), ck(r)): cell(r, value, fmt) for r in records}
    return dict(row_keys=row_keys, col_keys=col_keys, cells=cells)


def render(t: Dict[str, object], rows: Sequence[str], cols: Sequence[str], latex: bool) -> str:
    head = ["/".join(rows)] + ["/".join(map(str, c)) for c in t["col_keys"]]
    body = [[" ".join(map(str, r))] + [t["cells"].get((r, c), "") for c in t["col_keys"]]
            for r in t["row_keys"]]
    if latex:
        esc = lambda s: str(s).replace("_", r"\_")
        out = [r"\begin{tabular}{l" + "r" * len(t["col_keys"]) + "}", r"\toprule",
               " & ".join(map(esc, head)) + r" \\", r"\midrule"]
        out += [" & ".join(map(esc, row)) + r" \\" for row in body]
        return "\n".join(out + [r"\bottomrule", r"\end{tabular}"])
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(out + ["| " + " | ".join(row) + " |" for row in body])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    p.add_argument("--rows", nargs="+", required=True)
    p.add_argument("--cols", nargs="+", required=True)
    p.add_argument("--value", required=True)
    p.add_argument("--fmt", default="{:.3g}")
    p.add_argument("--latex", action="store_true")
    a = p.parse_args()
    t = pivot(load(a.jsonl), a.rows, a.cols, a.value, a.fmt)
    print(render(t, a.rows, a.cols, a.latex))


if __name__ == "__main__":
    main()
