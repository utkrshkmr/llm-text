"""A deterministic math-answer verifier (Chapter 37).

Pipeline:  model text --extract--> candidate --normalize--> plain expression
           --parse (SymPy)--> symbolic object --compare--> True / False

Every stage is a pure function so it can be unit-tested, cached, and run in a
sandboxed worker process with a timeout (see ``grade_answer_safe``).
"""
from __future__ import annotations

import re
from typing import List, Optional

# --------------------------------------------------------------------------- prompts
MATH_PROMPT = (
    "You are a careful math assistant.\n"
    "Solve the problem. Put only the final result on the last line in the form "
    "\\boxed{{ANSWER}}.\n\n"
    "Problem:\n{problem}\n\nSolution:"
)


def render_math_prompt(problem: str, template: str = MATH_PROMPT) -> str:
    return template.format(problem=problem)


# --------------------------------------------------------------------------- extraction
def last_boxed(text: str, macro: str = r"\boxed") -> Optional[str]:
    """Content of the *last* ``\\boxed{...}`` in ``text`` (handles nested braces).

    Returns None if there is no box or its braces are unbalanced.
    """
    if not text:
        return None
    start = text.rfind(macro)
    if start < 0:
        return None
    i = start + len(macro)
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1: j]
        j += 1
    return None  # unbalanced


NUMBER_RE = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def extract_final_candidate(text: str, fallback: str = "number_then_full") -> str:
    """Pick the string we will grade.

    Preference order: last boxed expression; otherwise depends on ``fallback``:
      * ``"number_then_full"``: last number in the text, else the whole text
      * ``"number_only"``     : last number, else ""
      * ``"none"``            : "" (only boxed answers count)
    """
    boxed = last_boxed(text or "")
    if boxed is not None:
        return boxed.strip().strip("$ ").strip()
    if fallback == "none" or not text:
        return ""
    nums = NUMBER_RE.findall(text)
    if nums:
        return nums[-1]
    return text.strip() if fallback == "number_then_full" else ""


# --------------------------------------------------------------------------- normalization
_SIMPLE_SUBS = [
    (r"\\left\s*", ""), (r"\\right\s*", ""),
    (r"\\[,;:!]", ""), (r"\\quad|\\qquad", ""),
    (r"\\[dt]frac", r"\\frac"),
    (r"\\cdot|\\times|\u00d7|\u00b7", "*"),
    (r"\\div", "/"),
    (r"\\pi", "pi"),
    (r"\\infty", "oo"),
    (r"\\%|%", ""),
    (r"\^\s*\{?\\circ\}?|\u00b0|\\degree", ""),
    (r"\\\$|\$", ""),
    (r"\\displaystyle", ""),
]
_SUPERSCRIPTS = str.maketrans("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079",
                              "0123456789")
_SPECIAL_TOKENS = re.compile(r"<\|[^>]*?\|>")


def _brace_arg(s: str, i: int):
    """Return (content, end_index) of a {...} group starting at s[i] == '{'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1: j], j + 1
    raise ValueError("unbalanced braces")


def _rewrite_macro(s: str, macro: str, nargs: int, fmt) -> str:
    """Rewrite \\macro{a}{b} -> fmt(a, b), innermost-safe via repeated passes."""
    for _ in range(50):
        k = s.find(macro)
        if k < 0:
            return s
        i = k + len(macro)
        args = []
        try:
            for _ in range(nargs):
                while i < len(s) and s[i] == " ":
                    i += 1
                if i < len(s) and s[i] == "{":
                    a, i = _brace_arg(s, i)
                elif i < len(s):          # \frac12 style single-char args
                    a, i = s[i], i + 1
                else:
                    return s
                args.append(a)
        except ValueError:
            return s
        s = s[:k] + fmt(*args) + s[i:]
    return s


def normalize_text(text: str) -> str:
    """Rewrite a LaTeX-ish answer into a plain expression SymPy can parse.

    Examples::

        "\\dfrac{14}{3}"        -> "(14)/(3)"
        "\\text{(B)}"           -> "(B)"
        "x = 5"                 -> "5"
        "45^\\circ"             -> "45"
        "1{,}000" / "1,000"     -> "1000"
        "2\\sqrt{3}"            -> "2sqrt(3)"
    """
    if text is None:
        return ""
    s = _SPECIAL_TOKENS.sub("", str(text)).strip()
    s = s.replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")
    # unwrap \text{...}, \mathrm{...}, \mbox{...}, \boxed{...}
    for m in (r"\text", r"\textbf", r"\mathrm", r"\mathbf", r"\mbox", r"\boxed"):
        s = _rewrite_macro(s, m, 1, lambda a: a)
    for pat, rep in _SIMPLE_SUBS:
        s = re.sub(pat, rep, s)
    s = s.replace("{,}", ",")
    # multiple-choice style prefix "B. 42" / "(C) 7" -> keep the value
    m = re.match(r"^\(?[A-Ea-e]\)?\s*[.:)]\s*(.+)$", s)
    if m:
        s = m.group(1)
    # leading "x =" / "y=" (single variable assignment)
    m = re.match(r"^[A-Za-z]\s*=\s*(.+)$", s)
    if m:
        s = m.group(1)
    s = _rewrite_macro(s, r"\frac", 2, lambda a, b: f"({a})/({b})")
    s = _rewrite_macro(s, r"\sqrt", 1, lambda a: f"sqrt({a})")
    s = s.translate(_SUPERSCRIPTS)
    s = s.replace("^", "**").replace("{", "(").replace("}", ")")
    s = s.replace("\\", "")
    # thousands separators: only when the whole thing is a single grouped number
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s.replace(" ", "")):
        s = s.replace(",", "")
    # mixed numbers "3 1/2" -> "3+1/2"
    s = re.sub(r"(?<=\d)\s+(\d+/\d+)", r"+\1", s)
    s = s.strip().rstrip(".").strip()
    s = re.sub(r"\s+", " ", s)
    return s


# --------------------------------------------------------------------------- symbolic equality
def sympy_parse(expr: str):
    """Parse a normalized string into a SymPy object, or None if it isn't math."""
    if not expr or len(expr) > 1000:
        return None
    try:
        import sympy  # noqa: F401
        from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                                parse_expr, standard_transformations)
        tr = standard_transformations + (implicit_multiplication_application, convert_xor)
        return parse_expr(expr.replace(" ", ""), transformations=tr, evaluate=True)
    except Exception:  # SympifyError, TokenError, SyntaxError, TypeError, ...
        return None


def equality_check(gt: str, pred: str, rel_tol: float = 1e-6) -> bool:
    """True if two *normalized* scalar strings denote the same value."""
    if gt == pred:
        return True
    a, b = sympy_parse(gt), sympy_parse(pred)
    if a is None or b is None:
        return False
    try:
        import sympy
        diff = sympy.simplify(a - b)
        if diff == 0:
            return True
        # numeric fallback (e.g. 0.333333 vs 1/3 at a tolerance)
        av, bv = complex(sympy.N(a)), complex(sympy.N(b))
        return abs(av - bv) <= rel_tol * max(1.0, abs(av))
    except Exception:
        return False


def split_top_level(text: str) -> Optional[List[str]]:
    """Split "(a, b)" / "[a, b]" / "a, b" on top-level commas; None if not a tuple."""
    t = text.strip()
    if len(t) >= 2 and t[0] in "([" and t[-1] in ")]":
        inner = t[1:-1]
    else:
        inner = t
    parts, depth, cur = [], 0, []
    for ch in inner:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return parts if len(parts) > 1 else None


def grade_answer(pred_text: str, gt_text: str) -> bool:
    """Grade a candidate against the reference; handles tuples/lists element-wise."""
    if not pred_text or not gt_text:
        return False
    gt, pred = normalize_text(gt_text), normalize_text(pred_text)
    if not gt or not pred:
        return False
    gt_parts, pred_parts = split_top_level(gt), split_top_level(pred)
    if gt_parts is None and pred_parts is None:
        return equality_check(gt, pred)
    if gt_parts is None or pred_parts is None or len(gt_parts) != len(pred_parts):
        return False
    # interval endpoints must agree in their bracket type
    if gt.strip()[:1] in "([" and pred.strip()[:1] in "([":
        if gt.strip()[0] != pred.strip()[0] or gt.strip()[-1] != pred.strip()[-1]:
            return False
    return all(equality_check(g, p) for g, p in zip(gt_parts, pred_parts))


def grade_response(response: str, gt_text: str, fallback: str = "number_then_full") -> bool:
    """Full pipeline: extract -> normalize -> compare."""
    return grade_answer(extract_final_candidate(response, fallback), gt_text)


# --------------------------------------------------------------------------- sandboxing
def _grade_worker(args):
    return grade_answer(*args)


def grade_many_safe(pairs, timeout_s: float = 5.0, workers: int = 2) -> List[Optional[bool]]:
    """Grade (pred, gt) pairs in worker processes; a hung SymPy call yields None.

    Symbolic simplification can blow up on adversarial strings (huge exponents,
    nested radicals). Production eval/RL reward servers isolate it like this.
    """
    from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeout
    results: List[Optional[bool]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_grade_worker, p) for p in pairs]
        for f in futs:
            try:
                results.append(f.result(timeout=timeout_s))
            except FTimeout:
                results.append(None)
    return results
