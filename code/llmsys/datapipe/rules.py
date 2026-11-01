"""Gopher-style heuristic quality rules with a per-rule report.

Thresholds follow the MassiveWeb filters described by Rae et al. (2021, Appendix A); all
of them are configurable because the right values depend on language and domain.
Returning *which* rule fired (not just a boolean) is what makes a filter debuggable and
lets the pipeline report how much each rule removes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

STOP_WORDS = frozenset({"the", "be", "to", "of", "and", "that", "have", "with"})


@dataclass
class RuleConfig:
    min_words: int = 50
    max_words: int = 100_000
    min_mean_word_len: float = 3.0
    max_mean_word_len: float = 10.0
    max_symbol_word_ratio: float = 0.1       # ('#' + '...') per word
    max_bullet_line_frac: float = 0.9
    max_ellipsis_line_frac: float = 0.3
    min_alpha_word_frac: float = 0.8
    min_stop_words: int = 2
    max_dup_line_frac: float = 0.3
    max_top_bigram_char_frac: float = 0.2


@dataclass
class RuleReport:
    passed: bool
    failed: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)


def _top_ngram_char_frac(words: List[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    gram, count = grams.most_common(1)[0]
    if count < 2:
        return 0.0
    total_chars = sum(len(w) for w in words)
    return count * sum(len(w) for w in gram) / max(1, total_chars)


def gopher_report(text: str, cfg: RuleConfig = RuleConfig()) -> RuleReport:
    words = text.split()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n = len(words)
    s: Dict[str, float] = {"words": n}
    failed: List[str] = []
    if not (cfg.min_words <= n <= cfg.max_words):
        failed.append("word_count")
    mean_len = sum(len(w) for w in words) / n if n else 0.0
    s["mean_word_len"] = mean_len
    if not (cfg.min_mean_word_len <= mean_len <= cfg.max_mean_word_len):
        failed.append("mean_word_len")
    sym = text.count("#") + text.count("...") + text.count("…")
    s["symbol_ratio"] = sym / n if n else 0.0
    if s["symbol_ratio"] > cfg.max_symbol_word_ratio:
        failed.append("symbol_ratio")
    if lines:
        bullets = sum(ln.lstrip().startswith(("-", "*", "•")) for ln in lines)
        ell = sum(ln.rstrip().endswith(("...", "…")) for ln in lines)
        s["bullet_frac"], s["ellipsis_frac"] = bullets / len(lines), ell / len(lines)
        s["dup_line_frac"] = 1 - len(set(lines)) / len(lines)
    else:
        s["bullet_frac"] = s["ellipsis_frac"] = s["dup_line_frac"] = 0.0
    if s["bullet_frac"] > cfg.max_bullet_line_frac:
        failed.append("bullet_lines")
    if s["ellipsis_frac"] > cfg.max_ellipsis_line_frac:
        failed.append("ellipsis_lines")
    if s["dup_line_frac"] > cfg.max_dup_line_frac:
        failed.append("dup_lines")
    alpha = sum(any(c.isalpha() for c in w) for w in words)
    s["alpha_frac"] = alpha / n if n else 0.0
    if s["alpha_frac"] < cfg.min_alpha_word_frac:
        failed.append("alpha_words")
    lower = [w.lower().strip(".,;:!?\"'()") for w in words]
    s["stop_words"] = len(STOP_WORDS.intersection(lower))
    if s["stop_words"] < cfg.min_stop_words:
        failed.append("stop_words")
    s["top_bigram_frac"] = _top_ngram_char_frac(lower, 2)
    if s["top_bigram_frac"] > cfg.max_top_bigram_char_frac:
        failed.append("repetition")
    return RuleReport(not failed, failed, s)
