"""Text normalisation and shingling shared by the dedup and classification stages."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+", flags=re.UNICODE)


def strip_accents(text: str) -> str:
    """NFD-decompose and drop combining marks: 'café' -> 'cafe'."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if not unicodedata.combining(ch))


def normalize_for_dedup(text: str) -> str:
    """Aggressive normalisation used before near-duplicate detection.

    lowercase -> NFD + strip accents -> remove punctuation -> collapse whitespace.
    Two documents that differ only in case, accents, punctuation or spacing map to the
    same string, which raises recall without inventing matches.
    """
    text = strip_accents(text.lower())
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def words(text: str) -> List[str]:
    return _WORD.findall(text)


def word_ngrams(text: str, n: int) -> List[str]:
    """Contiguous word n-grams ("shingles"). Short documents yield one shingle."""
    toks = text.split()
    if len(toks) < n:
        return [" ".join(toks)] if toks else []
    return [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def char_ngrams(text: str, ns: Iterable[int] = (1, 2, 3)) -> List[str]:
    """Character n-grams with word-boundary padding, the classic language-ID feature."""
    out: List[str] = []
    for w in text.lower().split():
        w = f"_{w}_"
        for n in ns:
            out.extend(w[i:i + n] for i in range(max(1, len(w) - n + 1)))
    return out
