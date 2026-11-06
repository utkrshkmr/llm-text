"""Dark-mode variants of the book's SVG figures.

Every colour in a figure is rewritten by one rule that keeps its hue and saturation and flips its
lightness onto the dark theme's range: white surfaces become the dark figure card, black ink becomes
the light text colour, and mid-lightness accents (the house blue, red, green, ochre) come out lighter
so they keep their contrast. Authors draw one light-mode SVG; the site build writes <name>-dark.svg.

check_svg() lists colour notations the rule cannot read (for example named colours other than
none/black/white, or currentColor); the build reports them so new figures stay convertible.
"""
from __future__ import annotations

import colorsys
import re

DARK_SURFACE_L = 0.155   # lightness that white maps to (the dark theme's figure card, #23262b)
DARK_INK_L = 0.86        # lightness that black maps to (close to the dark theme's text, #d9d5ce)

_PCT = re.compile(r"rgb\(\s*([\d.]+)%\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\)")
_INT = re.compile(r"rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)")
_HEX = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
_NAMED = re.compile(r'((?:fill|stroke|stop-color|flood-color|color)\s*[:=]\s*"?)(black|white)\b')
_ANY = re.compile(r'(?:fill|stroke|stop-color|flood-color|color)\s*[:=]\s*"?\s*([^;"\s>]+)')


def dark_rgb(r: float, g: float, b: float):
    """r, g, b in [0, 1] -> the dark-theme colour, same hue, mapped lightness."""
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l2 = DARK_INK_L - (DARK_INK_L - DARK_SURFACE_L) * l
    s2 = s * (0.85 if l > 0.8 else 1.0)          # pale tints become muted dark tints
    return colorsys.hls_to_rgb(h, l2, s2)


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(c * 255))) for c in rgb)


_COLOUR = re.compile(_PCT.pattern + "|" + _INT.pattern + "|" + _HEX.pattern + "|" + _NAMED.pattern)


def dark_svg(text: str) -> str:
    """One pass over every colour, so no colour is converted twice."""
    def rep(m):
        g = m.groups()
        if g[0] is not None:
            rgb = [float(x) / 100 for x in g[0:3]]
        elif g[3] is not None:
            rgb = [int(x) / 255 for x in g[3:6]]
        elif g[6] is not None:
            v = g[6] if len(g[6]) == 6 else "".join(c * 2 for c in g[6])
            rgb = [int(v[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        else:
            return g[7] + _hex(dark_rgb(*((0, 0, 0) if g[8] == "black" else (1, 1, 1))))
        return _hex(dark_rgb(*rgb))
    return _COLOUR.sub(rep, text)


def check_svg(text: str):
    """Colour values the converter does not understand (empty list = fine)."""
    bad = set()
    for v in _ANY.findall(text):
        v = v.strip().rstrip(")")
        if v in ("none", "black", "white", "transparent", "inherit") or v.startswith(("rgb(", "#", "url(")):
            continue
        bad.add(v)
    return sorted(bad)
