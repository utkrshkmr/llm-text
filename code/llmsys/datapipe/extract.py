"""HTML -> text: charset detection, boilerplate-aware extraction, whitespace cleanup.

This is a deliberately small, dependency-free extractor so the mechanics are visible.
Production pipelines use Resiliparse, trafilatura or jusText, which add
content-density heuristics and are orders of magnitude faster (C/C++ cores).
"""
from __future__ import annotations

import html as htmllib
import re
from html.parser import HTMLParser
from typing import List, Optional, Tuple

_META_CHARSET = re.compile(rb'<meta[^>]+charset=["\']?([A-Za-z0-9_\-]+)', re.I)
_HTTP_CHARSET = re.compile(r"charset=([A-Za-z0-9_\-]+)", re.I)

# Elements whose text is never main content.
DROP = {"script", "style", "noscript", "template", "svg", "head", "iframe", "button",
        "select", "option", "form"}
# Elements that are usually boilerplate (dropped when boilerplate=True).
BOILER = {"nav", "footer", "header", "aside", "menu"}
# Block-level elements: a newline before and after their content.
BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr",
         "table", "section", "article", "main", "blockquote", "pre", "dd", "dt", "hr",
         "header", "footer", "nav", "aside", "body", "title"}


def detect_encoding(body: bytes, http_content_type: Optional[str] = None) -> str:
    """Pick a codec: HTTP header > <meta charset> > BOM > UTF-8 if it decodes > cp1252."""
    if http_content_type:
        m = _HTTP_CHARSET.search(http_content_type)
        if m:
            return m.group(1).lower()
    m = _META_CHARSET.search(body[:4096])
    if m:
        return m.group(1).decode("ascii").lower()
    if body.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        body.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"


def decode_html(body: bytes, http_content_type: Optional[str] = None) -> Tuple[str, str]:
    """Return (text, codec). Never raises: unknown codecs fall back to utf-8 with replacement."""
    enc = detect_encoding(body, http_content_type)
    try:
        return body.decode(enc), enc
    except (LookupError, UnicodeDecodeError):
        try:
            return body.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            return body.decode("cp1252", errors="replace"), "cp1252"


class _TextExtractor(HTMLParser):
    def __init__(self, boilerplate: bool):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.skip_depth = 0
        self.drop = DROP | (BOILER if boilerplate else set())
        self.stack: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("br", "hr"):
            self.parts.append("\n")
            return
        if tag in self.drop:
            self.skip_depth += 1
        self.stack.append(tag)
        if tag in BLOCK:
            self.parts.append("\n")
        if tag == "li" and not self.skip_depth:
            self.parts.append("- ")

    def handle_endtag(self, tag):
        if tag in ("br", "hr"):
            return
        # tolerate sloppy HTML: pop until the matching tag (if present)
        if tag in self.stack:
            while self.stack:
                t = self.stack.pop()
                if t in self.drop:
                    self.skip_depth -= 1
                if t == tag:
                    break
        if tag in BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)


def html_to_text(html: str, boilerplate: bool = True) -> str:
    """Extract visible text. ``boilerplate=True`` also drops nav/header/footer/aside."""
    p = _TextExtractor(boilerplate)
    p.feed(html)
    p.close()
    text = htmllib.unescape("".join(p.parts))
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in text.split("\n")]
    out, blank = [], False
    for ln in lines:                 # collapse runs of blank lines
        if ln:
            out.append(ln)
            blank = False
        elif not blank and out:
            out.append("")
            blank = True
    return "\n".join(out).strip()


def extract_record_text(record) -> Optional[str]:
    """WARC response record -> extracted text (None for non-HTML responses)."""
    hdrs, body = record.http_split()
    ctype = hdrs.get("content-type", "")
    if ctype and "html" not in ctype:
        return None
    text, _ = decode_html(body, ctype)
    return html_to_text(text)
