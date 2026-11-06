"""Reading Common Crawl WET and WARC files: FastWARC when installed, a pure-Python fallback otherwise.

Both readers yield :class:`llmsys.datapipe.pipeline.Doc` records in file order, so a shard job is
deterministic whichever reader a machine has. WARC response records are decoded and extracted
(Resiliparse when installed, the Chapter 14 extractor otherwise); WET conversion records already hold
plain text. FastWARC reads a gzipped WET file several times faster than the pure-Python reader, which
matters for the real crawl and not at all for the synthetic one.
"""
from __future__ import annotations

from typing import Iterator, Optional

from llmsys.datapipe.extract import decode_html, html_to_text
from llmsys.datapipe.pipeline import Doc

try:  # pragma: no cover - optional accelerator
    from fastwarc.warc import ArchiveIterator, WarcRecordType  # type: ignore
    HAS_FASTWARC = True
except Exception:
    HAS_FASTWARC = False

try:  # pragma: no cover - optional accelerator
    from resiliparse.extract.html2text import extract_plain_text  # type: ignore
    from resiliparse.parse.encoding import bytes_to_str, detect_encoding  # type: ignore
    HAS_RESILIPARSE = True
except Exception:
    HAS_RESILIPARSE = False


def extract_html(body: bytes, content_type: Optional[str] = None) -> str:
    if HAS_RESILIPARSE:
        return extract_plain_text(bytes_to_str(body, detect_encoding(body)), main_content=True)
    text, _ = decode_html(body, content_type)
    return html_to_text(text)


def _iter_fastwarc(path: str) -> Iterator[Doc]:  # pragma: no cover - needs fastwarc
    with open(path, "rb") as f:
        for rec in ArchiveIterator(f, record_types=WarcRecordType.response | WarcRecordType.conversion):
            url = rec.headers.get("WARC-Target-URI") or ""
            body = rec.reader.read()
            if rec.record_type == WarcRecordType.conversion:
                yield Doc(url, body.decode("utf-8", errors="replace"))
            else:
                yield Doc(url, extract_html(body, rec.http_content_type))


def _iter_python(path: str) -> Iterator[Doc]:
    from llmsys.datapipe.records import iter_warc
    for rec in iter_warc(path):
        if rec.rec_type == "conversion":
            yield Doc(rec.url or "", rec.payload.decode("utf-8", errors="replace"))
        elif rec.rec_type == "response":
            hdrs, body = rec.http_split()
            yield Doc(rec.url or "", extract_html(body, hdrs.get("content-type")))


def iter_wet_docs(path: str, prefer_fastwarc: bool = True) -> Iterator[Doc]:
    """Documents of one WET/WARC file (optionally gzipped), in file order."""
    if HAS_FASTWARC and prefer_fastwarc:
        yield from _iter_fastwarc(path)
    else:
        yield from _iter_python(path)
