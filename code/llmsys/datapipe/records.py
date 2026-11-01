"""A minimal reader/writer for WARC-style record files.

Common Crawl ships three record files per segment:

* WARC -- raw HTTP responses (headers + HTML bytes),
* WAT  -- JSON metadata extracted from each response (links, title, ...),
* WET  -- plain text already extracted by Common Crawl ("conversion" records).

All three share the WARC container: a sequence of records, each a block of
``Key: Value`` header lines, a blank line, exactly ``Content-Length`` bytes of
payload, and two CRLFs. Production code should use FastWARC or warcio; this module
implements just enough of the format to make the container concrete and to let the
labs run without third-party dependencies.
"""
from __future__ import annotations

import gzip
import io
import uuid
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, Iterator, List, Optional


@dataclass
class WarcRecord:
    rec_type: str                       # "response", "conversion", "warcinfo", ...
    headers: Dict[str, str]             # WARC headers (WARC-Target-URI, ...)
    payload: bytes                      # for "response": HTTP headers + body

    @property
    def url(self) -> Optional[str]:
        return self.headers.get("WARC-Target-URI")

    def http_split(self):
        """For a response record, split into (http_header_dict, body_bytes)."""
        head, sep, body = self.payload.partition(b"\r\n\r\n")
        if not sep:
            return {}, self.payload
        lines = head.decode("iso-8859-1").split("\r\n")
        hdrs = {}
        for line in lines[1:]:
            k, _, v = line.partition(":")
            hdrs[k.strip().lower()] = v.strip()
        return hdrs, body


def _open(path: str, mode: str) -> BinaryIO:
    if path.endswith(".gz"):
        return gzip.open(path, mode)  # type: ignore[return-value]
    return open(path, mode)


def write_warc(records: List[WarcRecord], path: str) -> int:
    """Write records; returns bytes written (uncompressed)."""
    n = 0
    with _open(path, "wb") as f:
        for r in records:
            hdr = {"WARC-Type": r.rec_type,
                   "WARC-Record-ID": f"<urn:uuid:{uuid.uuid4()}>",
                   **r.headers,
                   "Content-Length": str(len(r.payload))}
            head = "WARC/1.0\r\n" + "".join(f"{k}: {v}\r\n" for k, v in hdr.items()) + "\r\n"
            blob = head.encode("utf-8") + r.payload + b"\r\n\r\n"
            f.write(blob)
            n += len(blob)
    return n


def iter_warc(path_or_bytes) -> Iterator[WarcRecord]:
    """Stream records from a (possibly gzipped) WARC/WET file or a bytes object."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        f: BinaryIO = io.BytesIO(path_or_bytes)
    else:
        f = _open(path_or_bytes, "rb")
    try:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.strip():
                continue
            if not line.startswith(b"WARC/"):
                raise ValueError(f"not a WARC record header: {line[:40]!r}")
            headers: Dict[str, str] = {}
            while True:
                h = f.readline()
                if h in (b"\r\n", b"\n", b""):
                    break
                k, _, v = h.decode("utf-8").partition(":")
                headers[k.strip()] = v.strip()
            length = int(headers.pop("Content-Length"))
            payload = f.read(length)
            rec_type = headers.pop("WARC-Type", "unknown")
            headers.pop("WARC-Record-ID", None)
            yield WarcRecord(rec_type, headers, payload)
    finally:
        f.close()


def make_response(url: str, html: str, charset: str = "utf-8",
                  declare_charset: bool = True) -> WarcRecord:
    """Build a WARC response record holding an HTTP response with an HTML body."""
    body = html.encode(charset, errors="replace")
    ctype = f"text/html; charset={charset}" if declare_charset else "text/html"
    http = (f"HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\n"
            f"Content-Length: {len(body)}\r\n\r\n").encode("ascii") + body
    return WarcRecord("response", {"WARC-Target-URI": url,
                                   "Content-Type": "application/http; msgtype=response"}, http)


def make_conversion(url: str, text: str) -> WarcRecord:
    """A WET-style conversion record (plain extracted text)."""
    return WarcRecord("conversion", {"WARC-Target-URI": url, "Content-Type": "text/plain"},
                      text.encode("utf-8"))
