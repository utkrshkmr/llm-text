"""Regex-based PII masking (emails, North-American phone numbers, IPv4 addresses).

Regexes trade recall for precision. These patterns are intentionally conservative: they
catch the common formats and avoid masking version numbers, prices and dates. Production
systems layer NER models and allow-lists on top of rules like these.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict

EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?![\w-])")
# (212) 555-0100, 212-555-0100, 212.555.0100, 212 555 0100, +1 212 555 0100, 1-212-555-0100
PHONE = re.compile(
    r"(?<![\w])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\d)")
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4 = re.compile(rf"(?<![\d.]){_OCTET}(?:\.{_OCTET}){{3}}(?!\d|\.\d)")


@dataclass
class MaskResult:
    text: str
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


@dataclass
class PIIMasker:
    email_token: str = "<EMAIL>"
    phone_token: str = "<PHONE>"
    ip_token: str = "<IP>"

    def mask(self, text: str) -> MaskResult:
        counts = {}
        # order matters: emails first (they can contain digit runs), then IPs, then phones
        for name, pat, tok in (("email", EMAIL, self.email_token),
                               ("ip", IPV4, self.ip_token),
                               ("phone", PHONE, self.phone_token)):
            text, n = pat.subn(tok, text)
            counts[name] = n
        return MaskResult(text, counts)
