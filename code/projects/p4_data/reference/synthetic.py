"""A tiny synthetic crawl with planted structure, for acceptance tests and `--synthetic` runs (Chapter 30).

The generator writes real WET files (gzipped WARC containers of ``conversion`` records, the format
Common Crawl ships) plus a ground-truth file, so every pipeline stage runs on exactly the input
format it sees in production while the acceptance tests know the right answer:

* ``article``     English prose to keep; some carry planted contact details (PII);
* ``neardup``     a copy of an article with one or two words changed and a sentence appended
                  (Jaccard on 5-gram shingles roughly 0.8-0.95): must be removed by near-dedup;
* ``exactdup``    a byte-identical copy under another URL: must be removed by exact dedup;
* ``footer``      an unrelated article sharing a long footer with other pages: must be kept
                  (the footer lines disappear in line dedup, the document does not);
* ``foreign``     Spanish or German text: language filter;
* ``spam``        keyword lists: heuristic rules;
* ``contaminated`` an article containing one benchmark item verbatim: decontamination.

Everything is generated from word lists written for this book; no real web content is included.
Contact details use reserved ranges (``example.com``, 555-01xx numbers, documentation IP blocks).
"""
from __future__ import annotations

import gzip
import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from llmsys.datapipe.records import make_conversion, write_warc
from llmsys.datapipe.synth import LANG_WORDS, SPAM

NOUNS = ("kernel memory bandwidth cache processor network compiler model dataset river forest "
         "climate engine bridge library garden museum orchestra theorem protein satellite harbor "
         "market council student teacher village scheduler pipeline shard manifest cluster").split()
ADJ = ("efficient large small modern ancient careful detailed reliable public local remarkable "
       "quiet complex simple northern parallel resumable deterministic").split()
VERBS = ("improves describes measures connects supports reduces explains protects transforms "
         "studies follows records compares").split()
FILL = LANG_WORDS["en"]
DOMAINS = ["news.example.org", "wiki.example.net", "blog.example.com", "docs.example.net",
           "forum.example.org", "shop.example.com"]
FOOTER = ("All articles on this site are published under the open documentation licence and may "
          "be shared with attribution. Contact the editors for corrections.")


@dataclass
class DocInfo:
    url: str
    kind: str                     # article | neardup | exactdup | footer | foreign | spam | contaminated
    lang: str
    keep: bool                    # should a correct pipeline keep this document?
    group: Optional[int] = None   # duplicate-family id (shared by an article and its copies)
    shard: int = 0
    pos: int = 0
    pii: Dict[str, int] = field(default_factory=dict)
    benchmark_item: Optional[int] = None


def _sentence(rng: random.Random) -> str:
    n1, n2, n3 = rng.sample(NOUNS, 3)
    extra = " ".join(rng.choices(FILL, k=rng.randint(3, 8)))
    return (f"The {rng.choice(ADJ)} {n1} {rng.choice(VERBS)} the {n2} "
            f"{rng.choice(['in', 'for', 'with', 'during', 'through'])} the {rng.choice(ADJ)} {n3} "
            f"{extra}.").capitalize()


def _article(rng: random.Random, n_sent: Tuple[int, int] = (8, 14)) -> str:
    sents = [_sentence(rng) for _ in range(rng.randint(*n_sent))]
    paras, i = [], 0
    while i < len(sents):
        k = rng.randint(2, 4)
        paras.append(" ".join(sents[i:i + k]))
        i += k
    return "\n".join(paras)


def _pii(rng: random.Random) -> Tuple[str, Dict[str, int]]:
    user = rng.choice(["alice", "bob.smith", "carol_j", "dev-team", "info"])
    email = f"{user}@{rng.choice(['example.com', 'example.org', 'mail.example.net'])}"
    area = rng.choice(["212", "415", "512", "617"])
    phone = rng.choice([f"({area}) 555-01{rng.randint(10, 99)}", f"{area}-555-01{rng.randint(10, 99)}"])
    ip = rng.choice(["192.0.2.", "198.51.100.", "203.0.113."]) + str(rng.randint(1, 254))
    return (f"Questions? Write to {email} or call {phone}. Our status server is reachable at {ip}.",
            {"email": 1, "phone": 1, "ip": 1})


def _foreign(rng: random.Random) -> Tuple[str, str]:
    lang = rng.choice(["es", "de", "fr"])
    sents = []
    for _ in range(rng.randint(8, 14)):
        sents.append((" ".join(rng.choices(LANG_WORDS[lang], k=rng.randint(8, 14))) + ".").capitalize())
    return "\n".join(" ".join(sents[i:i + 3]) for i in range(0, len(sents), 3)), lang


def _spam(rng: random.Random) -> str:
    return "\n".join(f"#{j} | " + " ".join(rng.choices(SPAM, k=rng.randint(3, 6))) + "..."
                     for j in range(rng.randint(25, 40)))


def _perturb(text: str, rng: random.Random) -> str:
    """One or two word substitutions plus an appended sentence: a near duplicate, not an exact one."""
    words = text.split(" ")
    for _ in range(rng.randint(1, 2)):
        i = rng.randrange(len(words))
        words[i] = rng.choice(NOUNS)
    return " ".join(words) + "\n" + _sentence(rng)


def make_benchmark(n: int = 40, seed: int = 123) -> List[str]:
    """Made-up benchmark items (word problems); each is at least 18 words, so a 13-gram index
    catches a verbatim copy."""
    rng = random.Random(seed)
    items = []
    for i in range(n):
        a, b, c = rng.randint(3, 40), rng.randint(2, 12), rng.randint(2, 9)
        who = rng.choice(["Maria", "Chen", "Amara", "Luis", "Priya", "Tomas"])
        thing = rng.choice(NOUNS)
        items.append(f"Question {i}: {who} buys {a} boxes of {thing} parts, each box holds {b} items, "
                     f"and gives away {c} items to a friend. How many {thing} items does {who} keep "
                     f"at the end of the day?")
    return items


def generate(out_dir: str, n_shards: int = 8, docs_per_shard: int = 60, seed: int = 0,
             mix: Optional[Dict[str, float]] = None) -> Dict:
    """Write ``out_dir/wet/shard-XXXX.warc.wet.gz``, ``truth.jsonl``, ``benchmark.jsonl`` and
    ``target.jsonl`` (clean held-out articles for validation). Returns the manifest dict."""
    rng = random.Random(seed)
    mix = mix or {"article": 0.42, "neardup": 0.14, "exactdup": 0.08, "footer": 0.08,
                  "foreign": 0.10, "spam": 0.10, "contaminated": 0.08}
    kinds = list(mix)
    bench = make_benchmark(seed=seed + 123)
    os.makedirs(os.path.join(out_dir, "wet"), exist_ok=True)
    infos: List[DocInfo] = []
    articles: List[Tuple[str, int]] = []   # (text, index into infos)
    next_group = 0
    paths = []
    for s in range(n_shards):
        records = []
        for p in range(docs_per_shard):
            kind = rng.choices(kinds, weights=[mix[k] for k in kinds])[0]
            if kind in ("neardup", "exactdup") and not articles:
                kind = "article"
            site = rng.choice(DOMAINS)
            url = f"https://{site}/{kind}/{s}-{p}"
            info = DocInfo(url, kind, "en", True, shard=s, pos=p)
            if kind == "article":
                text = _article(rng)
                if rng.random() < 0.35:
                    block, info.pii = _pii(rng)
                    text += "\n" + block
                articles.append((text, len(infos)))
            elif kind in ("neardup", "exactdup"):
                src_text, src_idx = rng.choice(articles)
                if infos[src_idx].group is None:
                    next_group += 1
                    infos[src_idx].group = next_group
                info.group = infos[src_idx].group
                info.pii = dict(infos[src_idx].pii)
                text = src_text if kind == "exactdup" else _perturb(src_text, rng)
            elif kind == "footer":
                text = _article(rng, (6, 10)) + "\n" + FOOTER
            elif kind == "foreign":
                text, info.lang = _foreign(rng)
                info.keep = False
            elif kind == "spam":
                text, info.keep = _spam(rng), False
            else:  # contaminated
                item = rng.randrange(len(bench))
                paras = _article(rng).split("\n")
                paras.insert(rng.randrange(len(paras) + 1), bench[item])
                text, info.benchmark_item, info.keep = "\n".join(paras), item, False
            infos.append(info)
            records.append(make_conversion(url, text))
        path = os.path.join(out_dir, "wet", f"shard-{s:04d}.warc.wet.gz")
        write_warc(records, path)
        paths.append(path)
    with open(os.path.join(out_dir, "truth.jsonl"), "w") as f:
        for info in infos:
            f.write(json.dumps(asdict(info)) + "\n")
    with open(os.path.join(out_dir, "benchmark.jsonl"), "w") as f:
        for i, t in enumerate(bench):
            f.write(json.dumps({"id": i, "text": t}) + "\n")
    target_rng = random.Random(seed + 7)
    with open(os.path.join(out_dir, "target.jsonl"), "w") as f:
        for i in range(max(20, docs_per_shard)):
            f.write(json.dumps({"url": f"https://target.example.org/{i}", "text": _article(target_rng)}) + "\n")
    manifest = {"n_shards": n_shards, "docs_per_shard": docs_per_shard, "seed": seed, "paths": paths,
                "kinds": {k: sum(i.kind == k for i in infos) for k in kinds},
                "truth": os.path.join(out_dir, "truth.jsonl"),
                "benchmark": os.path.join(out_dir, "benchmark.jsonl"),
                "target": os.path.join(out_dir, "target.jsonl")}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_truth(path: str) -> List[DocInfo]:
    with open(path) as f:
        return [DocInfo(**json.loads(line)) for line in f]


def truth_pairs(infos: Sequence[DocInfo]) -> set:
    """All (i, j) pairs of documents in the same duplicate family, by position in ``infos``."""
    groups: Dict[int, List[int]] = {}
    for i, d in enumerate(infos):
        if d.group is not None:
            groups.setdefault(d.group, []).append(i)
    pairs = set()
    for members in groups.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                pairs.add((members[x], members[y]))
    return pairs


def read_texts(wet_paths: Sequence[str]) -> List[str]:
    """Plain texts of the synthetic shards in generation order (for unit tests)."""
    from .wet import iter_wet_docs
    return [d.text for p in wet_paths for d in iter_wet_docs(p)]


def read_jsonl_texts(path: str) -> List[str]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]
