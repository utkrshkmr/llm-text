import math
import os
import tempfile
from collections import Counter

import numpy as np

from _labutil import load_impl

lab = load_impl(__file__)


# ----------------------------------------------------------------- Task 1-2: extraction
def test_detect_charset_priority():
    body = '<html><head><meta charset="iso-8859-1"></head>café</html>'.encode("cp1252")
    assert lab.detect_charset(body, "text/html; charset=UTF-8") == "utf-8"
    assert lab.detect_charset(body, "text/html") == "iso-8859-1"
    assert lab.detect_charset("plain café".encode("utf-8")) == "utf-8"
    assert lab.detect_charset("plain café".encode("cp1252")) == "cp1252"


def test_strip_html_drops_boilerplate_and_scripts():
    html = ("<html><head><title>T</title><script>var x = 1;</script><style>p{}</style></head>"
            "<body><nav><a>Home</a> <a>Login</a></nav><h1>Main title</h1>"
            "<p>First &amp; second.</p><p>Third   line.</p><footer>Copyright</footer></body></html>")
    t = lab.strip_html(html)
    assert "var x" not in t and "p{}" not in t and "Login" not in t and "Copyright" not in t
    assert "First & second." in t and "Third line." in t
    assert "Main title" in t.splitlines()[1] or "Main title" in t


# ----------------------------------------------------------------- Task 3: language ID
def test_naive_bayes_langid():
    from llmsys.datapipe.synth import language_training_set
    data = language_training_set(120, seed=1)
    model = lab.nb_train(data)
    test = language_training_set(30, seed=7)
    acc = sum(max(p := lab.nb_posterior(model, t), key=p.get) == y for t, y in test) / len(test)
    assert acc > 0.95
    post = lab.nb_posterior(model, test[0][0])
    assert abs(sum(post.values()) - 1) < 1e-9 and all(0 <= v <= 1 for v in post.values())


# ----------------------------------------------------------------- Task 4: PII
def test_mask_pii():
    s = ("Mail jane.doe+news@mail.example.org or call (415) 555-0123 / 617.555.0199 / "
         "+1 212 555 0100. Server 203.0.113.7. Version 1.2.3.4.5 and 999.1.1.1 are not IPs; "
         "price $12.50 is not a phone.")
    out, counts = lab.mask_pii(s)
    assert counts == {"email": 1, "ip": 1, "phone": 3}
    assert "<EMAIL>" in out and "<IP>" in out and out.count("<PHONE>") == 3
    assert "1.2.3.4.5" in out and "999.1.1.1" in out and "$12.50" in out


# ----------------------------------------------------------------- Task 5: rules
def test_rule_failures():
    good = " ".join(["The model and the data that we have with care."] * 12)
    assert lab.rule_failures(good) == []
    assert "word_count" in lab.rule_failures("Sign in to the page with password")
    spam = "\n".join(["# buy now cheap deal ..."] * 30)
    f = lab.rule_failures(spam)
    assert "symbol_ratio" in f and "ellipsis_lines" in f
    nums = " ".join(["$1.99 #4 12x3cm 555"] * 30)
    assert "alpha_words" in lab.rule_failures(nums)


# ----------------------------------------------------------------- Task 6: quality classifier
def test_quality_classifier_separates():
    from llmsys.datapipe.synth import quality_training_set
    tr, te = quality_training_set(400, seed=2), quality_training_set(100, seed=9)
    model = lab.train_quality([t for t, _ in tr], [y for _, y in tr], epochs=4)
    preds = [(lab.quality_score(model, t) >= 0.5) == bool(y) for t, y in te]
    assert sum(preds) / len(preds) > 0.9


# ----------------------------------------------------------------- Task 7: exact line dedup
def test_exact_line_dedup():
    docs = ["menu\nalpha one\nfooter", "menu\nbeta two\nfooter", "gamma\nmenu"]
    out = lab.exact_line_dedup(docs)
    assert out == ["alpha one", "beta two", "gamma"]


# ----------------------------------------------------------------- Task 8: MinHash + LSH
def test_minhash_estimates_jaccard():
    rng = np.random.default_rng(0)
    a = rng.integers(1, lab.P31, 256, dtype=np.int64)
    b = rng.integers(0, lab.P31, 256, dtype=np.int64)
    A = {f"w{i}" for i in range(100)}
    B = {f"w{i}" for i in range(30, 130)}               # J = 70 / 130
    est = (lab.minhash_signature(A, a, b) == lab.minhash_signature(B, a, b)).mean()
    assert abs(est - 70 / 130) < 0.08


def test_lsh_and_clusters():
    sigs = np.array([[1, 2, 3, 4], [1, 2, 9, 9], [7, 7, 3, 4], [5, 6, 8, 0]], dtype=np.int64)
    assert lab.lsh_pairs(sigs, bands=2) == {(0, 1), (0, 2)}
    keep = lab.cluster_keep(4, {(0, 1), (0, 2)}, seed=0)
    assert len(keep) == 2 and 3 in keep


def test_near_dedup_on_synthetic_crawl():
    from llmsys.datapipe.synth import generate_corpus
    from llmsys.datapipe.extract import html_to_text
    docs = generate_corpus(120, seed=3)
    texts = [html_to_text(d.html) for d in docs]
    keep, ncand, ntrue = lab.near_dedup(texts)
    groups = {}
    for i, d in enumerate(docs):
        if d.group:
            groups.setdefault(d.group, []).append(i)
    for members in groups.values():                     # exactly one survivor per family
        assert sum(m in keep for m in members) == 1
    assert ntrue <= ncand


# ----------------------------------------------------------------- Task 9: Bloom filter
def test_bloom_filter():
    bf = lab.Bloom(m_bits=20_000, k=7)
    items = [f"line {i}" for i in range(1000)]
    for x in items:
        bf.add(x)
    assert all(x in bf for x in items)                  # no false negatives
    fp = sum(f"other {i}" in bf for i in range(5000)) / 5000
    assert fp < 3 * lab.bloom_fp_rate(20_000, 7, 1000) + 0.01


# ----------------------------------------------------------------- Task 10: mixing
def test_unimax():
    sizes = {"web": 1000.0, "code": 100.0, "books": 10.0}
    w = lab.unimax(sizes, budget=300.0, max_epochs=2.0)
    assert abs(sum(w.values()) - 1) < 1e-9
    assert abs(w["books"] * 300 / 10 - 2.0) < 1e-9      # small source capped at 2 epochs
    assert abs(w["code"] - w["web"]) < 1e-9              # the rest is split uniformly


# ----------------------------------------------------------------- Task 11: end-to-end
def test_end_to_end_pipeline():
    with tempfile.TemporaryDirectory() as d:
        paths, docs = lab.make_synthetic_shards(os.path.join(d, "in"), n_shards=3, docs_per_shard=40)
        stats, kept, secs = lab.run_pipeline(paths, os.path.join(d, "out"), workers=2)
        assert sum(stats.values()) == len(docs)            # every doc accounted for once
        assert kept and all("<EMAIL>" not in k["text"] or "@" not in k["text"] for k in kept)
        assert not any(k["text"].lower().startswith("best deals") for k in kept)
        urls = {k["url"] for k in kept}
        kinds = Counter(d.kind for d in docs if d.url in urls)
        assert set(kinds) <= {"article", "template", "mirror"}
