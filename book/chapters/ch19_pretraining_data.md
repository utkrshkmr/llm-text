# Pretraining Data: From Common Crawl to Tokens {#ch:data}

Every chapter so far has taken the training data as given: a public-domain short story in
[@ch:pretrain], a token file in [@ch:text]. At frontier scale, this assumption hides
most of the work. Two models with the same architecture, optimizer and FLOP budget can differ
substantially in quality purely because of what they were trained on, and open-weight model releases
typically document their architecture down to the norm epsilon while saying very little about their
data. Data is also the part of the stack that scales with human effort rather than with hardware:
there is no kernel that tells you whether a page about mortgage rates is "good" text.

This chapter builds a pretraining-data pipeline from raw web crawls to the flat token files that
[@ch:text; @ch:pretrain] consume. We study where data comes from and what you may legally
use, how the Common Crawl container format works, how to turn HTML into text, and how to filter by
language, personal information, harmful content and quality, with rules and with classifiers. We
then cover exact and near-duplicate removal with hashing, Bloom filters, MinHash and
locality-sensitive hashing; decontamination against benchmarks; and how to mix sources. Throughout,
we treat the pipeline as a *system*: what each stage costs per document, what has to be global
versus per-shard, how to parallelise it over thousands of cores, and how to make it restartable and
auditable. [@ch:proj4] turns this material into a capstone project: build a filtered corpus
and train on it.

![Roadmap of this chapter. Stages 2--6 look at one document at a time and parallelise trivially.
Stages 7--9 need information about the whole corpus (duplicates, benchmark overlap, source sizes)
and therefore global data structures or a shuffle. Stage 10 reuses the token-shard format of
[@ch:text]. Stage 11 is about running all of it on thousands of cores without losing work or
visibility.](figures/ch19_pretraining_data-1.svg){#fig:data:roadmap}

## Where pretraining data comes from {#sec:data:sources}

"Trained on the internet" is a convenient slogan and a poor description. The web is a set of live
servers; you cannot train on a server. Between the live web and a training set sit three
transformations: *acquisition* (crawling or downloading dumps), *conversion* (HTML, PDF or
repositories into text), and *curation* (filtering, deduplication, mixing). Each one discards most
of its input.

##### Crawls.

A crawler starts from seed URLs, downloads pages, extracts links and repeats. Much of the web is
unreachable this way: application-like sites render content with JavaScript after user interaction,
walled gardens require logins, many sites forbid automated access through `robots.txt` or their
terms of service, and rate limits and bot detection block aggressive clients. Nearly everyone
therefore starts from **Common Crawl** (CC), a non-profit that has published monthly crawls since
2008. A single crawl contains billions of pages, packaged as WARC files ([@sec:data:warc]).
CC is a sample of the web shaped by its crawl policy; it is not the web.

##### Curated sources.

Several high-value sources are distributed as dumps, so no crawling is needed. Wikipedia publishes
periodic dumps of every language edition; GitHub repositories can be cloned through the git protocol
(Software Heritage archives code across forges); arXiv offers bulk access to PDFs and LaTeX sources;
Stack Exchange publishes anonymised XML dumps with votes and metadata; Project Gutenberg distributes
public-domain books. Each has its own format, licence and duplication pattern: code is heavily
forked, books are mirrored, forum posts quote each other.

##### What may you use?

In most jurisdictions almost everything on the web is copyrighted the moment it is written;
registration is not required. Using a work then requires either a licence (Creative Commons,
permissive software licences, commercial data deals) or an argument that the use is fair use (in the
US, a four-factor test weighing purpose and transformativeness, the nature of the work, the amount
used, and the effect on the market). Whether training on copyrighted text is fair use is legally
unsettled as of this writing, and the answer may differ by jurisdiction and by how a model is
deployed. Two practical consequences matter for engineers. First, *provenance metadata* (URL, crawl
date, licence, source) should travel with every document through the pipeline so that later removal
requests or licence audits are possible. Second, *licence laundering* is real: a permissive
collection licence does not make every item inside it permissively licensed. Datasets built only
from openly licensed text, such as the Common Pile (Kandpal et al., 2025), exist precisely to
explore how far one can get under strict provenance rules.

::: {#tab:data:lineage .table-float}
| **Dataset (year)** | **Processing idea worth remembering** |
|:---|:---|
| WebText / OpenWebText (2019) | Pages linked from Reddit posts with a minimum karma: human curation as a quality signal. |
| CCNet (2019) | Paragraph dedup, fastText language ID, and a Wikipedia-trained $n$-gram LM (KenLM) to score "Wikipedia-likeness". |
| C4 (2019) | One CC snapshot filtered by simple line and page rules (terminal punctuation, minimum words, bad-word list, no code braces). |
| The Pile (2021) | 22 curated domains (papers, code, books, forums, legal) mixed with filtered CC. |
| MassiveText / Gopher (2021) | Interpretable quality rules ([@sec:data:rules]) plus repetition filters and SafeSearch. |
| RefinedWeb (2023) | "Web only": careful extraction from WARC, rules, MinHash dedup at scale. |
| Dolma (2024) | Open toolkit and multi-source corpus; fastText language ID, Gopher/C4 rules, Jigsaw toxicity classifiers, Bloom-filter dedup. |
| DCLM (2024) | A benchmark for data curation: fixed training recipe, compare filters by downstream accuracy; a fastText quality classifier wins. |
| FineWeb / FineWeb-Edu (2024) | Large-scale ablations of every pipeline choice; an LLM-annotated "educational value" score distilled into a small classifier. |
| Nemotron-CC (2024) | Classifier ensembles to keep more tokens, plus LLM rephrasing of low-quality and task generation from high-quality pages. |
| The Stack v1/v2 (2022/2024) | Permissively licensed code, license detection, near-dedup, PII redaction. |

: A lineage of open pretraining corpora. Each line is the idea that later pipelines inherited; sizes
and exact recipes are in the cited papers ([@app:refs]).
:::

##### Training stages.

Modern recipes use data in phases. *Pretraining* consumes trillions of tokens of mostly web text.
*Mid-training* (also called annealing or the cooldown phase) continues on a smaller, higher-quality
mixture, often with a decaying learning rate ([@ch:optim]), to sharpen math, code and
long-context skills. *Post-training* ([@ch:instruct; @ch:grpo; @ch:dpo]) uses curated or
synthetic conversations and reinforcement learning. The trend across phases is from large amounts of
lower-quality data to small amounts of higher-quality data, and the boundaries are blurry.

## The Common Crawl container: WARC, WAT and WET {#sec:data:warc}

Common Crawl publishes each crawl as tens of thousands of *segments*, each holding three parallel
record files ([@fig:data:warc]):

- **WARC** (Web ARChive) files contain the raw HTTP exchanges: request metadata, the HTTP response
  headers, and the response body (usually HTML) exactly as served.

- **WAT** files contain JSON metadata computed from each response, such as the title, outgoing links
  and HTTP headers.

- **WET** files contain Common Crawl's own plain-text extraction of each HTML page.

All three use the WARC container: a sequence of records, each a block of `Key: Value` headers, a
blank line, exactly `Content-Length` bytes of payload, and a record separator. Files are compressed
as a *concatenation of gzip members, one per record*. That detail matters for systems work: a reader
can start decompressing at any record boundary, so files can be split and records skipped without
decompressing everything before them, and a corrupt record does not poison the rest of the file.

![Anatomy of a WARC response record. Each record is its own gzip member, so readers can seek, split
and recover at record granularity. WAT and WET files use the same container with JSON and plain-text
payloads.](figures/ch19_pretraining_data-2.svg){#fig:data:warc}

[@lst:data:warc] shows the essential loop of a WARC reader from `llmsys.datapipe.records`.
It is a teaching implementation; production code should use FastWARC (C++, much faster) or warcio.

``` {#lst:data:warc .python caption="Streaming WARC records (from `llmsys/datapipe/records.py`)."}
def iter_warc(path):
    f = gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")
    while True:
        line = f.readline()
        if not line:
            return                                  # end of file
        if not line.strip():
            continue                                # record separator
        headers = {}
        while (h := f.readline()) not in (b"\r\n", b"\n", b""):
            k, _, v = h.decode().partition(":")
            headers[k.strip()] = v.strip()
        payload = f.read(int(headers.pop("Content-Length")))
        yield WarcRecord(headers.pop("WARC-Type", "?"), headers, payload)
```

::: pitfall
[Read the data before writing filters]{.box-title}

The first thing to do with a crawl is to page through raw records (`zcat file.warc.gz | less`) and
through the WET text for the same URLs. Within a few dozen records you will see login pages, cookie
banners, product grids, machine-translated spam, pages in dozens of languages, and only occasionally
an article you would want a model to learn from. Every filter in this chapter exists because someone
looked at the data and saw a failure. Data is unfiltered web content and can be offensive; skip
records you would rather not read.
:::

## From HTML to text {#sec:data:extract}

Extraction converts raw bytes into a Unicode string containing the page's main content. It has two
independent sub-problems: decoding bytes into characters, and deciding which characters are content.

##### Decoding.

About 98% of web pages are UTF-8, so the rest determines robustness. A sensible cascade is: the
`charset` parameter of the HTTP `Content-Type` header, then a `<meta charset>` tag in the first few
kilobytes, then a byte-order mark, then "does it decode as UTF-8?", and finally a statistical
detector or a permissive legacy codec such as cp1252. Getting this wrong produces *mojibake*, such
as `cafÃ©` instead of `café`, which a model will dutifully learn. Resiliparse bundles a fast
detector (`detect_encoding`) with its extractor.

##### Content selection.

HTML contains far more visible text than a human perceives as the page's content: navigation menus,
sidebars, cookie notices, footers, "related articles", comment widgets. Extractors parse the DOM,
drop elements that are never content (`script`, `style`, `form`), optionally drop elements that are
usually boilerplate (`nav`, `header`, `footer`, `aside`), insert line breaks at block-level
elements, and apply content-density heuristics (long text blocks with few links are content). Tools
differ mainly in those heuristics: Resiliparse (fast, rule-based, "main content" mode), trafilatura
(more aggressive boilerplate removal), jusText (paragraph classification). Extraction is lossy by
design: tables are linearised, math markup and code formatting are often mangled, and images vanish.

``` {#lst:data:extract .python caption="Charset cascade and a DOM-walking extractor (abridged from `llmsys/datapipe/extract.py`)."}
def detect_encoding(body, http_content_type=None):
    if http_content_type and (m := _HTTP_CHARSET.search(http_content_type)):
        return m.group(1).lower()                   # 1. HTTP header
    if m := _META_CHARSET.search(body[:4096]):
        return m.group(1).decode("ascii").lower()   # 2. <meta charset=...>
    if body.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"                          # 3. byte-order mark
    try:
        body.decode("utf-8"); return "utf-8"        # 4. valid UTF-8?
    except UnicodeDecodeError:
        return "cp1252"                             # 5. permissive fallback

class _TextExtractor(HTMLParser):                   # stdlib, streaming
    def handle_starttag(self, tag, attrs):
        if tag in self.drop:      self.skip_depth += 1   # script, nav, footer, ...
        if tag in BLOCK:          self.parts.append("\n")
    def handle_data(self, data):
        if not self.skip_depth:   self.parts.append(data)
```

::: systemslens
[Extraction is the most expensive per-byte stage]{.box-title}

Parsing HTML touches every byte of every page, and pages average tens of kilobytes of markup for a
few kilobytes of text. A pure-Python parser like ours processes on the order of a megabyte per
second per core; C/C++ extractors are one to two orders of magnitude faster. That difference decides
whether reprocessing a crawl takes hours or weeks on a given cluster, which is why pipelines that
*can* reuse WET text often do, and why the ones that re-extract from WARC (RefinedWeb, DCLM) use
compiled tools. Measure your extractor in MB/s of input, not documents per second, because page
sizes are heavy-tailed. The DCLM authors reported that the choice of extractor alone measurably
changed downstream accuracy, so this is not merely a speed question.
:::

## Language identification {#sec:data:langid}

Most curated corpora restrict themselves to a fixed set of languages, often just English, so the
next stage estimates $p(\ell\mid x)$ for each language $\ell$. The standard tool is fastText's
`lid.176` model: a linear classifier over hashed character $n$-grams that supports 176 languages and
runs in microseconds. To see why such a simple model works, we build its classical cousin,
multinomial Naive Bayes over character $n$-grams.

Represent a document $x$ by the multiset of character $n$-grams $g$ of its words, padded with
boundary markers (`_the_` yields `_t`, `th`, `he`, `e_`, `_th`, ...). Naive Bayes assumes $n$-grams
are independent given the language: $$\begin{equation}
\log p(\ell\mid x) = \log p(\ell) + \sum_{g} c_g(x)\,\log p(g\mid\ell) - \log Z(x),
\qquad
p(g\mid\ell) = \frac{c_g(\ell) + \alpha}{\sum_{g'} c_{g'}(\ell) + \alpha V},
\label{eq:data:nb}
\end{equation}$$ with add-$\alpha$ smoothing over the $n$-gram vocabulary of size $V$. The
normaliser $Z(x)=\sum_\ell \exp(\cdot)$ is computed with the log-sum-exp trick, since the
unnormalised log-probabilities of a long document are in the thousands.

``` {#lst:data:nb caption="Naive Bayes posterior over languages (from `llmsys/datapipe/classify.py`)."}
def scores(self, text):
    feats = Counter(char_ngrams(text, self.ns))
    logp = {}
    for lang in self.langs:
        ll, unk = self.log_lik[lang], self.log_unseen[lang]
        logp[lang] = self.log_prior[lang] + sum(c * ll.get(g, unk)
                                                for g, c in feats.items())
    m = max(logp.values())                                  # log-sum-exp
    z = sum(math.exp(v - m) for v in logp.values())
    return {k: math.exp(v - m) / z for k, v in logp.items()}
```

##### Choosing a threshold.

A filter keeps documents with $p(\text{en}\mid x)\ge\tau$. Dolma used $\tau=0.5$ with fastText;
stricter thresholds trade recall for purity. The posterior is overconfident on long documents,
because independent evidence multiplies, and unreliable on short ones, so it is worth inspecting the
score distribution on a few hundred hand-labelled pages before picking $\tau$.

::: pitfall
[Where language ID fails]{.box-title}

Short texts such as titles or two-line pages; code, numbers and tables, which have no language (on
our synthetic crawl, product listings were confidently assigned to German); code-switched documents;
closely related languages and dialects; and boilerplate in one language around content in another.
Systematic failures are not neutral: a classifier that under-recognises a dialect removes that
dialect's speakers from the training set, which then shapes what the model can understand. For a
deployed product, audit per-language and per-dialect recall on curated samples and keep borderline
documents for review rather than silently dropping them.
:::

## Personal information and harmful content {#sec:data:pii}

##### PII masking.

Emails, phone numbers and IP addresses can identify or reach people. Pipelines typically mask them
with placeholder tokens rather than dropping whole documents. Regular expressions are the first line
of defence; [@lst:data:pii] shows patterns for the common North-American formats. Writing
them is an exercise in precision versus recall: the IPv4 pattern must accept `203.0.113.7.` at the
end of a sentence yet reject version strings like `1.2.3.4.5` and out-of-range octets like
`999.1.1.1`; the phone pattern must accept `(415) 555-0123`, `415.555.0123` and `+1 415 555 0123`
but not prices or ISBN fragments.

``` {#lst:data:pii .python caption="Conservative PII regexes (from `llmsys/datapipe/pii.py`)."}
EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+"
                   r"[A-Za-z]{2,}(?![\w-])")
PHONE = re.compile(r"(?<![\w])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])"
                   r"\d{3}[\s.-]\d{4}(?!\d)")
OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4  = re.compile(rf"(?<![\d.]){OCTET}(?:\.{OCTET}){{3}}(?!\d|\.\d)")

def mask(self, text):                    # order: emails, then IPs, then phones
    counts = {}
    for name, pat, tok in (("email", EMAIL, "<EMAIL>"), ("ip", IPV4, "<IP>"),
                           ("phone", PHONE, "<PHONE>")):
        text, counts[name] = pat.subn(tok, text)
    return MaskResult(text, counts)
```

Masking has side effects. A model trained on text full of `<EMAIL>` placeholders will generate
placeholders in contact sections; one trained on badly masked text may learn half-numbers. Some
pipelines replace PII with *synthetic but well-formed* values from reserved ranges instead. Regexes
also miss names and street addresses entirely; named-entity recognisers catch more at a much higher
cost per document, so they are usually applied only to sources likely to contain personal data.

##### Harmful content.

"Harmful" has no crisp definition, but most pipelines remove documents that are predominantly
pornographic, hateful or otherwise toxic. Two techniques dominate: word lists (C4 dropped any page
containing a term from a public list of offensive words) and classifiers trained on human labels.
Dolma, for example, trained fastText classifiers for NSFW and hate speech on the Jigsaw
toxic-comment dataset of annotated Wikipedia talk-page comments, and these models are publicly
available. Both approaches have documented biases. Word lists remove medical and LGBTQ+ health
content (Dodge et al., 2021) that happens to use anatomical or identity terms. Classifiers trained
on one register (Wikipedia comments) misfire on others and over-flag dialects such as
African-American English (Sap et al., 2019). Thresholds should therefore be set by looking at
samples near the boundary, and removal should be weighed against the downstream need for a model
that can *recognise* harmful content, for example to refuse or to moderate ([@ch:alignment]).

## Heuristic quality rules {#sec:data:rules}

After language and safety filtering, most surviving pages are still poor training text: login walls,
placeholder pages, product grids, keyword spam, pages whose content was images. The Gopher authors
(Rae et al., 2021) described a compact set of interpretable rules that remove much of this.
[@tab:data:gopher] lists them with the thresholds of the MassiveWeb pipeline.

::: {#tab:data:gopher .table-float}
| **Remove the document if ...** | **Catches** |
|:---|:---|
| it has fewer than 50 or more than 100,000 words | fragments, login pages; dumps |
| mean word length is outside $[3,10]$ characters | tables of numbers; concatenated junk |
| the symbol-to-word ratio (# or ellipsis) exceeds 0.1 | hashtag spam, truncated teasers |
| more than 90% of lines start with a bullet | menus, link lists |
| more than 30% of lines end with an ellipsis | paywalled teasers, search result pages |
| fewer than 80% of words contain an alphabetic character | listings, logs, code dumps |
| it contains fewer than 2 of {the, be, to, of, and, that, have, with} | non-prose, non-English residue |
| duplicate lines, paragraphs or top $n$-grams exceed set fractions | templated and repetitive pages |

: Gopher-style quality rules (after Rae et al., 2021, Appendix A). C4 used a different, line-level
set: keep lines ending in terminal punctuation with at least five words, and drop pages with fewer
than three sentences or containing "lorem ipsum" or a curly brace.
:::

The engineering lesson is to return *which* rules fired, not just a boolean
([@lst:data:rules]). Per-rule counts tell you which rule is doing the work, reveal rules
that fire on content you want, and make the writeup's "what did each filter remove?" question
answerable for free.

``` {#lst:data:rules .python caption="Gopher-style rules with a per-rule report (abridged from `llmsys/datapipe/rules.py`)."}
def gopher_report(text, cfg=RuleConfig()):
    words = text.split()
    lines = [l for l in text.splitlines() if l.strip()]
    n, failed = len(words), []
    if not (cfg.min_words <= n <= cfg.max_words):
        failed.append("word_count")
    mean = sum(map(len, words)) / n if n else 0.0
    if not (cfg.min_mean_word_len <= mean <= cfg.max_mean_word_len):
        failed.append("mean_word_len")
    sym = text.count("#") + text.count("...")
    if n and sym / n > cfg.max_symbol_word_ratio:
        failed.append("symbol_ratio")
    ell = sum(l.rstrip().endswith("...") for l in lines) / max(1, len(lines))
    if ell > cfg.max_ellipsis_line_frac:
        failed.append("ellipsis_lines")
    alpha = sum(any(c.isalpha() for c in w) for w in words) / max(1, n)
    if alpha < cfg.min_alpha_word_frac:
        failed.append("alpha_words")
    # ... bullets, stop words, duplicate lines, top-bigram repetition ...
    return RuleReport(not failed, failed, stats)
```

Rules are cheap (one pass over the words) and transparent, which is why several groups (C4, Gopher,
RefinedWeb, FineWeb, Dolma) deliberately relied on them rather than on learned quality scores, to
avoid baking a classifier's biases into the corpus. Their weakness is that they capture syntax, not
substance: a fluent page of SEO filler passes every rule.

## Model-based quality filtering {#sec:data:quality}

Learned quality filters all instantiate one recipe. Given a small set of **target** documents $T$
that look like what we want, and a huge pool of **raw** documents $R$, find a subset $T'\subset R$
that resembles $T$. The two requirements pull against each other. The filter should generalise
beyond $T$, since copying $T$ is pointless, and it must be extremely cheap, because it runs on all
of $R$. Two families of scorer dominate:

Generative.

: Fit a language model $p_T$ on $T$ (classically a 5-gram KenLM) and keep $x$ if its perplexity
  under $p_T$ is low. CCNet ranked CC by perplexity under a Wikipedia $n$-gram model; OpenWebMath
  combined rules, a KenLM trained on math, and a fastText classifier.

Discriminative.

: Train a binary classifier $p(T\mid x)$ on positives from $T$ and negatives sampled from $R$, then
  keep $x$ with $p(T\mid x)\ge\tau$, or keep it *stochastically* with probability increasing in the
  score, which preserves diversity. GPT-3 trained a linear classifier with WebText, Wikipedia and
  books as positives; LLaMA used pages *referenced by* Wikipedia as positives; DCLM trained fastText
  on instruction-like data (OpenHermes 2.5 and the r/explainlikeimfive subreddit) versus RefinedWeb;
  FineWeb-Edu prompted a large LLM to rate educational value and distilled those ratings into a
  small classifier.

![The target-versus-raw recipe behind language ID, toxicity and quality filters. Only the choice of
$T$ changes. The scorer must be cheap enough to run on all of $R$, which is why linear models over
hashed $n$-grams dominate.](figures/ch19_pretraining_data-3.svg){#fig:data:targetraw}

##### A fastText-style classifier from scratch.

fastText's supervised model averages learned embeddings of a document's word and $n$-gram features
and applies a linear layer. With embedding dimension one, the averaged embedding is itself the
logit, giving the model of [@lst:data:quality]: $$\begin{equation}
s(x) = \sigma\!\Big(b + \frac{1}{|F(x)|}\sum_{f\in F(x)} w_{h(f)}\Big),
\label{eq:data:hashed}
\end{equation}$$ where $F(x)$ are the unigrams and bigrams of $x$ and $h$ hashes each feature into
one of $B$ buckets (the *hashing trick*: no vocabulary to build or store, collisions are tolerable
when $B\gg$ active features). Mean pooling makes the score independent of document length. The
logistic-loss gradient with respect to $z$ is $\sigma(z)-y$, spread equally over the document's
features.

``` {#lst:data:quality .python caption="Training the hashed $n$-gram classifier with SGD (from `llmsys/datapipe/classify.py`)."}
def fit(self, texts, labels, epochs=5, lr=0.5, l2=1e-6):
    X = [hashed_features(t, self.buckets, self.bigrams) for t in texts]
    for _ in range(epochs):
        for i in self.rng.permutation(len(X)):
            f = X[i]                                     # int64 bucket ids
            z = self.b + (self.w[f].mean() if len(f) else 0.0)
            g = 1.0 / (1.0 + math.exp(-z)) - labels[i]   # dL/dz
            if len(f):
                np.add.at(self.w, f, -lr * (g / len(f) + l2 * self.w[f]))
            self.b -= lr * g
```

##### Building the training set is the real work.

Positives defined as "pages linked from Wikipedia" must first be fetched (respecting timeouts and
politeness), then extracted and filtered by the same language, safety and rule stages, otherwise the
classifier learns to love cookie banners that happen to appear on reputable sites. Negatives should
come from the *same* stage of the pipeline as the documents the classifier will score. Hold out a
test split, look at the highest- and lowest- scoring documents, and choose $\tau$ by the number of
tokens you need, not by accuracy.

::: keyidea
[Filtering strength is scale dependent]{.box-title}

No single quality threshold is optimal. A short training run benefits from aggressive filtering,
since there is plenty of high-quality data to fill its budget, while a long run on the same pipeline
would have to repeat that small high-quality set many times, which hurts
([@sec:data:mixing]). Nemotron-CC's motivation was precisely that the strongest filters
discard about 90% of tokens, leaving too little for multi-trillion-token runs. When you compare
filters, compare them at the token budget you will actually train with.
:::

::: napkinmath
[What a classifier costs on a whole crawl]{.box-title}

Suppose a crawl has $2.5\times10^{9}$ pages. A linear hashed-$n$-gram classifier costs a few
microseconds per page in C++, so scoring everything is about $2.5\times10^9\times 5\,\mu\text{s}
\approx 3.5$ core-hours: negligible. A 100M-parameter BERT-style scorer at, say, 5 ms per page on a
CPU core is $3.5\times10^{3}$ core-hours, or a few GPU-hours per billion pages when batched. A 70B
LLM annotating every page is out of the question, which is why FineWeb-Edu-style pipelines have an
LLM label a few hundred thousand pages and distil the labels into a cheap classifier.
:::

## Deduplication {#sec:data:dedup}

The web is full of repetition. Some of it is exact: mirrors, forks and default server pages. Much
more is partial: every page of a forum shares its header and footer, licence texts are filled-in
templates, and one product description was found repeated tens of thousands of times in C4.
Deduplication pays twice. Training gets more efficient, because tokens are not wasted on repeats,
and models memorise and regurgitate less, which matters for privacy and copyright (Lee et al.,
2022). The design space has three axes: *what is an item* (line, paragraph, three-sentence span,
document), *what is a match* (exact, sharing a sub-item, high overlap), and *what to do* (drop all
copies, keep one). The algorithmic constraint is that we cannot compare all pairs: $10^{9}$
documents form $5\times10^{17}$ pairs. Everything below is linear time.

### Exact deduplication with hashes {#sec:data:exact}

Replace each item by a fixed-size hash and compare hashes. For document-level exact dedup, keep the
first document with each (normalised) hash. For *line-level* dedup as described in the Project 4,
make two passes: count each line's hash across the corpus, then rewrite every document keeping only
lines that occur exactly once ([@lst:data:lines]). This removes site-wide boilerplate
(menus, footers, cookie notices) from *all* pages, leaving each page's unique content. Hashing to 64
bits makes the counter's keys 8 bytes regardless of line length; non-cryptographic hashes
(MurmurHash, xxHash, a truncated BLAKE2) are fine because we only need uniformity, not collision
resistance against an adversary.

``` {#lst:data:lines .python caption="Two-pass exact line deduplication (simplified from `llmsys/datapipe/dedup.py`, which also offers a keep-first policy)."}
def hash64(s):                          # stable across processes
    d = hashlib.blake2b(s.encode(), digest_size=8).digest()
    return int.from_bytes(d, "little")

def dedup_lines(docs):
    counts = Counter(hash64(ln)                                 # pass 1
                     for d in docs for ln in d.splitlines())
    out = []
    for d in docs:                                              # pass 2
        kept = [ln for ln in d.splitlines() if counts[hash64(ln)] == 1]
        out.append("\n".join(kept))
    return out
```

::: pitfall
[Removing spans can break documents]{.box-title}

C4 deduplicated three-sentence spans; removing a repeated span from the middle of a page can leave
incoherent text. Line-level dedup has the same failure mode at a smaller scale, and "unique only"
also deletes legitimately repeated lines ("Thank you!", "Chapter 1") from every document. Inspect
before-and-after samples, and consider exempting very short lines.
:::

##### Bloom filters.

When the set of seen items is too large for a hash table, a Bloom filter answers "have I (probably)
seen this?" in a fixed number of bits. It is an $m$-bit array and $k$ hash functions; inserting an
item sets its $k$ bits, and a query returns true if all $k$ bits are set. There are no false
negatives. After $n$ insertions a given bit is still zero with probability
$(1-1/m)^{kn}\approx e^{-kn/m}$, so the false-positive rate is $$\begin{equation}
\varepsilon \approx \left(1-e^{-kn/m}\right)^{k},
\qquad k^\star = \frac{m}{n}\ln 2,\qquad
\frac{m}{n} = \frac{-\ln\varepsilon}{(\ln 2)^2}\ \text{bits per item}.
\label{eq:data:bloom}
\end{equation}$$ Dolma used Bloom filters for paragraph-level dedup. A false positive drops a unique
item, which is a small, random loss of data rather than a correctness problem. Our implementation
derives the $k$ positions from one 128-bit hash by double hashing, $p_i = (h_1 + i\,h_2)\bmod m$.

::: napkinmath
[Exact dedup memory at web scale]{.box-title}

Suppose a corpus has $10^{11}$ distinct lines. A hash table of 8-byte keys and 4-byte counts with
typical 2$\times$ overhead needs about $10^{11}\times 24\,\text{B}=2.4$ TB, so it must be sharded
across machines by hash prefix: each line's hash decides which reducer counts it, a classic
MapReduce. A Bloom filter for membership at $\varepsilon=1\%$ needs
$-\ln 0.01/(\ln 2)^2 \approx 9.6$ bits per item, which is $1.2\times10^{11}$ bytes, or 120 GB, with
$k\approx 7$: large, but it fits in the RAM of one big machine.
:::

### Near duplicates: Jaccard similarity and MinHash {#sec:data:minhash}

Exact hashing misses documents that differ by a name, a date or some whitespace. To catch them we
represent a document by its set of word $n$-grams (*shingles*), after normalising text by
lower-casing, NFD-decomposing and stripping accents, removing punctuation and collapsing whitespace,
and measure overlap with the Jaccard similarity $$J(S,T) = \frac{|S\cap T|}{|S\cup T|}.$$ Two
documents are near duplicates if $J\ge\theta$ (typical $\theta\in[0.7,0.9]$ with 5-gram shingles).
Computing $J$ for all pairs is quadratic and storing shingle sets is larger than the documents
themselves. **MinHash** fixes the storage problem.

::: {#prop:data:minhash .proposition}
\[MinHash\] Let $h$ be a random hash function that induces a uniformly random ordering of all
shingles. Then $\Pr\!\big[\min_{s\in S}h(s) = \min_{t\in T}h(t)\big] = J(S,T)$.
:::

::: proof
*Proof.* Consider the element of $S\cup T$ with the smallest hash. By symmetry it is equally likely
to be any of the $|S\cup T|$ elements. The two minima are equal exactly when that element lies in
$S\cap T$, which happens with probability $|S\cap T|/|S\cup T|$. ◻
:::

With $k$ independent hash functions, the *signature* $\big(\min_S h_1,\ldots,\min_S h_k\big)$
estimates $J$ by the fraction of agreeing coordinates, an unbiased estimator with standard deviation
$\sqrt{J(1-J)/k}$: for $J=0.8$, $k=128$ gives $\pm0.035$. In practice the "random permutations" are
universal hash functions $h_i(x) = (a_i x + b_i)\bmod p$ applied to a base hash of each shingle;
with a 31-bit prime all products fit in 64-bit integers and the whole signature is one vectorised
NumPy expression ([@lst:data:minhash]). A document's signature costs $k$ integers ($k=128$
gives 512 bytes) no matter how long the document is.

![Left: why a MinHash collision has probability $J(S,T)$. Right: LSH banding. A signature is cut
into $b$ bands of $r$ rows; two documents collide if *all* rows of *some* band agree. The AND within
a band and the OR across bands produce the sharp S-curve of
[@fig:data:scurve].](figures/ch19_pretraining_data-4.svg){#fig:data:minhash}

``` {#lst:data:minhash .python caption="Vectorised MinHash signatures (from `llmsys/datapipe/dedup.py`)."}
MERSENNE_31 = (1 << 31) - 1

class MinHasher:
    def __init__(self, num_perm=128, ngram=5, seed=0):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, MERSENNE_31, size=num_perm, dtype=np.int64)
        self.b = rng.integers(0, MERSENNE_31, size=num_perm, dtype=np.int64)
        self.ngram = ngram
    def shingles(self, text):
        return set(word_ngrams(normalize_for_dedup(text), self.ngram))
    def signature(self, text):
        x = np.fromiter((hash64(s) % MERSENNE_31
                         for s in self.shingles(text)), np.int64)
        # (k, |S|) table of hashes; a*x < 2^62, so int64 never overflows
        hv = (self.a[:, None] * x[None, :] + self.b[:, None]) % MERSENNE_31
        return hv.min(axis=1)
```

### Locality-sensitive hashing {#sec:data:lsh}

Signatures shrink documents but still leave $O(N^2)$ comparisons. Locality-sensitive hashing turns
similarity search into hashing: split each $k=br$ signature into $b$ bands of $r$ rows, hash each
band, and declare two documents *candidates* if they land in the same bucket for at least one band
([@fig:data:minhash], right). If $J(S,T)=s$, one band matches with probability $s^r$, so
$$\begin{equation}
\Pr[\text{candidate}\mid s] = 1-\left(1-s^{r}\right)^{b},
\qquad s^\star \approx \left(1/b\right)^{1/r}.
\label{eq:data:scurve}
\end{equation}$$ Increasing $r$ (AND) pushes the curve right and sharpens it; increasing $b$ (OR)
pulls it left. [@fig:data:scurve] plots the trade-off for $k=128$. Lee et al. (2022) used
$k=9000$ with $b=20$, $r=450$, putting the steep part near $s^\star=(1/20)^{1/450}\approx0.993$; at
the threshold itself the collision probability is $1-(1-1/20)^{20}\approx 1-1/e$.

![The LSH S-curve, [@eq:data:scurve], for $k=br=128$ hashes. With $b=16, r=8$ (green) pairs
with $J=0.8$ become candidates with probability 0.95 and pairs with $J=0.5$ with probability 0.06;
the inflection point is $(1/16)^{1/8}\approx0.71$. The dashed line is a single MinHash, whose
collision probability is just $s$.](figures/ch19_pretraining_data-5.svg){#fig:data:scurve}

LSH candidates still include false positives, so a careful pipeline *verifies* each candidate pair
with the exact Jaccard similarity of the shingle sets, or with the signature estimate when the sets
are too large to keep. Verified pairs are then merged into clusters with union-find, since
duplication is transitive in practice (A matches B in one band, B matches C in another), and one
document per cluster survives ([@lst:data:neardedup]).

``` {#lst:data:neardedup .python caption="LSH candidates, verification and clustering (abridged from `llmsys/datapipe/dedup.py`)."}
def lsh_candidates(sigs, bands):                 # sigs: (N, k)
    r = sigs.shape[1] // bands
    pairs = set()
    for b in range(bands):
        buckets = defaultdict(list)
        block = np.ascontiguousarray(sigs[:, b * r:(b + 1) * r])
        for i in range(len(sigs)):
            buckets[block[i].tobytes()].append(i)    # band contents = bucket key
        for m in buckets.values():
            pairs.update((m[x], m[y]) for x in range(len(m))
                         for y in range(x + 1, len(m)))
    return pairs

def near_dedup(docs, num_perm=128, bands=16, ngram=5, threshold=0.8, seed=0):
    mh = MinHasher(num_perm, ngram, seed)
    sh = [mh.shingles(d) for d in docs]
    sigs = np.stack([mh.signature(d) for d in docs])
    uf = UnionFind(len(docs))
    for i, j in lsh_candidates(sigs, bands):
        if jaccard(sh[i], sh[j]) >= threshold:       # remove LSH false positives
            uf.union(i, j)
    # keep one (seeded-random) member of every cluster
```

::: systemslens
[Near-dedup as a MapReduce]{.box-title}

At web scale, [@lst:data:neardedup] becomes a dataflow job
([@fig:data:mapreduce]). **Map**: each worker reads a shard and emits, for every document
and every band, the key `(band_id, hash(band))` with the document id as the value; signatures cost
about $k$ multiply-mod operations per shingle. **Shuffle**: the framework groups by key; this is the
only all-to-all communication, and its volume is $b$ small records per document. **Reduce**: each
non-singleton bucket emits candidate pairs. A final connected-components pass (distributed
union-find or iterative label propagation) clusters them. Two practical hazards: *giant buckets*
(thousands of boilerplate-only pages share a band) blow up the pair count quadratically, so cap
bucket sizes or treat large buckets as single clusters; and the seeded choice of survivor must be
deterministic across reruns so that a restarted job keeps the same documents.
:::

![MinHash-LSH deduplication as a map--shuffle--reduce job. The only global communication is the
shuffle of $b$ small band records per document; everything else is local to a
worker.](figures/ch19_pretraining_data-6.svg){#fig:data:mapreduce}

::: napkinmath
[MinHash over a crawl]{.box-title}

Take $2.5\times10^{9}$ documents averaging about 1,000 words, so about $10^{3}$ 5-gram shingles
each, and $k=128$ hashes. Signatures cost $2.5\times10^9\times10^3\times128\approx3\times10^{14}$
multiply-mod operations. At $10^{9}$ vectorised operations per second per core that is about 90
core-hours, cheap next to extraction. Storing signatures takes
$2.5\times10^{9}\times128\times4\,\text{B}\approx1.3$ TB, and the shuffle moves $b=16$ records of
roughly 16 bytes per document, about 0.6 TB in total. The expensive part in practice is the skew
from giant buckets, not the arithmetic.
:::

## Decontamination {#sec:data:decontam}

If benchmark questions or answers appear in the training data, evaluation scores measure
memorisation rather than skill ([@ch:eval]). Decontamination indexes every long $n$-gram (13
words is a common choice) of every evaluation set in a hash set and drops, or surgically edits,
training documents that share them ([@lst:data:decontam]). Long $n$-grams make accidental
matches rare while catching copied problems. Paraphrased or translated test items escape this
filter, which is one reason why held-out, freshly written and dynamically updated benchmarks matter.

``` {#lst:data:decontam .python caption="Long-$n$-gram decontamination (from `llmsys/datapipe/decontam.py`)."}
class Decontaminator:
    def __init__(self, eval_texts, n=13):
        self.n, self.index = n, set()
        for t in eval_texts:
            grams = word_ngrams(normalize_for_dedup(t), n)
            self.index.update(hash64(g) for g in grams if len(g.split()) == n)
    def hits(self, text):
        grams = word_ngrams(normalize_for_dedup(text), self.n)
        return sum(hash64(g) in self.index
                   for g in grams if len(g.split()) == self.n)
```

The same machinery protects *validation* sets during data work. Project 4 evaluates filtered data by
perplexity on a held-out benchmark; you may use that benchmark to *design* filters, but no
validation text may enter the training set ([@ch:proj4]).

## Mixing sources and data schedules {#sec:data:mixing}

A corpus is a mixture of sources (web, code, papers, books, math, conversations). With token counts
$n_s$ and a training budget of $N$ tokens, a mixture $p$ implies
$$\text{epochs}_s = \frac{p_s\,N}{n_s}.$$ Common baselines are *proportional* ($p_s\propto n_s$),
*uniform*, and *temperature* sampling ($p_s\propto n_s^{\alpha}$ with $\alpha\in[0,1]$). Upweighting
high-quality sources helps until it forces many repetitions: repeating data a few times costs
little, but tens of epochs over a small source lead to overfitting and wasted compute (Muennighoff
et al., 2023). **UniMax** (Chung et al., 2023) makes the cap explicit: sample sources uniformly, but
never more than $C$ epochs of any source, redistributing the excess, as in
[@lst:data:unimax] and [@fig:data:unimax].

``` {#lst:data:unimax .python caption="UniMax as water-filling (from `llmsys/datapipe/mixing.py`)."}
def unimax(sizes, budget, max_epochs):
    remaining, alloc = budget, {}
    order = sorted(sizes, key=lambda s: sizes[s])    # smallest first
    for i, s in enumerate(order):
        share = remaining / (len(order) - i)         # equal share of rest
        alloc[s] = min(share, max_epochs * sizes[s]) # ... capped
        remaining -= alloc[s]
    total = sum(alloc.values())
    return {s: alloc[s] / total for s in sizes}
```

![Allocating a 1T-token budget across five hypothetical sources of 1000, 100, 20, 10 and 2.2 billion
unique tokens (illustrative sizes). Proportional mixing gives the small high-quality sources almost
nothing. Uniform mixing would repeat the books source about 90 times. UniMax with an 8-epoch cap
gives books, math and papers exactly 8 epochs each and splits the remaining budget evenly between
web and code (computed with
`llmsys.datapipe.mixing.unimax`).](figures/ch19_pretraining_data-7.svg){#fig:data:unimax}

##### Learning a mixture.

More principled approaches train many small models on random mixtures, for example drawn from a
Dirichlet, fit a regression from mixture weights to a target metric, and optimise the regression
(e.g. RegMix, Liu et al., 2024). This relies on two hopes: the regression is accurate near its
optimum, and the optimal mixture at small scale transfers to large scale. The second fails in a
predictable way. Small runs never exhaust a small source, so they over-reward upweighting it.
*Simulated epoching* downsamples every source in proportion, so that small runs repeat data as much
as the large run would. The general theme, which recurs in [@ch:scalinglaws], is to make
small-scale experiments look like the large-scale setting before trusting them.

##### Schedules.

The mixture need not be constant. Many recipes end pretraining with an annealing or mid-training
phase on high-quality and domain data while the learning rate decays, which is cheap and effective
([@ch:optim; @ch:scalinglaws]).

## Synthetic and post-training data {#sec:data:synthetic}

Two uses of model-generated data now appear throughout the pipeline. In pretraining, a strong model
can *rephrase* low-quality pages into cleaner prose and turn high-quality pages into
question--answer pairs or summaries (Nemotron-CC), recovering tokens that filters would discard. In
post-training, the standard recipe is to define environments and prompts, then collect responses
from a teacher, sometimes many per prompt, and filter them. OpenThoughts (2025), for example,
reported that sampling several responses per prompt helped and that the strongest model was not
necessarily the best teacher. For software engineering, synthetic tasks are mined or generated from
real repositories (SWE-smith, SWE-rebench); executable environments are expensive to build, which
has motivated datasets of trajectories that do not require execution.
[@ch:instruct; @ch:distill] use such data. Two cautions apply. First, the licence and
provenance status of text generated by models trained on unlicensed data is unclear. Second,
synthetic data inherits its teacher's errors and blind spots, so it needs the same filtering,
deduplication and decontamination as web text.

## Engineering the pipeline {#sec:data:engineering}

##### Order stages by cost and selectivity.

Each stage runs only on what earlier stages kept, so put cheap, highly selective stages first: empty
and length checks, then rules, then language ID, then classifiers, then (global) dedup. A useful
mental model is the *funnel* ([@fig:data:funnel]): track documents and characters remaining
after each stage, and the time spent in each. On our synthetic crawl the classifier removed nothing
because the rules had already removed every spam page; on a real crawl the classifier typically
removes a large fraction of rule-passing pages. Only measurement tells you which situation you are
in.

![A filtering funnel measured on 800 pages of the synthetic crawl used in Lab 14
(`llmsys.datapipe.synth`), with the stages in the order shown. About 60% of characters survive. Real
crawls lose far more at every stage; the point of the plot is the habit of drawing
it.](figures/ch19_pretraining_data-8.svg){#fig:data:funnel}

##### Per-shard versus global work.

Stages 2--6 of [@fig:data:roadmap] are pure functions of one document, so a crawl's
thousands of files can be processed independently by a process pool on one machine, a Slurm array or
`submitit` job, Modal's `.map`, Ray, or Spark. Dedup, decontamination and mixing need corpus-level
information and therefore run as separate passes after filtering ([@fig:data:arch]).

![Pipeline architecture. Per-document stages run as independent, idempotent shard jobs that write
atomically and record completion in a manifest. Corpus-level stages run afterwards over the filtered
output, which is much smaller than the input.](figures/ch19_pretraining_data-9.svg){#fig:data:arch}

##### Make every job idempotent and resumable.

Long data jobs fail: preempted nodes, corrupt records, out-of-memory errors on pathological pages.
Four habits make failures cheap. (1) Write each output to a temporary name and `os.replace` it into
place, so a crash never leaves a half-written shard that looks finished. (2) Record finished shards
and their counters in a manifest, and skip them on restart. (3) Catch exceptions *per record*, count
them, and continue: one malformed page must not kill a shard. (4) Make randomness (sampling,
survivor choice, train/validation splits) a deterministic function of document ids and a seed, so
that reruns reproduce outputs. [@lst:data:shards] shows the worker of
`llmsys.datapipe.pipeline.run_shards`.

``` {#lst:data:shards .python caption="The shard worker: per-process pipeline, atomic write, returned counters (from `llmsys/datapipe/pipeline.py`)."}
def _process_one(args):
    in_path, out_path, factory, reader = args
    pipe = factory()                     # load classifiers ONCE per process
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for doc in reader(in_path):      # streaming: O(1 record) memory
            out = pipe.process(doc)      # per-stage seen/kept/chars/time accounting
            if out is not None:
                f.write(json.dumps({"url": out.url, "text": out.text,
                                    "meta": out.meta}) + "\n")
    os.replace(tmp, out_path)            # atomic: never a half-written shard
    return {"shard": os.path.basename(in_path),
            "stages": {k: vars(v) for k, v in pipe.stats.items()}}

with cf.ProcessPoolExecutor(max_workers=workers) as ex:     # in run_shards
    for fut in cf.as_completed([ex.submit(_process_one, j) for j in jobs]):
        results.append(fut.result())
```

::: pitfall
[Process pools and big models]{.box-title}

Pass a *factory* to workers, not a loaded model: pickling a 1 GB classifier to every task is slow,
and with the `spawn` start method (the default on macOS and Windows) everything the task references
must be importable and picklable. Load models in a worker initializer or lazily in a module-level
cache. Use `len(os.sched_getaffinity(0))`, not `os.cpu_count()`, to size the pool inside containers
and Slurm allocations. Monitor memory: $W$ workers each holding a fastText model and a large page
can exceed node RAM even when one worker is fine.
:::

::: napkinmath
[How long does a crawl take?]{.box-title}

Assume $2.5\times10^{9}$ pages at an average of 20 KB of HTML each, which is 50 TB of HTML. If the
per-document stages cost 5 ms per page per core (compiled extractor plus a few linear classifiers),
the job needs $2.5\times10^{9}\times5\,\text{ms}\approx3.5\times10^{3}$ core-hours: about 3.5 hours
on 1,000 cores, or nine days on one 16-core machine. Pure-Python stages at 50 ms per page multiply
both numbers by ten. Measure seconds per shard on a sample of shards, multiply by the number of
shards, and divide by the number of cores before committing to a design. Project 4 asks exactly this
question for its subset and for a full crawl.
:::

##### Tokenise last, once.

The final stage encodes kept documents, appends the end-of-text token after each one, and writes
flat `uint16` (or `uint32` for vocabularies above 65,536) arrays, exactly the shard format of
[@sec:text:pipeline]. Parallelise over documents with `pool.imap(..., chunksize=...)` and
write per-worker shards rather than gathering a Python list of billions of integers in one process.
One terabyte of text is roughly $2.5\times10^{11}$ GPT-2 tokens at about four bytes per token, which
is 500 GB as `uint16`.

##### Validate data with models.

The ultimate test of a filter is the loss or accuracy of a model trained on its output. DCLM and
FineWeb institutionalised this as *data ablations*: fix the architecture, the training recipe and a
small compute budget, then vary only the data and compare downstream metrics. Keep the comparison
fair (same token budget, same tokenizer, decontaminated evaluation) and remember the scale caveat of
[@sec:data:quality].

::: systemslens
[Throughput accounting for a data pipeline]{.box-title}

Report, per stage: documents and characters in and out, seconds per document, and the fraction of
total time. Report for the job: input MB/s, documents/s, and parallel efficiency
($\text{speedup}/\text{workers}$). Efficiency well below one usually means I/O contention (many
workers reading gzipped files from one network disk), memory pressure (swapping), skew (a few shards
much larger than the rest), or per-task overhead (tiny tasks that ship large models). Lab 14's
`bench.py scaling` measures this curve; run it on a machine you are not sharing.
:::

## Summary {#sec:data:summary}

- Data goes from live services through crawls and dumps to processed corpora. Each step discards
  most of its input and carries legal and ethical constraints, so keep provenance with every
  document.

- Common Crawl's WARC/WAT/WET files share one record container. Gzip-per-record makes them
  splittable and robust; WET text is convenient, while re-extracting from WARC gives more control.

- Extraction is two problems: decoding (a charset cascade) and content selection (DOM heuristics).
  It is lossy and the most expensive stage per byte.

- Language ID, toxicity and quality filters all follow one recipe: a cheap linear classifier over
  hashed $n$-grams scoring raw documents for similarity to a target set. The threshold is a
  token-budget decision, and the right filtering strength depends on scale.

- Gopher-style rules are cheap and interpretable; report which rule fired.

- Exact dedup uses hashes, two-pass counting and Bloom filters; near dedup uses MinHash
  ($\Pr[\text{collision}]=J$), LSH banding (S-curve $1-(1-s^r)^b$), verification and union-find. At
  scale it is a map--shuffle--reduce job.

- Decontaminate training data against evaluation sets with long $n$-grams; never let validation text
  into training data.

- Mixtures must respect epochs; UniMax caps repetition, and small-scale mixture search must simulate
  large-scale epoching.

- Engineer for failure and visibility: stages ordered by cost, a factory per worker, atomic writes,
  manifests, per-stage counters, deterministic randomness, and data ablations to validate.

## Exercises {#sec:data:exercises}

::: exercise
[Charset cascade]{.box-title}

Construct three byte strings on which naive decoding, `body.decode("utf-8", errors="replace")`,
produces visibly wrong text but the cascade of [@lst:data:extract] succeeds. Then construct
one on which the cascade itself fails, and explain what extra signal would fix it.
:::

::: exercise
[Naive Bayes overconfidence]{.box-title}

Show that if a document is the concatenation of $m$ copies of a short text, the Naive Bayes log-odds
between two languages scales linearly in $m$. What does this imply for using $p(\ell\mid x)$ as a
confidence score on long documents? Propose a length-normalised score.
:::

::: exercise
[PII regex audit]{.box-title}

Write ten strings that the phone regex in [@lst:data:pii] masks but should not, or misses
but should catch, including international formats and extensions. Measure precision and recall of
the regex on your list, then improve it without adding false positives.
:::

::: exercise
[Which rule does the work?]{.box-title}

Run `gopher_report` on the Lab 14 synthetic crawl and tabulate, for each page kind, how often each
rule fires. Which single rule removes the most pages? Which rules are redundant on this data, and
would you expect them to be redundant on a real crawl?
:::

::: exercise
[Hashing-trick collisions]{.box-title}

For the classifier of [@eq:data:hashed] with $B$ buckets and a document with $F$ distinct
features, compute the expected number of colliding feature pairs. For $B=2^{18}$ and $F=2000$, is
the effect on the score likely to matter? How does the answer change across the corpus?
:::

::: exercise
[MinHash variance]{.box-title}

Prove that the MinHash estimator $\hat J = \frac1k\sum_i \mathbb{1}[\min_S h_i=\min_T h_i]$ is
unbiased with variance $J(1-J)/k$, assuming independent hash functions. How many hashes give a
standard deviation of 0.02 at $J=0.8$? Verify empirically with `MinHasher`.
:::

::: exercise
[Designing the S-curve]{.box-title}

You want pairs with $J\ge0.85$ to be candidates with probability at least 0.98, and pairs with
$J\le0.5$ with probability at most 0.01. Find the smallest $k=br$ and a $(b,r)$ split that achieves
both, using [@eq:data:scurve]. Plot your curve against [@fig:data:scurve].
:::

::: exercise
[Giant buckets]{.box-title}

Suppose 0.1% of $10^{9}$ documents are near-identical cookie-consent pages that share every band.
How many candidate pairs does naive bucketing generate? Design a bucket-size cap and a modified
reduce step that still removes these duplicates in linear time.
:::

::: exercise
[Bloom filter sizing]{.box-title}

Paragraph-level dedup must handle $3\times10^{10}$ paragraphs with a false-positive rate of at most
$10^{-3}$. Compute $m$, $k$ and the memory. If the filter must be sharded across 16 machines, how
should paragraphs be routed, and what is the communication volume?
:::

::: exercise
[UniMax by hand]{.box-title}

For sources of 500B, 50B, 5B and 0.5B tokens, a 300B-token budget and a cap of 3 epochs, compute the
UniMax weights and the epochs per source. Compare with temperature sampling at $\alpha=0.3$.
:::

::: exercise
[Crawl-time estimate]{.box-title}

Measure seconds per shard for Lab 14's pipeline on your machine with `bench.py stages`, then
estimate the wall time to process a 90,000-file crawl on (a) your laptop, (b) a 64-core server, and
(c) 1,000 cloud vCPUs. Which stage would you rewrite first, and in what language?
:::

## Lab 15: A pretraining-data pipeline {#sec:data:lab}

::: labbox
[A pretraining-data pipeline from raw crawl to clean text]{.box-title}

**Code:** `code/labs/lab19_datapipe`

**Goals**

Build every stage of a web-data pipeline on a synthetic, labelled crawl, verify each stage against
ground truth, and measure what each stage costs and removes.

**Setup**

``` bash
cd code
pytest -q labs/lab19_datapipe                     # grades starter.py
LAB_IMPL=solution pytest -q labs/lab19_datapipe   # reference
```

The crawl comes from `llmsys.datapipe.synth`: articles wrapped in site boilerplate, foreign-language
pages, keyword spam, login walls, product listings, filled-in licence templates (near duplicates)
and mirrors (exact duplicates), with planted contact details from reserved ranges and some pages in
cp1252. Everything is pure Python and NumPy.

**Tasks**

1.  `detect_charset`: the cascade of [@sec:data:extract].

2.  `strip_html`: a regex extractor that drops non-content blocks, turns block tags into newlines,
    unescapes entities and collapses whitespace.

3.  `char_ngram_counts`, `nb_train`, `nb_posterior`: the Naive Bayes language identifier of
    [@eq:data:nb], with log-sum-exp normalisation.

4.  `mask_pii`: emails, IPv4 addresses and NANP phone numbers, with counts per type.

5.  `rule_failures`: Gopher-style rules returning the names of the rules that fired.

6.  `hashed_ngrams`, `train_quality`, `quality_score`: the hashed $n$-gram logistic classifier of
    [@eq:data:hashed].

7.  `exact_line_dedup`: two-pass, hash-counted line deduplication.

8.  `minhash_signature`, `lsh_pairs`, `cluster_keep`, `near_dedup`: MinHash with universal hashing,
    LSH banding, union-find clustering, verified near-dedup.

9.  `Bloom` and `bloom_fp_rate`: a double-hashing Bloom filter and [@eq:data:bloom].

10. `unimax`: epoch-capped mixture weights.

11. `filter_document` and `run_pipeline`: per-shard filtering in a process pool, then global
    near-dedup; every input document must be accounted for in the counters.

**Measurements**

- `python labs/lab19_datapipe/bench.py stages`: per-stage ms/doc and drop rates. Which stage
  dominates? (In our noisy measurements the pure-Python language ID cost several times more per
  document than the rules.) Reorder the stages and confirm that the total time changes but the
  output does not.

- `bench.py scaling --workers 1 2 4 8`: speedup and parallel efficiency. Explain any deviation from
  linear scaling.

- `bench.py lsh`: reproduce [@fig:data:scurve] numerically; then sweep `bands` in
  `near_dedup` on the synthetic crawl and plot precision and recall against the ground-truth
  duplicate groups.

- Optional: `run.py` on one real WARC or WET file with FastWARC, Resiliparse and fastText installed.
  Read 25 random kept and dropped documents and annotate them.

**Deliverables**

Passing tests; a funnel plot like [@fig:data:funnel] for your pipeline; a per-stage cost
table; the scaling curve; the LSH precision and recall sweep; and a paragraph on what you would
change for a real crawl.

**Stretch goals**

Replace the Python MinHash loop with a fully vectorised batch version and measure the speedup; add a
KenLM-style $n$-gram perplexity filter trained on the positive set; implement document packing and
tokenisation into a `uint16` shard with `llmsys.datapipe.pipeline.tokenize_to_bin` and train the
Lab 13 model on your filtered output.
:::
