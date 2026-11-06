## Chapter 14: Pretraining Data: From Common Crawl to Tokens {#chapter-14-pretraining-data-from-common-crawl-to-tokens .unnumbered}

#### Exercise 14.1

\(a\) The cp1252 bytes `caf\xe9` with no declaration: naive UTF-8 decoding yields "caf" followed by
the replacement character U+FFFD; the cascade sees invalid UTF-8 and falls back to cp1252, giving
"café". (b) A page whose HTTP header says `charset=iso-8859-1` and contains `\xe9`: the header wins
and the text decodes correctly. (c) A UTF-8 page with a byte-order mark: naive decoding keeps a
stray U+FEFF at the start; the BOM rule strips it. A failure case is a Shift-JIS or GBK page with no
declaration: it is invalid UTF-8, so the cascade falls back to cp1252 and produces mojibake. The
missing signal is a statistical detector that scores byte-pair frequencies under each candidate
encoding (as chardet or Resiliparse do), possibly combined with language ID of the decoded result.

#### Exercise 14.2

With independent features, the log-odds between languages $a$ and $b$ is
$\log\frac{p(a)}{p(b)} + \sum_g c_g(x)\log\frac{p(g\mid a)}{p(g\mid b)}$. Concatenating $m$ copies
multiplies every $c_g$ by $m$, so the evidence term scales by $m$ and the posterior saturates to 0
or 1 regardless of how ambiguous each copy is. Posteriors on long documents are therefore not
calibrated confidences. A length-normalised score divides the evidence by the number of $n$-grams,
$\frac{1}{|x|}\sum_g c_g\log\frac{p(g\mid a)}{p(g\mid b)}$ (the average log-likelihood ratio per
feature), and thresholds that, or calibrates it with logistic regression on held-out labelled data
(Platt scaling).

#### Exercise 14.3

False positives include long numeric identifiers with separators (`123-456-7890` used as an order
number), dates such as `2024 555 0123` in tables, and fax numbers that you may decide to keep. False
negatives include extensions (`x123`), international numbers (`+44 20 7946 0958`), vanity numbers
(`1-800-FLOWERS`), and numbers split across lines by extraction. Measure precision as
masked-and-correct over all masked strings, and recall as masked-and-correct over all true phone
numbers in your list. Improvements that keep precision include requiring context words ("call",
"tel", "phone") for ambiguous formats, adding an extension group, and accepting `+\d{1,3}` country
codes only when followed by a plausible national format.

#### Exercise 14.4

On 400 synthetic pages, every rule-failing page belongs to spam, listings, login pages or foreign
pages. Spam fails the symbol-ratio, ellipsis, alphabetic-word and stop-word rules (all 41 pages);
listings fail symbol-ratio, bullet, alphabetic-word and stop-word rules (31/31); login pages fail
word count and stop words (26/26); foreign pages fail only the stop-word rule (57/57). The stop-word
rule alone removes every bad page here, so the other rules are redundant on this crawl. On a real
crawl they are not: SEO spam and many menus contain stop words, short English pages fail only the
word-count rule, and repetition rules catch templated pages that pass everything else.

#### Exercise 14.5

Each of the $\binom{F}{2}$ feature pairs collides with probability $1/B$, so the expected number of
colliding pairs is $\binom{F}{2}/B = 2000\cdot1999/2/2^{18}\approx 7.6$. With 2,000 features
averaged into one logit, a handful of shared buckets shifts the score negligibly. Across the corpus,
however, collisions are between *features*, not documents: a frequent feature colliding with a
strongly weighted rare feature biases every document containing it. Larger $B$ or signed hashing (a
second hash choosing $\pm1$) reduces this bias.

#### Exercise 14.6

Let $X_i=\mathbb{1}[\min_S h_i = \min_T h_i]$. By [@prop:data:minhash],
$\mathbb{E}[X_i]=J$, so $\mathbb{E}[\hat J]=J$. Each $X_i$ is Bernoulli($J$) with variance $J(1-J)$,
and independence gives $\mathrm{Var}(\hat J)=J(1-J)/k$. For a standard deviation of 0.02 at $J=0.8$:
$k = 0.16/0.0004 = 400$. Empirically, repeat `MinHasher(400).signature` with different seeds on a
pair of sets with known $J$ and compare the sample standard deviation with 0.02.

#### Exercise 14.7

A brute-force search over $k=br$ finds the smallest feasible design at $k=312$ with $b=26$, $r=12$:
$P(0.85)=0.981$ and $P(0.5)=0.0063$. Intuitively, $r$ must be large enough to suppress $J=0.5$
($0.5^{12}\approx2.4\times10^{-4}$ per band) while $b$ restores recall at 0.85. Plotting
$1-(1-s^{12})^{26}$ shows a curve much steeper than the $k=128$ curves of
[@fig:data:scurve], centred near $(1/26)^{1/12}\approx0.76$.

#### Exercise 14.8

$10^{6}$ identical pages in one bucket produce $\binom{10^6}{2}\approx5\times10^{11}$ pairs per
band, which is infeasible. Cap bucket sizes at $M$ (say $10^{3}$). For an oversized bucket, do not
enumerate pairs: emit $(\text{first},i)$ edges from one representative to every member, which is
linear in bucket size, optionally after verifying a random sample of members against the
representative. Union-find then merges the whole bucket into one cluster in $O(n\,\alpha(n))$.

#### Exercise 14.9

Bits per item $= -\ln 10^{-3}/(\ln2)^2\approx14.4$, so $m\approx4.3\times10^{11}$ bits
$\approx54$ GB, with $k\approx10$. To shard over 16 machines, route each paragraph by a hash prefix
to one shard that owns its own filter of about 3.4 GB. Each paragraph then needs only one network
message carrying its 8--16-byte hash, about $3\times10^{10}\times16$ B $\approx
0.5$ TB in total, and all $k$ bit lookups for that paragraph are local to its shard.

#### Exercise 14.10

UniMax (budget 300B, cap 3): the 0.5B source gets 1.5B (3 epochs), the 5B source 15B (3 epochs), and
the remaining 283.5B is split equally between the two large sources, 141.75B each. The weights are
$(0.4725, 0.4725, 0.05, 0.005)$ and the epochs $(0.28, 2.84, 3, 3)$. Temperature sampling with
$\alpha=0.3$ gives weights $(0.532,0.267,0.134,0.067)$ and epochs $(0.32, 1.6, 8.0, 40.2)$: the
smallest source would be repeated 40 times, which is exactly what UniMax prevents.

#### Exercise 14.11

Let $t$ be the measured seconds per shard on one core and $S=90{,}000$ shards. With $c$ effective
cores the time is roughly $S t / (c\,\eta)$, where $\eta$ is the parallel efficiency from
`bench.py scaling` (often 0.6--0.9 on one machine, lower when I/O-bound). For example, if $t=60$ s:
a 4-core laptop needs about 15 days at $\eta=1$, a 64-core server about a day, and 1,000 vCPUs about
1.5 hours. Rewrite first the stage that dominates the per-stage table. In our measurements that is
pure-Python language ID and extraction, so replace them with fastText and Resiliparse (C++) or a
vectorised implementation before considering anything else.
