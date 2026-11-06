## Chapter 33: Self-Refinement and Scoring {#chapter-33-self-refinement-and-scoring .unnumbered}

#### Exercise 33.1

With `logits[1:]`, the row used for target token $i{+}1$ is the distribution for token $i{+}2$, so
every answer token is scored under the wrong context (and the gather still succeeds, because shapes
match after the targets are sliced accordingly). The values are log-probabilities of real tokens
under some distribution, hence plausible negative numbers. `test_token_logprobs_match_incremental`
catches it, because it compares each entry with a token-by-token incremental computation.

#### Exercise 33.2

$\log\sum_j e^{z_j} = \log\bigl(e^M\sum_j e^{z_j - M}\bigr) = M + \log\sum_j
e^{z_j-M}$. All exponents are $\le 0$, so no term overflows; the term with $z_j = M$ equals 1, so
the sum lies in $[1, V]$ and its log in $[0, \log V]$ --- no underflow to $\log 0$. The output for
the argmax entry is $M - \mathrm{LSE} = -\log\sum_j e^{z_j - M} \in [-\log V, 0]$, and it is the
largest output entry.

#### Exercise 33.3

Appending tokens adds terms $\log p < 0$, so the joint log-probability strictly decreases. The
average can increase: an answer with token log-probs $(-3, -3)$ (mean $-3$) extended by
$(-0.1, -0.1)$ has mean $-1.55$. To compare a 50-token and a 500-token solution use a
length-normalized score (mean log-prob or perplexity), ideally restricted to the final-answer tokens
if you care about the result rather than the prose.

#### Exercise 33.4

Compute $\bar\ell$ for each answer under the original prompt, grade with the verifier, bin by score
decile and plot per-bin accuracy. A useful scorer shows accuracy rising with the decile; small base
models are often only weakly calibrated, and the minimum-token or boxed-tokens-only variants can be
more informative than the plain mean. Report the area under the ROC curve of score versus
correctness as a single-number summary.

#### Exercise 33.5

Loop over position chunks of size $C$: run the model once to get the hidden states (or logits), and
for each chunk compute `log_softmax` in float32 on $C$ rows and gather the targets, keeping only the
gathered values. Peak extra memory drops from $TV\times 4$ B ($\approx 2.5$ GB at $T = 4096$) to
$CV\times 4$ B ($\approx 39$ MB at $C = 64$). Going further and chunking the final projection avoids
materializing full logits at all.

#### Exercise 33.6

Report the three rates per scorer. Typically: no scorer has the highest harm rate (correct drafts
broken by spurious critiques); the heuristic scorer mainly rejects revisions that lose the box or
ramble; the log-prob scorer rejects many revisions and therefore has a low harm rate but also a low
fix rate. The best ratio of fixes to harms is model-dependent, which is the point of measuring it.

#### Exercise 33.7

Ask the draft to state a checkable consequence (e.g. substitute the answer into the original
equation) and evaluate it with SymPy; if the check fails, put "Substituting your answer gives ...,
which is not equal to ..." into the review. With external feedback, revisions are triggered only by
real errors and carry specific information, which typically improves the fix rate and nearly
eliminates harm on correct drafts.

#### Exercise 33.8

Order each call as `[question][draft][critique][instruction]` (or run the loop as one growing
conversation), so each call's prompt is the previous call's prompt plus its output plus a short
instruction. Then, with perfect prefix caching, each call prefills only its new instruction: total
prefill $\approx q = 150$ tokens instead of $q + k(2q + 2a + c) = 150 +
2(300 + 1000 + 120) = 2990$, a saving of $\approx 2840$ tokens (95%). Decode work is unchanged.

#### Exercise 33.9

At $\approx$`<!-- -->`{=html}2000 decode tokens, best-of-4 decodes four 500-token answers in one
batched call and scores them with four prefills; refinement with $k = 2$ makes five sequential
calls. Expect best-of-4 to have roughly a third of the latency; which is more accurate depends on
the scorer's calibration (best-of-$n$) versus the critique's quality (refinement), and must be
measured.
