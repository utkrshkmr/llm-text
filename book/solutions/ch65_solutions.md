## Chapter 48: Alignment, Safety, and Multimodality {#chapter-48-alignment-safety-and-multimodality .unnumbered}

#### Exercise 48.1

With 600,000 samples (computed with `mixture_weights` and `epochs_per_source`): $\alpha=1$: chat
87.3%, code 8.7%, math 3.5%, safety 0.52%; every source is seen 1.05 times. $\alpha=0.5$: 62.8 /
19.8 / 12.6 / 4.9%; epochs 0.75 / 2.4 / 3.8 / 9.7. $\alpha=0.3$: 47.7 / 23.9 / 18.2 / 10.3%; epochs
0.57 / 2.9 / 5.5 / 20.6. None of the three meets both constraints: safety under 5 epochs needs
$p_{\text{safety}}\le 5\cdot3000/600000=2.5\%$, and at least 1% of samples needs $p\ge1\%$. Scanning
$\alpha$ gives roughly $0.67\lesssim\alpha\lesssim0.87$ (for example $\alpha=0.75$: 1.67%, 3.35
epochs). Alternatively keep $\alpha=0.5$ and cap safety at 2%: `caps={"safety": 0.02}` redistributes
the excess proportionally (chat 64.6%, code 20.4%, math 12.9%) and gives exactly 4.0 epochs of
safety data.

#### Exercise 48.2

Over lengths $L\in[200,1800]$: (a) padding each conversation to the next multiple of $m$ wastes
$1-\mathbb{E}[L]/\mathbb{E}[\lceil L/m\rceil m]\approx20\%$ of tokens for $m=512$ and $51\%$ for
$m=2048$, since every conversation pads to 2048. (b) Under concat-and-chunk, a document of length
$L$ placed at a uniformly random offset crosses a boundary with probability $\min(1,(L-1)/m)$: about
94% of conversations are split for $m=512$ and 49% for $m=2048$. Short rows waste nothing under
packing but truncate the context of almost every example; long rows split fewer examples. Choose $m$
at least as large as a typical conversation and use concat-and-chunk or packing rather than padding;
use a block mask if cross-document attention matters.

#### Exercise 48.3

With the book's byte-level BPE trained on text containing `Answer:**yes**`,
`encode("Answer:" + "**yes**")` gives 4 ids while `encode("Answer:") + encode("**yes**")` gives 5:
the pre-tokeniser treats the punctuation run `:**` as one chunk in the full string, so a learned
merge spans the boundary. The prompt alone ends with a separate `:` token, so `len(encode(prompt))`
counts one token that no longer exists in the full encoding. The mask then covers the first response
token (or leaves a prompt token unmasked, depending on the merge). Encoding the two parts separately
([@lst:align:stream]) or templates that end with whitespace avoid it.

#### Exercise 48.4

If $p+b\le1$ and $1-p+b\le1$, then $p_1=p+b$ and $p_2=1-(1-p+b)=p-b$, whose average is $p$. If
$p+b>1$ (and $b\le p$), $p_1=1$ and the average is $(1+p-b)/2<p$: clipping destroys the symmetry, so
strong preferences are pulled towards 0.5. Fixes: (i) estimate the bias directly by asking the judge
to compare a response with *itself* (true $p=0.5$) and correct in logit space, since a bias that is
additive in logits is never clipped; (ii) read the judge's token log-probabilities for "1"/"2" and
average logits instead of clipped probabilities; (iii) report agreement between the two orders and
treat disagreements as ties.

#### Exercise 48.5

With true $\theta_0=0$ and $\theta_1=2$ (20 repetitions; computed with
`length_controlled_win_rate`), the mean LC estimate and its standard deviation were $0.511\pm0.058$
($n=50$), $0.505\pm0.034$ ($n=200$) and $0.495\pm0.019$ ($n=800$). The estimate is essentially
unbiased and its noise falls like $1/\sqrt n$, while the raw win rate is far from 0.5 whenever
lengths differ systematically. With little overlap around $\Delta\ell=0$ the intercept is an
*extrapolation*: $\theta_0$ and $\theta_1$ become strongly correlated and the LC estimate's variance
explodes, or it is driven by the functional form (the $\tanh$) rather than the data. Report the
length distribution and refuse to interpret LC win rates without overlap.

#### Exercise 48.6

For a 1.7B policy: MCQ prefill $2\times1.7\times10^9\times400\times14{,}042\approx1.9\times10^{16}$
FLOPs, about 39 s at 50% of an H100; chat decode reads about 3.4 GB per step, roughly 1.0 ms per
step, so about 2 s for $4\times512$ steps. An 8B judge with order swapping:
$2\times8.03\times10^9\times1000\times1610\approx2.6\times10^{16}$ FLOPs, about 52 s. The judge
still dominates. Judging costs the same as all generation (about 41 s) when
$2N\cdot1000\cdot1610=41\times0.5\times989\times10^{12}$, that is $N\approx6\times10^9$ parameters.
Any judge larger than about 6B makes judging the bottleneck for a policy of this size.

#### Exercise 48.7

95% Wilson intervals: A unsafe 7/100: $[3.4\%,13.7\%]$; A over-refusal 4/100: $[1.6\%,9.8\%]$; B
unsafe 3/100: $[1.0\%,8.5\%]$; B over-refusal 12/100: $[7.0\%,19.8\%]$. The unsafe intervals overlap
heavily, so the data do not support "B is safer". B's higher over-refusal is also not significant at
this size. A two-proportion test for 7% versus 3% at $\alpha=0.05$ and 80% power needs
$n\approx(1.96+0.84)^2\,(0.07\cdot0.93+0.03\cdot0.97)/0.04^2\approx460$ prompts per checkpoint.
Paired designs (same prompts, McNemar's test, [@ch:eval]) need somewhat fewer.

#### Exercise 48.8

The guard's prefill is $2\times10^9\times300=6\times10^{11}$ FLOPs, about 1.2 ms at 50% of peak, and
reading 2 GB of weights takes about 0.6 ms, so in practice a few milliseconds once kernel-launch
overheads are included (use CUDA graphs, [@ch:inference]). The main model's prefill is
$2\times8\times10^9\times300=4.8\times10^{12}$ FLOPs, about 10 ms. (a) Running the guard first adds
its whole latency, a few ms, to TTFT. (b) Running it concurrently adds roughly nothing, because the
guard finishes before the main prefill. A flagged request wastes the main model's prefill (about
10 ms of GPU time) plus its KV allocation, and the server must hold back the first streamed token
until the guard verdict arrives.

#### Exercise 48.9

Each rank holds $32768/64=512$ pairs. (a) CLIP: all-gathered text and image embeddings are
$2\times32768\times1024\times2$ B $\approx134$ MB; the rank's rows of the image-to-text logits in
FP32 are $512\times32768\times4\approx67$ MB, and the same again for text-to-image, about 134 MB in
total, plus gradients of the same size. (b) Ring-chunked SigLIP holds its own embeddings (about
2 MB), one incoming text chunk (1 MB) and one $512\times512$ FP32 logit block (1 MB); 63 ring steps
pass the chunks around. The lab test checks that `siglip_loss_chunked` equals `siglip_loss` to
$10^{-5}$.

#### Exercise 48.10

Any two requests with identical text and different images collide under an ids-only hash, and
`multimodal_block_hashes` separates them because their first image block hashes `img:A` versus
`img:B` (the lab test does exactly this). With two adjacent images and no token between them, the
function sees one run of placeholders and tags *all* of them with the first image's hash: swapping
the second image then produces a false hit. Inserting `<img_start>`/`<img_end>` marker tokens around
each image (as real chat templates do) breaks the run, so each image gets its own tag.
