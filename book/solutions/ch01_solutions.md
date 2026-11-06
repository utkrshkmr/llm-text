## Chapter 1: The Landscape {#chapter-1-the-landscape .unnumbered}

#### Exercise 1.1

By the definition of conditional probability,
$p(x_1,\dots,x_T)=p(x_T\mid x_{<T})\,p(x_1,\dots,x_{T-1})$; applying this recursively down to
$p(x_1)$ gives the product. No modelling assumption is involved, so an exact autoregressive model
can represent any distribution. With a causal mask, the output at position $t$ depends only on
$x_{\le t}$, so position $t$ is a valid prediction of $x_{t+1}$ for every $t$: one forward pass over
$T$ tokens yields $T$ supervised predictions (the last uses the next token in the stream). A masked
LM computes a loss only at the $\approx0.15T$ masked positions, because unmasked positions can see
their own answer through bidirectional attention.

#### Exercise 1.2

Perplexity is $e^{2.0}\approx7.39$: the model is, on average, as uncertain as if it were choosing
uniformly among about 7.4 tokens. A uniform model has loss $\ln 50{,}257\approx10.82$ nats and
perplexity 50,257. This is a useful sanity check: a freshly initialised model should start near
10.8.

#### Exercise 1.3

Wikipedia: $0.03\times300\text{B}=9\text{B}$ tokens sampled from a 3B-token source, about 3 epochs.
Common Crawl: $0.60\times300\text{B}=180\text{B}$ from 410B, about 0.44 epochs. Repeating small,
clean, information-dense sources raises the quality of the effective training distribution. The cost
is a risk of memorisation and diminishing returns after a few epochs, so heavy repetition is
reserved for the highest-quality data.

#### Exercise 1.4

$C=64\times30\times86{,}400\,\text{s}\times989\times10^{12}\times0.35\approx5.74\times10^{22}$
FLOPs. With $C=6ND$ and $D=20N$, $C=120N^2$, so $N=\sqrt{C/120}\approx2.2\times10^{10}$ (about 22B
parameters) and $D\approx4.4\times10^{11}$ tokens. If the model will be served heavily, total cost
is training plus $2N$ FLOPs per served token. Shrinking $N$ (say to 7B) and spending the same
compute on more tokens ($D=C/(6N)\approx1.4$T) gives a slightly worse model at equal compute but a
$3\times$ cheaper one to serve, which is usually the better trade.

#### Exercise 1.5

bf16: weights $4.02\times10^9\times2=8.04$ GB, KV at 4,096 tokens $=147{,}456\times4{,}096=0.60$ GB,
total 8.64 GB, giving $8.64/3.35\times10^{12}\approx2.58$ ms, or about 387 tokens/s. int4 weights:
2.01 GB + 0.60 GB = 2.61 GB, about 0.78 ms, or about 1,280 tokens/s. Weights still dominate, but the
KV share of traffic rises from 7% to 23%; at longer contexts or larger batches the KV cache becomes
the dominant term, which motivates KV-cache quantisation ([@ch:efficient]).

#### Exercise 1.6

Per step, compute time is $B\cdot2N/F$ and memory time is $Nb/W$ (weights read once, shared by the
batch). Equating gives $B^\star=bF/(2W)$. For an H100 in bf16,
$B^\star=2\times989/(2\times3.35)\approx295$, matching the lab's 296--320 (the lab includes the LM
head and attention FLOPs and embedding bytes, which shift the value slightly). With long contexts
each sequence adds its own KV bytes, which are *not* shared across the batch. Attention over the
cache has an operational intensity of $H/G$ FLOP/byte in bf16 ([-@eq:pytorch:decodeI]): 2 for
Qwen3-0.6B and 1.7B, 4 for Qwen3-4B and 8B, regardless of batch size. That is two orders of
magnitude below the ridge, so the step needs a larger batch, or never becomes compute-bound when KV
traffic dominates.

#### Exercise 1.7

Qwen3-8B uses 147,456 bytes of KV per token. Short answers: $500+300=800$ tokens, 118 MB per
request, about 339 concurrent requests in 40 GB. With thinking: about $500+6{,}000=6{,}500$ tokens,
958 MB per request, about 41 concurrent requests. Under memory-bound decoding, throughput scales
roughly with batch, so throughput in tokens/s drops by $339/41\approx8.2\times$. Each request also
emits $6{,}000/300=20\times$ more tokens, so requests per second fall by about
$8.2\times20\approx165\times$.

#### Exercise 1.8

$0.97^{30}\approx0.40$. If wrong answers are spread over, say, ten distinct values, a simulation of
plurality voting (counting ties as failures) gives about 0.57 success at 5 samples, 0.81 at 11, 0.89
at 15 and 0.96 at 21, so 15--17 samples are needed for 0.9. Independence is optimistic because a
model's errors are correlated: the same misconception recurs across samples, concentrating wrong
votes on one value and making plurality voting far less effective.

#### Exercise 1.9

Example probes: (i) "All squares have five sides. ABCD is a square. How many sides does ABCD have?"
(closed-world answer 5; a good reasoner flags the false premise). (ii) A freshly generated puzzle
with random names and quantities, for instance a scheduling constraint problem produced by a script
with a computed answer. (iii) A question whose answer runs against frequency, for instance "Which is
heavier, a kilogram of feathers or 900 g of steel?". Base models often continue the text without
answering; instruction-tuned models answer fluently but fall for (iii) more often; reasoning models
do best on (ii). Score automatically with programmatic answers and the verifier of
[@ch:eval], and for (i) with a rubric for "flags the inconsistency" judged by an LLM
([@ch:eval]).

#### Exercise 1.10

A reasonable ranking is distillation, then RL, then inference-time scaling. Strict latency rules out
heavy per-query sampling. One week of compute is enough to fine-tune on traces from a stronger
teacher, which is cheap and stable, and the fine-tuned model serves at its normal speed. RL is
attractive if a reliable automatic reward exists (for instance, review comments that can be checked
against later bug fixes) but is costlier to make work; it can follow distillation as a second stage.
A light form of inference-time scaling (one short chain of thought) can still be used if the latency
budget allows.

#### Exercise 1.11

(a) Setting the context term of ([-@eq:landscape:kaplan]) equal to $2N$ gives
$n_{\text{ctx}}^\star = N/(L\,d_{\text{attn}})$. With $N$ the block weight matrices (as
`kaplan_forward_flops` counts them): GPT-2 small, $N = 84.9$M, $L\,d_{\text{attn}} = 12\times768$,
so $n_{\text{ctx}}^\star = 9{,}216 = 12d$. Qwen3-0.6B, $N=440$M, $L\,d_{\text{attn}}=28\times2048$,
so $n_{\text{ctx}}^\star=7{,}680$. Qwen3-8B, $N=6.95$B, $L\,d_{\text{attn}}=36\times4096$, so
$n_{\text{ctx}}^\star=47{,}104$. The lab's `kaplan_forward_flops(cfg, 0)` and
`kaplan_forward_flops(cfg, 1)` give $2N$ and the per-token slope directly.

(b) Per layer, $N/L = d\,(2d_{\text{attn}} + 2d_{\text{kv}}) + 3\,d\,d_{\text{ff}}$ for a SwiGLU model
with $d_{\text{kv}} = G d_h$ (GPT-2: $4d^2 + 8d^2$). So
$n_{\text{ctx}}^\star = d\,(2 + 2d_{\text{kv}}/d_{\text{attn}} + 3d_{\text{ff}}/d_{\text{attn}})$, which
is $12d$ only for the GPT shape. Qwen3-0.6B has $d_{\text{attn}}=2d$ (16 heads of 128 on a
1,024-wide stream) and $d_{\text{kv}} = d_{\text{attn}}/2$, giving $1024\times(2+1+4.5) = 7{,}680$,
1.6 times below Kaplan's $12d=12{,}288$. Note also that the rule marks the point where the context
term *equals* $2N$, so "relatively small" really means "well inside" it.

(c) At $n_{\text{ctx}}=32{,}768$, $C_{\text{forward}}/2N = 1 + 32{,}768/47{,}104 = 1.70$; the backward
pass of attention costs twice its forward, like every matmul, so training cost is
$6N\times1.70$ per token. Against $6ND$ with Kaplan's $N$ the budget is too small by a factor of
1.70; if the team used the full 8.19B parameters (embeddings and head included) for $N$, the
true cost $6\times(6.95\times1.70 + 0.62)$ GFLOP per token against $6\times8.19$ is still too
small, by a factor of about 1.5. Underestimating compute by half is the difference between
finishing and not.

(d) Kaplan et al.: stage, pretraining; resource, FLOPs (a compute budget); mechanism, allocation of
a fixed budget between $N$ and $D$; exactness, changes the model; hardware ratio, none (FLOPs
only, which is exactly its blind spot); evaluation, one dataset, one schedule length, up to 1.5B
parameters. Hoffmann et al.: the same first four answers, with schedules matched to each token
budget and models up to 16B plus the 70B Chinchilla run, which is why its allocation replaced
Kaplan's.

#### Exercise 1.12

Prediction (`report.py --hw A100-40GB-PCIe` or `decode_step_roofline` directly): at a context of
about 160 tokens (the 32-token prompt plus half of 256 new tokens used by the benchmark) a decode
step moves 1.21 GB, so the bound is 0.78 ms per token (1,285 tokens/s) on the A100-40GB PCIe and
0.36 ms (2,768 tokens/s) on the H100; at 1,024 tokens of context, 0.84 ms and 0.39 ms. The ratio,
2.15, is the bandwidth ratio $3.35/1.555$, because both are bandwidth-bound. The critical batch is
201 on the A100 and 296 on the H100, the ratio of the ridge points (1.47), not of the bandwidths.

Measurement (`labs/lab43_inference/bench.py --exp compile --device cuda --size 0.6B`, measured on
2x A100-40GB (PCIe), torch 2.14.1): eager decoding runs at 29.8 tokens/s (33.6 ms per token,
43 times the bound) and `torch.compile` at 162 tokens/s (6.2 ms, 8 times the bound); the
`reduce-overhead` mode reaches 153 tokens/s because Inductor skips CUDA graphs when the KV cache is
mutated in place. Eager decoding of a 0.6B model is regime (b) of [@fig:pytorch:async]: several
hundred small kernels per token, each paying Python and launch overhead, so the step is bound by
the host, not by either roof. That also answers the question about ratios: the predicted ratio of
critical batches is a property of the GPUs and would hold once batches are large, but the measured
batch-1 speed ratio between an A100 and an H100 would be close to 1 in eager mode (the same host
does the same work) and would approach 2.15 only as compilation, CUDA graphs and a lean decode
loop remove the overhead.
