## Chapter 9: Beyond Softmax Attention {#chapter-9-beyond-softmax-attention .unnumbered}

#### Exercise 9.1

With $\boldsymbol{v}$ augmented by a 1, every form accumulates
$\sum_{j\le i}(\text{decay})\,\phi(\boldsymbol{k}_j)\cdot1=\boldsymbol{z}_i$ in the last column of
the state, and the last output column is $\phi(\boldsymbol{q}_i)^\top\boldsymbol{z}_i$: exactly the
denominator of [@eq:attnalt:linear]. With an arbitrary feature map the features can have
mixed signs and the denominator can cross zero; $\mathrm{elu}+1>0$ keeps it positive, but it can
still be tiny when $\phi(\boldsymbol{q}_i)$ is small, which amplifies noise. This is why
implementations add an $\varepsilon$ or replace the division by an output normalisation.

#### Exercise 9.2

For chunk $c$ starting at $s$ with incoming state $\mathbf{S}$ (the state after token $s-1$),
unrolling [@eq:attnalt:gated] gives, for $i\ge s$,
$$\mathbf{S}_i=\Big(\prod_{l=s}^{i}\gamma_l\Big)\mathbf{S}+\sum_{j=s}^{i}\Big(\prod_{l=j+1}^{i}\gamma_l\Big)\boldsymbol{k}_j\boldsymbol{v}_j^\top ,$$
so
$\boldsymbol{y}_i^\top=\big(\prod_{l=s}^{i}\gamma_l\big)\boldsymbol{q}_i^\top\mathbf{S}+\sum_{j=s}^{i}D_{ij}(\boldsymbol{q}_i^\top\boldsymbol{k}_j)\boldsymbol{v}_j^\top$
with $D_{ij}=\prod_{l=j+1}^i\gamma_l$. The new state is the same expression at the chunk end $e$. In
logs, with $c_i=\sum_{l=s}^{i}\log\gamma_l$: the inter-chunk factor is $\exp(c_i)$ (`cum.exp()`),
the intra-chunk mask is $\exp(c_i-c_j)$ (`decay_mask`), the state's carried factor is $\exp(c_e)$
(`total.exp()`), and the per-token write weight is $\exp(c_e-c_j)$ (`w`), matching the listing.

#### Exercise 9.3

Per token and layer, causal softmax attention does about $2Td$ multiply-adds (scores and outputs,
averaged over the causal triangle); chunked linear attention does $Cd+2dd_{h}$. At $T=32{,}768$,
$d=4096$: softmax needs $2.7\times10^8$ multiply-adds and linear attention
$64\cdot4096+2\cdot4096\cdot128\approx1.3\times10^6$, about $205\times$ fewer. The curves cross near
$T\approx160$.

#### Exercise 9.4

Retrieval returns
$\mathbf{S}^\top\boldsymbol{k}_j=\boldsymbol{v}_j+\sum_{i\neq j}(\boldsymbol{k}_i^\top\boldsymbol{k}_j)\boldsymbol{v}_i$.
For random unit keys $\boldsymbol{k}_i^\top\boldsymbol{k}_j$ has variance about $1/d$, so with
unit-variance values the interference has variance $(n-1)/d$ per coordinate and the
signal-to-interference ratio is about $d/(n-1)$: retrieval degrades once $n$ approaches $d$. With
the delta rule and sequential writes, each write first removes what the memory currently returns for
its key, so the most recent writes are retrieved almost exactly, while older entries are perturbed
by later writes along non-orthogonal directions. The memory behaves like a recency-weighted store
with "overwrite" semantics instead of an ever-blurrier sum. For orthonormal keys ($n\le d$) it is
exact.

#### Exercise 9.5

For $T=2^{16}$: Hillis--Steele uses 16 rounds and
$\sum_{r=0}^{15}(T-2^r)=16T-(T-1)\approx9.8\times10^5$ combines; Blelloch uses 32 rounds (up-sweep
and down-sweep) and $2(T-1)\approx1.3\times10^5$ combines. Even for a single sequence, every round
of either algorithm exposes thousands of independent operations per channel, which is enough to fill
132 SMs once the many channels of a real layer are counted, so the work-efficient Blelloch scan (or
a blocked scan: sequential within a block, scan across blocks) wins. With 256 sequences the
parallelism across sequences and channels is ample, and simple chunked sequential scans are best.

#### Exercise 9.6

Implementations that form $P_i=\prod_{l\le i}\gamma_l$ explicitly and then compute $D_{ij}=P_i/P_j$,
or that rescale the state by $1/P$, fail: $P$ underflows to 0 and the ratio becomes $0/0$. The
reference code only ever computes $\exp$ of a difference of cumulative log-gates that is $\le0$
(`decay_mask` for $j\le i$, `cum.exp()`, `(total - cum).exp()`), so every factor lies in $(0,1]$ and
underflow merely produces a correct 0.

#### Exercise 9.7

Per full-attention layer the cache costs $2\cdot8\cdot128\cdot2=4096$ B per token, or 537 MB for
131,072 tokens; a window-4096 layer costs 16.8 MB. The budget is $40/32=1.25$ GB per sequence. With
2 full layers: $1.074+34\times0.0168=1.65$ GB, which is too much. With 1:
$0.537+35\times0.0168=1.12$ GB, which fits. So at this batch size and length only one of the 36
layers can be full attention; a recurrent alternative for the other layers instead of windows frees
about 0.55 GB more per sequence.

#### Exercise 9.8

$\boldsymbol{q}^\top(\mathbf{W}^{UK}\boldsymbol{c})=(\mathbf{W}^{UK\top}\boldsymbol{q})^\top\boldsymbol{c}$
by associativity. During decode with $T$ cached tokens: reconstructing keys costs
$T\cdot H\cdot128\cdot512$ multiply-adds plus $T\cdot H\cdot128$ for the dots; absorption costs
$H\cdot512\cdot128$ once plus $T\cdot H\cdot512$ for the dots, about $128\times$ fewer for large
$T$. In training every token is both a query and a key: keys are materialised once per token
($T\cdot H\cdot128\cdot512$), the same cost as absorbing queries, and attention over 128-dimensional
keys is cheaper than over 512-dimensional latents, so training uses the non-absorbed form.

#### Exercise 9.9

Two experts, top-1, capacity factor 1.0, two tokens per batch: capacity is one token per expert.
Alone, token A routes to expert 1 and is processed. Add token B that also prefers expert 1 and comes
first in the dispatch order (or has a higher router score, if priority routing is used): B takes the
slot and A is dropped, so its MoE output becomes zero and its final output changes. Testing must
therefore either use a dropless implementation, fix the batch composition, or compare outputs per
request only under batch-invariant settings. Bitwise "same input, same output" tests across
different batches will fail by design.

#### Exercise 9.10

Per sequence keep (i) a block table into the paged KV pool for each attention layer and (ii) an
index into a pool of fixed-size state slots for the recurrent layers. (a) Prefix caching: when a
prompt prefix is first computed, snapshot the recurrent states exactly at the prefix boundary into a
prefix cache keyed by the prefix hash, alongside the refcounted KV blocks; a hit copies the state
snapshot into a fresh slot. (b) Parallel sampling: share KV blocks with copy-on-write and copy the
state slot 8 times. The copy is cheap and fixed-size, but it is a real copy, because recurrent
states cannot be shared once the continuations diverge. (c) Preemption: swap both the KV blocks and
the state slot to host memory. With recompute-based preemption, the recurrent layers must re-run the
whole prefix, since there are no per-token states to keep.
