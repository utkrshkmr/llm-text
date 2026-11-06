# Encoders, Encoder--Decoders and Cross-Attention {#ch:encoders}

Every model we have built so far reads from left to right. The causal mask of [@ch:attention]
hides the future from each position, the loss of [@ch:gpt] asks every position to predict the
token after it, and the KV cache of [@ch:modern] exists only because that mask makes earlier
keys and values reusable. Remove the mask and all three change at once: the model can see the whole
sequence, it can no longer be trained to predict the next token (the answer is in its input), and at
inference it does one forward pass per sequence with nothing to cache. That is an *encoder*, the
other half of the original transformer (Vaswani et al., 2017), the shape of BERT (Devlin et al.,
2019), of the text embedding models that feed every retrieval system, of the rerankers that sort
what retrieval returns, and of the vision towers that let a language model see.

This chapter builds encoders from the pieces we already have and treats the places where they differ
from decoders as the interesting ones. We start with the masks, because a single boolean matrix
decides whether a model is a decoder, an encoder or the prefix-LM hybrid between them. We then make
the training objectives exact---not "mask some tokens" but the precise 80/10/10 rule of BERT and the
sentinel construction of T5 (Raffel et al., 2020)---and count how many gradient signals each one
buys per token read. The middle of the chapter joins an encoder to a decoder through cross-attention
and shows that an encoder--decoder has two caches of different character: a self-attention cache
that grows and a cross-attention cache that is written once. The second half turns the encoder into
the tools it is actually used for today: sentence embeddings trained with a contrastive loss,
cross-encoder rerankers, and vision transformers. We finish where this book always finishes, at the
machine: encoder inference is compute-bound, uses no KV cache, and is served by batching dynamically
over sequences of unequal length, a very different serving problem from the one of
[@ch:inference].

The lab trains a small encoder with masked prediction on a few kilobytes of text and then converts
it into a sentence-embedding model with a contrastive loss; the embeddings it produces are the kind
that [@ch:retrieval] will index, and the vision encoder of [@sec:encoders:vit] is
the tower that [@ch:alignment] bolts onto a language model.

![Roadmap of this chapter: masks and objectives (1--3), the encoder and the encoder--decoder in code
(4--6), the encoder as an embedding, reranking and vision model (7--9), and how encoders are served
(10). Amber stages are the systems view.](figures/ch10_encoders-1.svg){#fig:encoders:roadmap}

## Three masks, three models {#sec:encoders:masks}

Attention computes, for every query position $i$, a weighted average over the keys $j$ it is
*allowed* to see. The set of allowed keys is the whole architecture decision.
[@fig:encoders:masks] shows the three patterns that matter.

![The three attention masks for $T=7$ (rows are queries $i$, columns keys $j$; blue cells are
allowed). The causal mask lets $i$ see $j\le i$; the encoder mask lets everyone see everyone; the
prefix-LM mask lets every position see the whole prefix of length $p$ and is causal in the suffix,
so the prefix is read like an encoder and the suffix is generated like a
decoder.](figures/ch10_encoders-2.svg){#fig:encoders:masks}

##### Causal.

Query $i$ sees keys $j\le i$. The representation at position $i$ is a function of tokens $x_{\le i}$
only, so it can be trained to predict $x_{i+1}$ and, at inference, the keys and values of positions
$<i$ never change when a new token is appended, which is what makes the KV cache possible.

##### Bidirectional.

Every query sees every key. The representation at position $i$ depends on the whole sequence,
including $x_i$ itself, so the next-token objective is meaningless: the model would copy $x_{i+1}$
from its input. Appending a token changes every representation, so nothing can be cached across
steps---but there are no steps: an encoder processes a sequence exactly once.

##### Prefix-LM.

Key $j$ is visible to query $i$ when $j<p$ or $j\le i$. The first $p$ tokens (the prefix) are read
bidirectionally; the remaining tokens are generated causally and can see the whole prefix. This is
the mask of the original encoder--decoder flattened into one stack of layers (Raffel et al., 2020
compared the two; Tay et al., 2023 mixed it with other denoisers in UL2), and it is also, exactly,
the mask that a multimodal model uses over an image prefix followed by text
([@ch:alignment]).

The code for all three is a handful of lines ([@lst:encoders:masks]). We follow the
convention of [@ch:attention]: a mask is a boolean tensor broadcastable to $(B,H,T_q,T_k)$
that is `True` where attention is *forbidden*, and padding is a second mask of real tokens, shape
$(B,T)$, that is folded in over the key axis.

``` {#lst:encoders:masks .python caption="Encoder and prefix-LM masks (cf. `llmsys/encoder.py`)."}
def bidirectional_mask(pad_mask):
    """Nothing is forbidden except padding keys. pad_mask: (B, T), True = real."""
    if pad_mask is None:
        return None
    return (~pad_mask)[:, None, None, :]          # (B, 1, 1, T): same for every query

def prefix_lm_mask(prefix_len, T, device=None):
    """Key j is allowed for query i iff j < prefix_len or j <= i. (B|1, 1, T, T)."""
    if isinstance(prefix_len, int):
        prefix_len = torch.tensor([prefix_len], device=device)
    i = torch.arange(T, device=prefix_len.device)[:, None]
    j = torch.arange(T, device=prefix_len.device)[None, :]
    allowed = (j[None] < prefix_len[:, None, None]) | (j <= i)[None]
    return (~allowed)[:, None]
```

##### Padding queries.

`bidirectional_mask` masks padding *keys* only. A padding query (a row of the attention matrix
belonging to a pad token) still attends to the real tokens and produces some output; we let it, and
make every consumer ignore those positions explicitly---the pooling in
[@sec:encoders:embed] averages over real tokens only and the loss carries `ignore_index` at
pad positions. The alternative, masking entire rows, produces a row of $-\infty$ scores whose
softmax is NaN; the `nan_to_num` in `_attend` exists for the case where a key mask does mask a whole
row, as happens for a fully padded sequence in a batch.

##### Special tokens.

BERT frames every input as `[CLS] a [SEP] b [SEP]`. The `[CLS]` token has no content of its own; it
is a position whose output the model learns to use as a summary because the pretraining head reads
it there, and it is the natural place to attach a classifier ([@ch:classify]). `[SEP]` marks
sentence boundaries, and a learned *segment* embedding (0 for $a$, 1 for $b$) tells attention which
side of the boundary a token is on. None of these needs special treatment in the mask: they are real
tokens, attended to and attending like any other. They must be excluded from the *objective*, which
is the subject of the next section.

::: keyidea
[A model is its mask]{.box-title}

Encoder, decoder and prefix-LM share every parameter shape. The difference is one boolean matrix
and, following from it, what the loss can ask of each position and what can be cached at inference.
Everything in this chapter is a consequence of letting position $i$ see position $j>i$.
:::

## Masked-token prediction {#sec:encoders:mlm}

If the model may see $x_i$, it cannot be asked to predict it, so BERT hides it first: select a
fraction $\rho$ (15%) of the positions, corrupt the input at those positions, and ask the model to
recover the original token there. The loss is a cross-entropy over the vocabulary at the selected
positions and nothing elsewhere: $$\begin{equation}
\mathcal{L}_\text{MLM} = -\frac{1}{|M|}\sum_{i\in M}\log p_\theta\!\left(x_i \mid \tilde{\boldsymbol{x}}\right),
\qquad \tilde{\boldsymbol{x}} = \text{corrupt}(\boldsymbol{x}, M),
\label{eq:encoders:mlm}
\end{equation}$$ where $M$ is the set of selected positions and $\tilde{\boldsymbol{x}}$ the
corrupted sequence.

##### The 80/10/10 rule.

The obvious corruption, replacing every selected token by `[MASK]`, creates a mismatch: at
pretraining time the model only ever predicts at positions holding `[MASK]`, a token that never
appears in downstream inputs, so it has no incentive to build good representations for *real* tokens
at positions it must later read. BERT therefore replaces a selected token by `[MASK]` only 80% of
the time, by a uniformly random token 10% of the time, and leaves it unchanged the remaining 10%.
The label is always the original token. The random replacements force the model to doubt every input
token (any of them could be wrong); the unchanged ones force it to represent real tokens well enough
to reproduce them. Special tokens are never selected.

[@lst:encoders:mlm] is the complete construction. Two details are easy to get wrong. The
labels use `IGNORE_INDEX` ($-100$, the value `F.cross_entropy` skips) rather than zero, so an
unselected position contributes nothing rather than a loss towards token 0. And the three random
draws ($u$ selects, $r$ picks the corruption, and the random replacement tokens) are made with a
single generator in a fixed order, which is what makes the lab's test able to compare your
construction with ours bit for bit.

``` {#lst:encoders:mlm .python caption="BERT's masked-token targets (from `llmsys/encoder.py`)."}
IGNORE_INDEX = -100

def mlm_corrupt(ids, mask_token_id, vocab_size, special_ids=(), mask_prob=0.15,
                generator=None):
    special = torch.zeros_like(ids, dtype=torch.bool)
    for s in special_ids:
        special |= ids == s
    u = torch.rand(ids.shape, generator=generator, device=ids.device)
    selected = (u < mask_prob) & ~special
    labels = torch.where(selected, ids, torch.full_like(ids, IGNORE_INDEX))
    r = torch.rand(ids.shape, generator=generator, device=ids.device)
    inputs = ids.clone()
    inputs[selected & (r < 0.8)] = mask_token_id              # 80%: [MASK]
    rand_slot = selected & (r >= 0.8) & (r < 0.9)             # 10%: random token
    random_tokens = torch.randint(0, vocab_size, ids.shape, generator=generator,
                                  device=ids.device)
    inputs[rand_slot] = random_tokens[rand_slot]
    return inputs, labels                                     # 10%: left unchanged
```

##### The head.

BERT's prediction head is a small MLP (Linear, GELU, LayerNorm) followed by a projection to the
vocabulary whose weight is *tied* to the input embedding table, plus a bias. Since only
$|M|\approx0.15\,T$ positions carry a label, we gather those rows *before* the vocabulary projection
([@lst:encoders:head]). For a small encoder the vocabulary GEMM is the single most
expensive operator---$2\cdot d\cdot V$ FLOPs per position against $24d^2$ for a block---so scoring
15% of the positions instead of all of them removes most of its cost. (A decoder cannot do this:
every position has a label.)

``` {#lst:encoders:head .python caption="The MLM head scores only the selected positions."}
class MLMHead(nn.Module):
    def __init__(self, emb_dim, tok_emb):
        super().__init__()
        self.transform = nn.Sequential(nn.Linear(emb_dim, emb_dim), GELU(), LayerNorm(emb_dim))
        self.decoder_weight = tok_emb.weight        # tied with the input embedding table
        self.bias = nn.Parameter(torch.zeros(tok_emb.num_embeddings))

    def forward(self, hidden, labels=None):
        if labels is not None:
            keep = labels != IGNORE_INDEX
            hidden, labels = hidden[keep], labels[keep]   # (sum |M|, d): 15% of the rows
        logits = F.linear(self.transform(hidden), self.decoder_weight, self.bias)
        if labels is None:
            return logits
        return logits, F.cross_entropy(logits.float(), labels)
```

::: pitfall
[Tied heads need a small initialisation]{.box-title}

With the head tied to the embedding table, the initial logits are
$\boldsymbol{h}^\top \mathbf{E}^\top$ with $\boldsymbol{h}$ of roughly unit norm per coordinate
after the LayerNorm. If the embeddings are drawn with unit variance (PyTorch's `nn.Embedding`
default), the logits have standard deviation $\sqrt d\approx 28$ for $d=768$ and the initial loss is
tens of nats instead of $\log V\approx 10$; the first hundreds of steps are spent undoing that. BERT
draws every weight from $\mathcal N(0,0.02^2)$, and so does our `EncoderModel`. The lab's test
checks that the loss at step zero is within one nat of $\log V$.
:::

##### Why bidirectionality helps understanding and hurts generation.

A decoder's representation of "bank" in "the bank of the river" is computed before "river" arrives;
an encoder's sees the whole phrase. For tasks whose output is a judgement about a complete input
(classification, extractive question answering, retrieval, token tagging) this is pure gain, and for
years encoders of a given size beat decoders of the same size on such benchmarks (Devlin et al.,
2019). For generation it is a loss: an encoder has no factorisation of $p(\boldsymbol{x})$ into a
product of conditionals, so it cannot sample a sequence left to right, and a masked model fills in
each hidden token as if the others were fixed, which produces incoherent text when many are hidden
at once (Wang and Cho, 2019 discuss sampling from BERT as a Markov random field). The 15% masking
rate also means that each training sequence yields a loss at only 15% of its positions, a cost we
quantify in [@sec:encoders:signal].

## Span corruption and the prefix-LM objective {#sec:encoders:span}

T5 keeps the idea of corrupting the input and recovering it, but moves the recovery to a decoder,
which produces a *sequence* rather than one token per masked position. This allows the model to hide
whole spans, and to learn to generate while still reading its input bidirectionally.

##### Span corruption with sentinels.

Choose a noise mask over the $T$ positions that marks about $\rho T$ of them as noise, grouped into
spans of mean length $\mu$ (T5 uses $\rho=0.15$, $\mu=3$). Each span of noise in the encoder input
is collapsed to a single *sentinel* token, numbered in order of appearance (`<X>`, `<Y>`, ..., held
in reserved vocabulary slots). The decoder target lists, for each span, its sentinel followed by the
original tokens of the span, and ends with one more unused sentinel so the decoder can say "no more
spans". [@fig:encoders:span] shows the worked example that the lab's first span test uses.

![T5 span corruption of an 8-token sequence with the noise mask $01100101$. Each run of noise in the
input becomes one sentinel; the target spells each span out behind its sentinel and closes with an
unused sentinel. The encoder reads 7 tokens, the decoder predicts 9, and the model is never shown
the 4 hidden tokens on the input side.](figures/ch10_encoders-3.svg){#fig:encoders:span}

The construction has three steps once the noise mask is fixed ([@lst:encoders:span]): find
span *starts* (noise positions whose predecessor is clean), number them cumulatively to obtain each
position's sentinel, and then keep the clean positions plus the starts (rewritten as sentinels) for
the encoder while spelling the noise positions out behind their sentinels for the decoder.

``` {#lst:encoders:span .python caption="Span corruption from a noise mask (from `llmsys/encoder.py`)."}
def span_corrupt(ids, sentinel_start, noise_density=0.15, mean_span_length=3.0,
                 eos_id=None, generator=None):
    noise = random_spans_noise_mask(len(ids), noise_density, mean_span_length, generator)
    prev = torch.cat([torch.zeros(1, dtype=torch.bool), noise[:-1]])
    starts = noise & ~prev                               # first position of each span
    span_id = torch.cumsum(starts.long(), 0) - 1         # 0, 0, 1, 1, 1, 2, ... on noise
    sentinels = sentinel_start + span_id
    enc = torch.where(starts, sentinels, ids)[~noise | starts]
    tgt = []
    for pos in torch.nonzero(noise).flatten().tolist():
        if starts[pos]:
            tgt.append(int(sentinels[pos]))
        tgt.append(int(ids[pos]))
    tgt.append(sentinel_start + int(starts.sum()))       # the closing, unused sentinel
    if eos_id is not None:
        tgt.append(eos_id)
    return enc, torch.tensor(tgt, dtype=ids.dtype)
```

The noise mask itself (`random_spans_noise_mask`, in the module) follows T5: compute the number of
noise tokens $n=\mathrm{round}(\rho T)$ and the number of spans $k=\mathrm{round}(n/\mu)$, split $n$
into $k$ positive lengths and the $T-n$ clean tokens into $k$ positive lengths, and interleave them
starting with a clean span. Starting with a clean span guarantees that the encoder input never
begins with a sentinel, and using the same number of clean and noise spans guarantees that spans
never merge. Because the lengths are exact, a 512-token sequence at $\rho=0.15$, $\mu=3$ always has
77 noise tokens in about 26 spans: the encoder reads about $435+26=461$ tokens and the decoder
predicts about $77+27=104$. We measured exactly these averages over 200 draws of the library
function.

##### Prefix-LM.

The prefix-LM objective needs no corruption at all. Pick a split point $p$, apply the prefix-LM mask
of [@fig:encoders:masks], and train with the ordinary next-token loss on the suffix only:
$\text{labels}_i = x_{i+1}$ for $p-1\le i<T-1$ and `IGNORE_INDEX` elsewhere
([@lst:encoders:prefix]). Position $p-1$ is the last prefix token; it is the first position
that predicts, because it is the first whose target ($x_p$) it has not seen. The lab's leakage test
perturbs each target token in turn and checks that the logits at the position predicting it do not
move, which is the only test of a mask that matters.

``` {#lst:encoders:prefix .python caption="Prefix-LM targets."}
def prefix_lm_targets(ids, prefix_len):
    T = ids.shape[-1]
    labels = torch.full_like(ids, IGNORE_INDEX)
    labels[..., prefix_len - 1: T - 1] = ids[..., prefix_len:]
    return ids, labels, prefix_lm_mask(prefix_len, T, device=ids.device)
```

::: deepdive
[UL2: a mixture of denoisers]{.box-title}

Tay et al. (2023) observed that the objectives above are points on a continuum (how much is hidden,
in how long spans, and whether the hidden part is at the end) and trained one model on a mixture:
"R-denoising" is T5's regular span corruption ($\rho=0.15$, short spans), "X-denoising" hides much
more (long spans or $\rho\ge0.5$), and "S-denoising" is the prefix-LM objective. A *mode token*
prepended to the input tells the model which denoiser produced the example, so the same weights
serve as an encoder-like infiller and as a causal generator. The same recipe trained PaLM 2 and the
encoder--decoder models of the Flan-UL2 family; in decoder-only training, mixing a fraction of
prefix-LM examples into the causal mixture is a cheap way to teach a model to use a long
bidirectionally read context, and it is the mask VLMs use over image tokens
([@ch:alignment]).
:::

## Gradient signal per token {#sec:encoders:signal}

Each objective pays the same price per token read---one forward and one backward through every
layer---but they differ sharply in what that pass buys.

::: napkinmath
[Loss terms per token processed]{.box-title}

Let a training sequence have $T=512$ tokens. The **causal LM** produces $T-1=511$ loss terms: a
fraction $\approx 1.0$ of positions. **MLM** at $\rho=0.15$ produces $77$: fraction $0.15$. **Span
corruption** ($\rho=0.15$, $\mu=3$) feeds the encoder $461$ tokens and the decoder $104$, of which
all $104$ carry a loss term (27 of them are sentinels, nearly free to predict); the fraction of
processed positions with a loss is $104/565=0.18$, and the fraction of *content* predictions is
$77/565=0.14$. A **prefix-LM** with a uniformly random split produces $T/2$ terms on average:
fraction $0.5$. In FLOPs per loss term the causal LM is cheapest by $6$--$7\times$ over MLM; this is
the usual argument that at equal compute, decoders see more supervision. The counter-argument is
that an MLM loss term is a harder, more informative prediction (both sides of context are available,
so the residual uncertainty is lower, and the gradient carries more information per term). ELECTRA
(Clark et al., 2020) closed most of the gap on the encoder side by replacing the 15% generation loss
with a binary "was this token replaced?" loss at *every* position, exactly the loss-fraction
argument turned into a method.
:::

::: systemslens
[Throughput needed to match a decoder]{.box-title}

An MLM model and a causal model of the same size and the same FLOPs per token need
$1/0.15\approx6.7$ and $1$ sequences respectively per 511 loss terms. If a decoder pretraining run
reaches 150000 loss terms per second per GPU (a typical BERT-base-size figure on an A100 at bf16),
an MLM run with the same per-token cost produces 22500. Encoder pretraining is therefore dominated
by the same question as decoder pretraining---tokens per second per GPU---with the extra knowledge
that only 15% of them count. The one saving on the encoder side is real: the vocabulary projection
runs on 15% of the positions. For BERT-base ($d=768$, $V=30522$) the head is
$2dV=46.9\,\mathrm{MFLOP}$ per scored position against $24d^2L=170\,\mathrm{MFLOP}$ for the 12
blocks; scoring every position would add 28% to the forward, scoring 15% adds 4%.
:::

## A bidirectional encoder in code {#sec:encoders:model}

The encoder block is the GPT-2 block of [@ch:gpt] with the mask removed and a padding mask
threaded through; [@lst:encoders:attn] gives the attention and
[@lst:encoders:model] the block and model. We keep GPT-2's parameter names (`W_query`,
`W_key`, `W_value`, `out_proj`) so weights move between the causal and bidirectional variants
without renaming; the state dicts of `EncoderModel` and `GPTModel` differ only in that the encoder
has no output head.

``` {#lst:encoders:attn .python caption="Bidirectional attention: the causal attention of [@ch:attention] without the triangle."}
class BidirectionalAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=True, use_sdpa=False):
        super().__init__()
        self.d_out, self.num_heads, self.head_dim = d_out, num_heads, d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout, self.use_sdpa = dropout, use_sdpa

    def forward(self, x, pad_mask=None, attn_mask=None):
        B, T, _ = x.shape
        q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        mask = bidirectional_mask(pad_mask)              # None, or padding keys only
        if attn_mask is not None:                        # e.g. a prefix-LM or packing mask
            mask = attn_mask if mask is None else (mask | attn_mask)
        ctx = _attend(q, k, v, mask, self.dropout, self.training, self.use_sdpa)
        return self.out_proj(ctx.transpose(1, 2).reshape(B, T, self.d_out))
```

``` {#lst:encoders:model .python caption="Encoder block and model (from `llmsys/encoder.py`)."}
class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.att = BidirectionalAttention(d, d, cfg["n_heads"], cfg["drop_rate"],
                                          cfg.get("qkv_bias", True))
        self.ff = FeedForward(d, cfg.get("hidden_dim"))
        self.norm1, self.norm2 = LayerNorm(d), LayerNorm(d)
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, pad_mask=None, attn_mask=None):
        x = x + self.drop(self.att(self.norm1(x), pad_mask, attn_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x

class EncoderModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg["n_layers"]))
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.apply(self._init_weights)                   # N(0, 0.02): see the pitfall above

    def forward(self, ids, pad_mask=None, attn_mask=None, position_ids=None):
        T = ids.shape[1]
        if position_ids is None:
            position_ids = torch.arange(T, device=ids.device)[None]
        x = self.drop_emb(self.tok_emb(ids) + self.pos_emb(position_ids))
        for blk in self.blocks:
            x = blk(x, pad_mask, attn_mask)
        return self.final_norm(x)                        # (B, T, d): heads attach here
```

Three things are worth noticing. The model returns hidden states, not logits: an encoder has many
heads (MLM, classification, pooling, pair scoring) and none of them belongs in the trunk. There is a
`position_ids` argument, which a decoder never needs; it exists for the *packed* batches of
[@sec:encoders:serving], where several sequences share one row and each restarts its
positions at zero. And there is a dropout rate in the configuration: BERT, unlike modern decoders,
was trained for many epochs over a 3.3-billion-word corpus and used dropout 0.1 everywhere.

Original BERT used post-LN blocks (normalise after the residual add); we use the pre-LN layout of
the rest of the book, which trains more stably at depth ([@ch:archdesign]) and is what
current encoders such as ModernBERT (Warner et al., 2024) use, together with RoPE, GeGLU MLPs and
unpadded attention. The design space of [@ch:archdesign] transfers to encoders almost
unchanged; the one component that does not transfer is the KV cache.

## Encoder--decoder models and cross-attention {#sec:encoders:crossattn}

An encoder on its own cannot generate. The original transformer solved this by pairing it with a
decoder whose every block contains a second attention, *cross-attention*, in which the queries come
from the decoder's own stream and the keys and values come from the encoder's output. The decoder
block has three sublayers ([@fig:encoders:encdec]): causal self-attention over the target
so far, cross-attention over the source, and the feed-forward network.

![An encoder--decoder. The encoder reads the source once and emits a memory $\mathbf{H}$. Each
decoder block attends causally to the target so far (its keys and values accumulate in a growing
cache, as in [@ch:modern]) and then attends to $\mathbf{H}$ through cross-attention, whose
keys and values depend only on $\mathbf{H}$ and are therefore computed once per layer and reused at
every step.](figures/ch10_encoders-4.svg){#fig:encoders:encdec}

##### Derivation.

Let $\mathbf{H}\in\mathbb R^{S\times d}$ be the encoder output for a source of $S$ tokens and
$\mathbf{Y}_t\in\mathbb R^{t\times d}$ the decoder stream after self-attention at step $t$.
Cross-attention in one head is $$\begin{equation}
\mathrm{CrossAttn}(\mathbf{Y}_t,\mathbf{H}) = \operatorname{softmax}\!\left(\frac{(\mathbf{Y}_t\mathbf{W}_Q)(\mathbf{H}\mathbf{W}_K)^\top}{\sqrt{d_{h}}}\right)\mathbf{H}\mathbf{W}_V ,
\label{eq:encoders:cross}
\end{equation}$$ which is [@ch:attention]'s attention with one substitution: $\mathbf{K}$ and
$\mathbf{V}$ are projections of $\mathbf{H}$ rather than of $\mathbf{Y}_t$. Two consequences follow
immediately. There is no causal mask---every target position may read the whole source, and that is
the point---only a padding mask over source keys. And $\mathbf{H}\mathbf{W}_K$ and
$\mathbf{H}\mathbf{W}_V$ do not depend on $t$: during generation the source is fixed, so these two
matrices, of shape $(S,d)$ per layer, can be computed once at the first decoding step and read at
every later one. The per-step cost of cross-attention is then one query projection ($2d^2$ FLOPs)
and one attention over $S$ keys ($4Sd$ FLOPs); the $4Sd^2$ FLOPs of projecting the memory are paid
once per sequence, not once per token.

##### The code.

[@lst:encoders:cross] implements the module with a `CrossKVCache` that is deliberately
*not* the `KVCache` of [@ch:modern]. The self-attention cache has an `update` that appends;
the cross cache has `set` and `get` and nothing else, so it cannot grow by accident, and its
`nbytes` is constant over a generation. The attention module reads from the cache when the layer's
entry exists and otherwise projects `memory` and, if a cache was handed in, fills it; `memory` may
therefore be `None` at every step after the first.

``` {#lst:encoders:cross .python caption="Cross-attention with a one-shot cache (from `llmsys/encdec.py`)."}
class CrossKVCache:
    """K, V of the encoder output, one (B, H, S, D) pair per decoder layer, written once."""
    def __init__(self, n_layers):
        self.k, self.v = [None] * n_layers, [None] * n_layers
    def has(self, layer_idx):   return self.k[layer_idx] is not None
    def set(self, layer_idx, k, v):  self.k[layer_idx], self.v[layer_idx] = k, v
    def get(self, layer_idx):   return self.k[layer_idx], self.v[layer_idx]
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)

class CrossAttention(nn.Module):
    def forward(self, x, memory=None, memory_pad_mask=None, cross_cache=None):
        B, T, _ = x.shape
        q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if cross_cache is not None and cross_cache.has(self.layer_idx):
            k, v = cross_cache.get(self.layer_idx)      # every step after the first
        else:
            k, v = self.project_kv(memory)              # (B, H, S, D) each, from the encoder
            if cross_cache is not None:
                cross_cache.set(self.layer_idx, k, v)
        mask = None if memory_pad_mask is None else (~memory_pad_mask)[:, None, None, :]
        ctx = _attend(q, k, v, mask, 0.0, self.training, self.use_sdpa)
        return self.out_proj(ctx.transpose(1, 2).reshape(B, T, self.d_out))

class DecoderBlock(nn.Module):
    def forward(self, x, memory=None, memory_pad_mask=None, self_cache=None, cross_cache=None):
        x = x + self.self_att(self.norm1(x), cache=self_cache)        # causal, growing cache
        x = x + self.cross_att(self.norm2(x), memory, memory_pad_mask, cross_cache)
        x = x + self.ff(self.norm3(x))
        return x
```

Greedy decoding ([@lst:encoders:greedy]) makes the two caches' roles explicit: the encoder
runs once, the memory is passed only at step zero, and both caches travel through every step. The
lab asks you to write this loop and proves it correct the only way that matters---the logits from
token-by-token decoding with both caches equal those of one full forward pass over the complete
target, and the key projection of each cross-attention layer runs exactly once.

``` {#lst:encoders:greedy .python caption="Encode once, decode with two caches."}
@torch.no_grad()
def greedy_decode(model, src, bos_id, eos_id, max_new, src_pad_mask=None):
    memory = model.encode(src, src_pad_mask)                     # the only encoder call
    self_cache = KVCache(model.n_dec_layers)                     # grows by one token per step
    cross_cache = CrossKVCache(model.n_dec_layers)               # filled at step 0, then read
    cur = torch.full((src.shape[0], 1), bos_id, dtype=torch.long, device=src.device)
    out = []
    for step in range(max_new):
        logits = model.decode(cur, memory if step == 0 else None, src_pad_mask,
                              self_cache, cross_cache)
        cur = logits[:, -1].argmax(-1, keepdim=True)
        out.append(cur)
        if bool((cur == eos_id).all()):
            break
    return torch.cat(out, dim=1)
```

##### Relative positions in T5.

T5 has no position embeddings at the input. Instead each attention layer adds a learned scalar bias
$b_{\beta(j-i)}$ to the logit between query $i$ and key $j$, where $\beta$ maps the signed distance
into 32 buckets: exact buckets for distances up to 8, logarithmically spaced buckets up to a maximum
distance of 128, and one bucket for everything farther. In the bidirectional encoder half the
buckets serve negative and half positive distances; the decoder, which looks back only, uses all of
them for the past. The function `relative_position_bucket` in `llmsys/encdec.py` reproduces this
mapping (the lab tests it against hand-computed values: distances $0,-1,-7,-8$ land in buckets
$0,1,7,8$, every distance beyond 128 in bucket 15, and $+1$ in bucket 17). Cross-attention has no
relative bias: source and target positions are not on a common axis. [@ch:archdesign] places
this scheme among the other position encodings; here we only note that because the bias is added
*inside* the attention it must be recomputed for the new query row at every decoding step, whereas
RoPE's rotation is applied once to each cached key.

## Two caches and the cost of a decoding step {#sec:encoders:caches}

Decoding is memory-bound ([@ch:inference]): the time of a step is set by the bytes it must
read, which are the weights it touches plus the caches of every sequence in the batch. For an
encoder--decoder a step reads $$\begin{equation}
\text{bytes/step} = b_w N_\text{dec} + B\cdot 2L_dd_{\text{model}}\, b_w\,(S + t),
\label{eq:encoders:stepbytes}
\end{equation}$$ where $N_\text{dec}$ counts the decoder's parameters only (the encoder is idle),
$b_w$ is the bytes per element, and the second term is the two caches: $2L_dd_{\text{model}}b_w$
bytes per cached token (one key and one value of width $d_{\text{model}}=Hd_{h}$ per layer), over
$S$ source tokens in the cross cache and $t$ target tokens in the self cache. A decoder-only model
that holds source and target in one context reads $b_w N + B\cdot2Ld_{\text{model}}b_w(S+t)$ with
its full parameter count and layer count. [@lst:encoders:stepbytes] states the two formulas
as code; the lab compares them.

``` {#lst:encoders:stepbytes .python caption="Bytes per decoding step (from `llmsys/encdec.py`)."}
def kv_bytes_per_token(n_layers, n_kv_heads, head_dim, bytes_per_el=2):
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_el

def encdec_step_bytes(n_dec_params, n_dec_layers, n_heads, head_dim, src_len, tgt_len,
                      batch=1, bytes_per_el=2):
    per_tok = kv_bytes_per_token(n_dec_layers, n_heads, head_dim, bytes_per_el)
    return n_dec_params * bytes_per_el + batch * per_tok * (tgt_len + src_len)

def decoder_only_step_bytes(n_params, n_layers, n_kv_heads, head_dim, src_len, tgt_len,
                            batch=1, bytes_per_el=2):
    per_tok = kv_bytes_per_token(n_layers, n_kv_heads, head_dim, bytes_per_el)
    return n_params * bytes_per_el + batch * per_tok * (src_len + tgt_len)
```

::: systemslens
[Encoder--decoder serving cost against a decoder-only model of equal size]{.box-title}

Serving an encoder--decoder is one prefill of the source through the *encoder* followed by decoding
with two attention reads per step. Compare a T5-base-shaped model ($d_{\text{model}}=768$, $H=12$,
$L_e=L_d=12$) with a decoder-only model of equal non-embedding parameter count. A decoder block has
$16d^2$ parameters (self-attention $4d^2$, cross-attention $4d^2$, MLP $8d^2$) and an encoder block
$12d^2$, so the encoder--decoder has $(12+16)\cdot12\,d^2=336d^2$ parameters and the equal-size
decoder-only model has $L=28$ blocks of $12d^2$. Take $S=512$, bf16, batch 1. Per decoding step, the
encoder--decoder reads its $16d^2\cdot12$ decoder parameters, 226 , plus
$2\cdot12\cdot768\cdot2=36.9\,\mathrm{\kilo\byte}$ per cached token over $512+t$ tokens: 245  at
$t=1$ and 264  at $t=512$. The decoder-only model reads all 396  of weights plus 86  per cached
token (28 layers): 440  and 484 . The encoder--decoder step moves *about half the bytes*, for two
reasons that compound: the encoder's parameters are not touched during decoding, and the cross cache
is as deep as the decoder, not as deep as the whole model. At batch 32 and $t=512$ the figures are
1.43  against 3.2  ([@fig:encoders:stepbytes]). The prefill is cheaper too: the source
passes through $12\cdot12d^2$ encoder parameters rather than $28\cdot12d^2$, i.e. $2.3\times$ fewer
FLOPs per source token. What the encoder--decoder gives up is flexibility: the source is read once,
through a fixed encoder, and cannot be re-read or extended as a conversation grows; every turn of a
dialogue would re-encode the whole history. That, more than any quality argument, is why chat
assistants are decoder-only while translation, summarisation and speech-to-text systems (Whisper is
an encoder--decoder) stayed with the two-stack design.
:::

![Bytes read per decoding step (analytic, [@eq:encoders:stepbytes]) for a T5-base-shaped
encoder--decoder and a 28-layer decoder-only model with the same 198M non-embedding parameters,
$S=512$ source tokens, bf16. At every batch size and target length the encoder--decoder reads about
half as many bytes: idle encoder weights and a shallower
cache.](figures/ch10_encoders-5.svg){#fig:encoders:stepbytes}

::: pitfall
[Padding the source pads the cross cache]{.box-title}

The cross cache holds $S$ entries per sequence, where $S$ is the *padded* source length of the
batch. A batch that mixes a 20-token and a 500-token source stores 500 keys and values for both in
every decoder layer, and every decoding step reads all of them (the padding mask only zeroes their
weights). Sort or bucket sources by length before batching them for an encoder--decoder exactly as
[@sec:encoders:serving] does for a plain encoder; the waste is paid at every step, not
once.
:::

## Encoders as embedding models {#sec:encoders:embed}

The most common use of an encoder today is not to predict masked tokens but to map a whole text to
one vector, so that semantically similar texts land near each other and a nearest-neighbour search
over millions of such vectors finds the relevant ones. [@ch:retrieval] builds the index; here
we build the model that fills it.

##### Pooling.

An encoder returns $T$ vectors; an embedding is one. Three reductions are in use
([@lst:encoders:pool]). *CLS pooling* reads the output at position 0, the `[CLS]` token,
which BERT's next-sentence objective trained to summarise the pair. *Mean pooling* averages the
outputs over the real (non-padding) tokens, which Sentence-BERT (Reimers and Gurevych, 2019) found
to work better than CLS once the model is fine-tuned for similarity. *Last-token pooling* reads the
output at the last real token; it is the only sensible choice for a *decoder-based* embedder,
because under a causal mask the last position is the only one that has seen the whole text. The
padding mask matters in two of the three: the mean must divide by the number of real tokens, and
"last" means the last real token, not the last column, which in a padded batch is pad for every
sequence but the longest.

``` {#lst:encoders:pool .python caption="Pooling token states to one vector (from `llmsys/encoder.py`)."}
def pool(hidden, pad_mask=None, how="mean"):
    """(B, T, d) -> (B, d). pad_mask: (B, T) bool, True for real tokens."""
    if pad_mask is None:
        pad_mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    if how == "cls":
        return hidden[:, 0]
    if how == "mean":
        m = pad_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)
    if how == "last":                                      # decoder-based embedders
        last = pad_mask.long().cumsum(1).argmax(1)         # index of the last True
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
```

##### Pretrained encoders are poor embedders.

A natural expectation is that an MLM-pretrained encoder, mean-pooled, already gives useful sentence
vectors. It does not. The lab measures this directly: on our synthetic paraphrase task,
nearest-neighbour accuracy over 200 held-out meanings is 0.094 for a randomly initialised encoder,
falls to 0.013 after masked-prediction pretraining, and rises to 1.000 after a few hundred
contrastive steps (measured, [@sec:encoders:lab]). The drop is not a bug. MLM shapes the
output space for a different reader---the tied vocabulary head---and the resulting token vectors are
*anisotropic*: they occupy a narrow cone, so every pair of sentences has cosine similarity near one
and the differences that matter are drowned out (Ethayarajh, 2019 documented this for BERT and
GPT-2). Reimers and Gurevych found BERT's raw CLS and mean embeddings to score below averaged GloVe
vectors on semantic similarity benchmarks. A contrastive objective is what turns an encoder into an
embedder.

##### The bi-encoder and InfoNCE.

A bi-encoder encodes queries and documents *independently* to unit vectors
$\boldsymbol{q}_i,\boldsymbol{d}_j\in\mathbb R^D$ and scores them by the dot product, so that
document vectors can be computed once, stored, and searched. Training uses pairs
$(\boldsymbol{q}_i,\boldsymbol{d}_i)$ that belong together (a question and the passage that answers
it; two paraphrases; a sentence and its translation) and treats all other documents in the batch as
negatives. The InfoNCE loss (van den Oord et al., 2018; in retrieval, DPR by Karpukhin et al., 2020)
is a cross-entropy in which the "classes" are the $B$ documents of the batch: $$\begin{equation}
\mathcal L_\text{InfoNCE} = -\frac1B\sum_{i=1}^{B}\log
\frac{\exp(\boldsymbol{q}_i^\top\boldsymbol{d}_i/\tau)}{\sum_{j=1}^{B}\exp(\boldsymbol{q}_i^\top\boldsymbol{d}_j/\tau)} .
\label{eq:encoders:infonce}
\end{equation}$$ The temperature $\tau$ scales cosine similarities, which live in $[-1,1]$, to a
range where softmax has useful gradients: with $\tau=0.05$ a similarity gap of $0.1$ becomes a logit
gap of 2. Smaller $\tau$ sharpens the loss towards the hardest negative in the batch; too small and
the gradients concentrate on a handful of examples. Values between $0.01$ and $0.1$ are universal.
The implementation ([@lst:encoders:infonce]) is a matrix product and a cross-entropy with
the identity as target; it also returns the in-batch retrieval accuracy, a free diagnostic.

``` {#lst:encoders:infonce .python caption="InfoNCE with in-batch and optional hard negatives."}
def info_nce_loss(q, d, temperature=0.05, hard_negatives=None, symmetric=False):
    """q, d: (B, D) unit vectors, d[i] is the positive of q[i]."""
    cands = d if hard_negatives is None else torch.cat([d, hard_negatives.reshape(-1, d.shape[-1])])
    logits = q @ cands.T / temperature                   # (B, B [+ N])
    target = torch.arange(q.shape[0], device=q.device)
    loss = F.cross_entropy(logits, target)
    if symmetric:                                        # documents must also pick their query
        loss = 0.5 * (loss + F.cross_entropy(logits[:, : q.shape[0]].T, target))
    acc = (logits.argmax(1) == target).float().mean()
    return loss, acc
```

::: systemslens
[Batch size is the negative sampler]{.box-title}

In InfoNCE the batch *is* the set of negatives, so a larger batch is not just a throughput knob:
with $B$ queries and $B$ documents the loss sees $B^2-B$ negative pairs for $2B$ encoder forwards,
and the hardest negative in a batch of 32768 is far harder than the hardest in a batch of 128.
Contrastive training therefore runs at batch sizes that no language-modelling run would use (CLIP,
Radford et al., 2021, used 32768; E5 and GTE use tens of thousands). Two systems consequences
follow. First, the $B\times B$ logit matrix is built across all data-parallel ranks: each rank
encodes its shard, an all-gather collects every rank's document vectors (cheap: $B\times D$ numbers,
a few megabytes), and every rank scores its queries against the global set; the gradient flows only
through the rank's own vectors unless the gather is made differentiable. Second, activation memory
for $B$ sequences through the encoder at that batch size exceeds any GPU, so GradCache (Gao et al.,
2021) computes the embeddings once without a graph, obtains the gradient of the loss with respect to
each embedding, and then re-runs the encoder in small chunks, back-propagating each chunk's
embedding gradient---a form of checkpointing in which the loss, not the layers, is the boundary.
:::

##### Hard negatives.

In-batch negatives are mostly easy: a random passage is obviously unrelated to a random question,
and after a few hundred steps most of the sum in [@eq:encoders:infonce] is already
negligible. The models that top retrieval benchmarks mine *hard* negatives---passages that an
earlier model ranked highly but that are not the answer---and append them to the candidate set;
`hard_negatives` in [@lst:encoders:infonce] does exactly that (the lab checks that adding
near-duplicates of the queries as negatives raises the loss). Hard-negative mining is an outer loop
over training: train, index the corpus, retrieve negatives for every training query, retrain (ANCE,
Xiong et al., 2021, does it asynchronously with a lagging copy of the encoder). It is also a source
of label noise: a mined "negative" is often a correct answer the annotators missed, which is why
many recipes distil from a cross-encoder's scores ([@sec:encoders:rerank]) instead of using
hard labels.

##### Matryoshka embeddings.

An index of $N$ vectors of dimension $D$ costs $4ND$ bytes in fp32, and search cost scales with $D$.
Matryoshka representation learning (Kusupati et al., 2022) trains the embedding so that its first
$m$ coordinates are themselves a good embedding for several nested $m$ (say $64,128,256,\ldots,D$),
by summing the InfoNCE loss over the truncated-and-renormalised prefixes (`matryoshka_loss` in the
module). A deployment then stores the full vector and searches with the first 128 coordinates,
reranking a short list with the full ones---the same coarse-to-fine idea as [@ch:retrieval]'s
index structures, placed inside the embedding itself.

##### Late interaction.

Between the bi-encoder (one vector per text, no interaction) and the cross-encoder (full token
interaction, no precomputation) sits ColBERT (Khattab and Zaharia, 2020): store one vector *per
token* of each document, and score a query against a document by *MaxSim*---for every query token,
take its best-matching document token and sum: $$\begin{equation}
s(q,d) = \sum_{i\in q}\max_{j\in d}\ \boldsymbol{q}_i^\top\boldsymbol{d}_j .
\label{eq:encoders:maxsim}
\end{equation}$$ Document vectors are still precomputed, so the encoder runs once per document, but
the index is $T_d$ times larger (ColBERTv2 compresses the token vectors to a few bits each to
compensate). The function `late_interaction_score` in the module computes
[@eq:encoders:maxsim] for a batch of queries against a batch of documents in one `einsum`.

##### Decoder-only LLMs as embedders.

Since 2023 the top of the embedding leaderboards has been occupied not by BERT-style encoders but by
7B-parameter decoders fine-tuned contrastively (E5-Mistral, Wang et al., 2024; GritLM, Muennighoff
et al., 2024; NV-Embed, Lee et al., 2024). The pretrained decoder brings far more knowledge than a
110M encoder; the price is the causal mask, which makes every position but the last see only its
prefix. Three remedies are in use: last-token pooling with an instruction prefix that tells the
model what to embed for; removing the causal mask during the contrastive fine-tune (the mask is a
runtime argument, so this is a one-line change and the model adapts quickly---LLM2Vec, BehnamGhader
et al., 2024); and a "latent attention" pooling layer that reads all positions. The systems cost is
the one we quantify in [@sec:encoders:serving]: embedding a passage with a 7B model costs
roughly $60\times$ the FLOPs of BERT-base.

## Cross-encoders and reranking {#sec:encoders:rerank}

A bi-encoder compresses a document to one vector *before* it knows the query. A *cross-encoder*
encodes the pair together---`[CLS] query [SEP] document [SEP]`---so that every query token attends
to every document token in every layer, and reads a relevance score from the `[CLS]` output
([@fig:encoders:biandcross]). It is more accurate for exactly that reason and unusable for
first-stage retrieval for exactly that reason: the score of each of $N$ documents needs its own
forward pass that cannot be precomputed. The standard architecture is therefore a pipeline: a
bi-encoder (or BM25) retrieves the top $k\approx100$ candidates from $N$ millions, and a
cross-encoder reranks the $k$ (Nogueira and Cho, 2019). [@lst:encoders:cross-enc] builds
the pair input and the head. The segment embedding (0 for the query side, 1 for the document) is the
one BERT-specific ingredient; the document is truncated before the query, because the query is short
and must survive intact.

![Three ways to score a query against a document. The bi-encoder precomputes one vector per document
and compares with a dot product; late interaction precomputes one vector per document token and
compares with MaxSim; the cross-encoder lets query and document attend to each other and pays one
forward pass per pair. Accuracy rises and precomputability falls from left to
right.](figures/ch10_encoders-6.svg){#fig:encoders:biandcross}

``` {#lst:encoders:cross-enc .python caption="Pair input and cross-encoder head (from `llmsys/encoder.py`)."}
def build_pair_input(q_ids, d_ids, cls_id, sep_id, pad_id, max_len):
    q_ids = list(q_ids)[: max_len - 3]
    d_ids = list(d_ids)[: max_len - 3 - len(q_ids)]      # truncate the document first
    ids = [cls_id, *q_ids, sep_id, *d_ids, sep_id]
    seg = [0] * (len(q_ids) + 2) + [1] * (len(d_ids) + 1)
    n_pad = max_len - len(ids)
    ids, seg = ids + [pad_id] * n_pad, seg + [0] * n_pad
    pad_mask = [True] * (max_len - n_pad) + [False] * n_pad
    return torch.tensor(ids), torch.tensor(pad_mask), torch.tensor(seg)

class CrossEncoderReranker(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.seg_emb = nn.Embedding(2, encoder.cfg["emb_dim"])   # query side / document side
        self.score = nn.Linear(encoder.cfg["emb_dim"], 1)

    def forward(self, ids, pad_mask, segment_ids):
        enc = self.encoder
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = enc.drop_emb(enc.tok_emb(ids) + enc.pos_emb(pos)[None] + self.seg_emb(segment_ids))
        for blk in enc.blocks:
            x = blk(x, pad_mask)
        return self.score(enc.final_norm(x)[:, 0]).squeeze(-1)  # read [CLS]: (B,)
```

::: napkinmath
[Reranking 100 candidates]{.box-title}

A BERT-base cross-encoder over a 32-token query and a 224-token passage processes $T=256$ tokens per
pair: $2\cdot85\cdot10^6\cdot256+4\cdot256^2\cdot768\cdot12=46\,\mathrm{GFLOP}$. Reranking $k=100$
candidates is 4.6 , about 15  at the A100's bf16 peak and 40 --60  at the utilisation we actually
measure in [@sec:encoders:serving]. A bi-encoder scores the same 100 candidates with a
$100\times768$ matrix--vector product---0.15 ---because the documents were encoded in advance. The
cross-encoder is $10^7$ times more expensive per candidate and still cheap enough to run on every
query, which is why the two-stage pipeline is the default: retrieval pays per *corpus* document
once, reranking pays per *candidate* per query.
:::

Rerankers are also trained from pairs, usually with a pointwise binary loss (relevant or not) or a
listwise softmax over the candidates of one query; because they are expensive to run they are also
the usual *teacher* for bi-encoders, whose student loss matches the teacher's score distribution
over the candidate list rather than the hard labels ([@ch:distill] covers the general
machinery). Recent rerankers are decoders too: a 7B model asked to output "yes" or "no" after
reading a query--document pair, with the logit of "yes" as the score, is a cross-encoder in all but
name.

## Vision transformers as encoders {#sec:encoders:vit}

Nothing in [@lst:encoders:model] knows that its input is text. The embedding table turns
integers into vectors; everything after it is a sequence model. A vision transformer (ViT,
Dosovitskiy et al., 2021) replaces the table with a function that turns an image into a sequence of
vectors and keeps the rest. It is the purest demonstration that an encoder is a set-of-vectors
processor with a position signal added, and it is the vision tower of nearly every vision--language
model ([@ch:alignment]).

##### Patchify.

An image of $C$ channels and $H\times W$ pixels is cut into non-overlapping $P\times P$ patches,
$N=(H/P)(W/P)$ of them, each flattened to a vector of $CP^2$ numbers and projected linearly to
$d_{\text{model}}$ ([@fig:encoders:patchify]). With $P=16$ a $224\times224$ image becomes
$N=196$ tokens of dimension $3\cdot16^2=768$; the "embedding table" is a
$768\times d_{\text{model}}$ matrix. In code ([@lst:encoders:patchify]) patchify is a
reshape and a permute, and the lab checks it against `F.unfold`, the generic sliding-window
operator, which produces the same layout. A `Conv2d` with kernel and stride $P$ computes the same
thing; we write the linear version because it makes the token view explicit.

![Patchify. The image is cut into $P\times P$ squares in row-major order, each is flattened
channel-major into a vector of $CP^2$ numbers and projected to $d_{\text{model}}$; a learned `[CLS]`
token and a learned position table (one row per patch) are added, and the result is an ordinary
encoder input. A $224\times224$ image with $P=16$ gives $N=196$ tokens; $P=14$ at $448\times448$
gives 1024.](figures/ch10_encoders-7.svg){#fig:encoders:patchify}

``` {#lst:encoders:patchify .python caption="Patchify and the ViT encoder (from `llmsys/vit.py`)."}
def patchify(img, P):
    """(B, C, H, W) -> (B, N, C*P*P): same layout as F.unfold(img, P, stride=P).transpose(1, 2)."""
    B, C, H, W = img.shape
    x = img.reshape(B, C, H // P, P, W // P, P)
    x = x.permute(0, 2, 4, 1, 3, 5)                       # (B, H/P, W/P, C, P, P)
    return x.reshape(B, (H // P) * (W // P), C * P * P)

class ViTEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, P = cfg["emb_dim"], cfg["patch"]
        self.grid = cfg["img_size"] // P
        self.patch_embed = PatchEmbed(cfg["in_ch"], P, d)               # one Linear(C*P*P, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + self.grid ** 2, d))
        self.blocks = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg["n_layers"]))
        self.final_norm = LayerNorm(d)

    def forward(self, img):
        x = self.patch_embed(img)                                       # (B, N, d)
        pos = self.pos_emb
        if x.shape[1] != self.grid ** 2:                                # another resolution
            P = self.cfg["patch"]
            pos = interpolate_pos_embed(pos, self.grid, (img.shape[-2] // P, img.shape[-1] // P))
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + pos
        for blk in self.blocks:
            x = blk(x)                                                  # no mask: all tokens real
        return self.final_norm(x)                                       # (B, 1+N, d)
```

##### Positions in two dimensions.

The patches arrive in row-major order, but the encoder has no notion of "the patch below this one"
unless the position signal gives it one. The original ViT simply learns a table with one row per
patch position, exactly like GPT-2's `pos_emb`, and lets the model discover the 2-D structure (it
does: the learned table's rows are similar for nearby grid positions). To run a model trained at
$224$ pixels on $384$-pixel images---more patches, a different grid---the table is reshaped to its
$g\times g$ grid, interpolated bicubically to the new grid and flattened back
(`interpolate_pos_embed`), which works because the table is smooth over the grid. Later encoders use
2-D RoPE (separate rotations for row and column index) or factorised tables, and native-resolution
models such as NaViT (Dehghani et al., 2023) pack patches from several images of different sizes
into one sequence with the block-diagonal attention mask of [@sec:encoders:serving], which
is where the `attn_mask` and `position_ids` arguments of our encoder earn their keep.

##### CLS or pooled.

For classification ViT reads the `[CLS]` output through a linear head; for CLIP-style contrastive
training against text (Radford et al., 2021) that output is projected to the shared embedding space
and trained with exactly [@eq:encoders:infonce], with image--caption pairs in place of
query--document pairs. A VLM instead takes *all* $N$ patch outputs, maps them through a small
projector to the language model's width, and splices them into the text sequence as if they were
tokens; the language model then attends to them under the prefix-LM mask of
[@fig:encoders:masks]. The cost of doing so is entirely in $N$.

::: napkinmath
[Resolution and patch size set the token count]{.box-title}

ViT-B/16 has the shape of BERT-base ($d=768$, $L=12$, $85$M non-embedding parameters). At
$224\times224$ it processes $1+196$ tokens:
$2\cdot85\cdot10^6\cdot197+4\cdot197^2\cdot768\cdot12=35\,\mathrm{GFLOP}$ per image, 1.5 ms at the
A100's bf16 roof---the figure usually quoted as 17.6 GMACs. At $384\times384$, $N=576$ and the cost
is 111  ($3.2\times$): the GEMMs scale with $N$ and attention with $N^2$, and attention is now 11%
of the total. Halving the patch size to 8 at $224$ gives $N=784$ and 156  ($4.4\times$). For a VLM
the token count is also what the language model has to carry: 1024 image tokens from a
$448\times448$, $P=14$ tower occupy 1024 slots of the 7B decoder's context and its KV cache (0.5 
per token for a Llama-7B-shaped model, half a gigabyte per image), which is why VLMs pool or merge
image tokens $2\times2$ or $4\times4$ before the language model sees them.
:::
