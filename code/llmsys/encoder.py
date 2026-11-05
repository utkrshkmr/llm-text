"""Bidirectional encoders, their pretraining objectives, and embedding heads (Chapter 9).

The decoder-only models of the earlier chapters see only the past. An encoder sees the whole
sequence at once, which changes three things: the attention mask (none, or a prefix-LM mask),
the training objective (reconstruct corrupted tokens instead of predicting the next one), and
the serving pattern (one forward pass per sequence, no KV cache). This module provides

* masks:            ``bidirectional_mask``, ``prefix_lm_mask``
* objectives:       ``mlm_corrupt`` (BERT 80/10/10), ``span_corrupt`` (T5 sentinels),
                    ``prefix_lm_targets``, ``loss_fraction``
* the model:        ``BidirectionalAttention``, ``EncoderBlock``, ``EncoderModel``, ``MLMHead``
* embedding heads:  ``pool``, ``SentenceEmbedder``, ``info_nce_loss``, ``matryoshka_loss``,
                    ``late_interaction_score``, ``CrossEncoderReranker``, ``build_pair_input``
* serving:          ``length_sorted_batches``, ``padding_waste``, ``pack_sequences``, and the cost
                    model ``encoder_flops_per_sequence``, ``encoder_activation_bytes_per_sequence``,
                    ``batch_for_compute_bound``

Everything reuses the norms, MLP and attention core of ``llmsys.layers``.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import FeedForward, GELU, LayerNorm, _attend

IGNORE_INDEX = -100   # the label value F.cross_entropy skips

# Tiny configuration for tests and the lab (trains in seconds on a CPU).
ENCODER_CONFIG_TINY = dict(vocab_size=512, context_length=64, emb_dim=64, n_heads=4, n_layers=2,
                           drop_rate=0.0, qkv_bias=True)


# --------------------------------------------------------------------------- masks
def bidirectional_mask(pad_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Forbidden-mask for an encoder: nothing is masked except padding keys.

    pad_mask: (B, T) bool, True for real tokens. Returns (B, 1, 1, T) bool (True = forbidden)
    or None when there is no padding to hide. Padding *queries* are left alone: their outputs are
    garbage, and every consumer (pooling, loss) must ignore them explicitly.
    """
    if pad_mask is None:
        return None
    return (~pad_mask)[:, None, None, :]


def prefix_lm_mask(prefix_len: torch.Tensor | int, T: int, device=None) -> torch.Tensor:
    """Prefix-LM mask (True = forbidden), shape (B, 1, T, T) or (1, 1, T, T) for an int prefix.

    Positions < prefix_len attend to the whole prefix (bidirectional, like an encoder); positions
    >= prefix_len attend to the prefix and causally to earlier suffix tokens. Equivalently: key j
    is allowed for query i iff j < prefix_len or j <= i.
    """
    if isinstance(prefix_len, int):
        prefix_len = torch.tensor([prefix_len], device=device)
    i = torch.arange(T, device=prefix_len.device)[:, None]            # queries
    j = torch.arange(T, device=prefix_len.device)[None, :]            # keys
    allowed = (j[None] < prefix_len[:, None, None]) | (j <= i)[None]  # (B, T, T)
    return (~allowed)[:, None]


# --------------------------------------------------------------------------- objectives
def mlm_corrupt(ids: torch.Tensor, mask_token_id: int, vocab_size: int,
                special_ids: Sequence[int] = (), mask_prob: float = 0.15,
                generator: Optional[torch.Generator] = None):
    """BERT masked-token prediction targets.

    Select ``mask_prob`` of the non-special positions. Of those, 80% are replaced by [MASK],
    10% by a random token and 10% are left unchanged; the label at *every* selected position
    is the original token, all other labels are IGNORE_INDEX. Returns (inputs, labels).
    """
    device = ids.device
    special = torch.zeros_like(ids, dtype=torch.bool)
    for s in special_ids:
        special |= ids == s
    u = torch.rand(ids.shape, generator=generator, device=device)
    selected = (u < mask_prob) & ~special
    labels = torch.where(selected, ids, torch.full_like(ids, IGNORE_INDEX))
    r = torch.rand(ids.shape, generator=generator, device=device)
    inputs = ids.clone()
    inputs[selected & (r < 0.8)] = mask_token_id                       # 80%: [MASK]
    rand_slot = selected & (r >= 0.8) & (r < 0.9)                      # 10%: random token
    random_tokens = torch.randint(0, vocab_size, ids.shape, generator=generator, device=device)
    inputs[rand_slot] = random_tokens[rand_slot]
    return inputs, labels                                              # 10%: unchanged


def random_spans_noise_mask(length: int, noise_density: float, mean_span_length: float,
                            generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """T5's span mask: ``length`` positions, about ``noise_density`` of them noise, grouped in
    spans of mean length ``mean_span_length``. Noise and non-noise spans alternate, starting
    with a non-noise span, so the sequence never begins with a sentinel. Returns a bool (T,) mask."""
    if length < 2:
        raise ValueError("span corruption needs at least two tokens")
    n_noise = int(round(length * noise_density))
    n_noise = min(max(n_noise, 1), length - 1)
    n_nonnoise = length - n_noise
    # every noise span needs a clean span in front of it, so neither side can run out of tokens
    n_spans = max(1, min(int(round(n_noise / mean_span_length)), n_noise, n_nonnoise))

    def segment(n_items: int, n_segments: int) -> torch.Tensor:
        # positive integer lengths summing to n_items: pick n_segments-1 cut points
        cuts = torch.randperm(n_items - 1, generator=generator)[: n_segments - 1] + 1
        cuts = torch.sort(cuts).values
        edges = torch.cat([torch.zeros(1, dtype=torch.long), cuts,
                           torch.tensor([n_items])])
        return edges[1:] - edges[:-1]

    noise_lens = segment(n_noise, n_spans)
    nonnoise_lens = segment(n_nonnoise, n_spans)
    lens = torch.stack([nonnoise_lens, noise_lens], dim=1).reshape(-1)   # interleave
    is_noise = torch.arange(2 * n_spans) % 2 == 1
    return torch.repeat_interleave(is_noise, lens)


def span_corrupt(ids: torch.Tensor, sentinel_start: int, noise_density: float = 0.15,
                 mean_span_length: float = 3.0, eos_id: Optional[int] = None,
                 generator: Optional[torch.Generator] = None):
    """T5 span corruption for one 1-D token sequence.

    Each noise span in the input is replaced by one sentinel token (sentinel_start, +1, +2, ...).
    The target lists, for each span, its sentinel followed by the original tokens, and ends with
    the next unused sentinel (then ``eos_id`` if given). Returns (encoder_input, decoder_target).
    """
    noise = random_spans_noise_mask(len(ids), noise_density, mean_span_length, generator)
    noise = noise.to(ids.device)
    # a span *starts* where noise is True and the previous position was not noise
    prev = torch.cat([torch.zeros(1, dtype=torch.bool, device=ids.device), noise[:-1]])
    starts = noise & ~prev
    span_id = torch.cumsum(starts.long(), 0) - 1                    # which span each noise pos is in
    sentinels = sentinel_start + span_id
    enc = torch.where(starts, sentinels, ids)[~noise | starts]      # keep clean tokens + sentinels
    tgt_full = []                                                   # sentinel, then its span
    for pos in torch.nonzero(noise).flatten().tolist():
        if starts[pos]:
            tgt_full.append(int(sentinels[pos]))
        tgt_full.append(int(ids[pos]))
    tgt_full.append(sentinel_start + int(starts.sum()))
    if eos_id is not None:
        tgt_full.append(eos_id)
    return enc, torch.tensor(tgt_full, dtype=ids.dtype, device=ids.device)


def prefix_lm_targets(ids: torch.Tensor, prefix_len: int):
    """Prefix-LM: read ``prefix_len`` tokens bidirectionally, predict the rest left to right.

    Returns (inputs, labels, mask): inputs are the full sequence, labels[i] = ids[i+1] for the
    suffix positions and IGNORE_INDEX elsewhere (including the final position), mask is the
    (1, 1, T, T) prefix-LM mask.
    """
    T = ids.shape[-1]
    labels = torch.full_like(ids, IGNORE_INDEX)
    labels[..., prefix_len - 1: T - 1] = ids[..., prefix_len:]
    return ids, labels, prefix_lm_mask(prefix_len, T, device=ids.device)


def loss_fraction(labels: torch.Tensor) -> float:
    """Fraction of positions that contribute a loss term (gradient signal per token read)."""
    return float((labels != IGNORE_INDEX).float().mean())


# --------------------------------------------------------------------------- model
class BidirectionalAttention(nn.Module):
    """Multi-head attention with no causal mask. Same parameter names as the causal
    ``MultiHeadAttention`` of ``llmsys.layers`` so weights can move between the two."""

    def __init__(self, d_in: int, d_out: int, num_heads: int, dropout: float = 0.0,
                 qkv_bias: bool = True, use_sdpa: bool = False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out, self.num_heads, self.head_dim = d_out, num_heads, d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout, self.use_sdpa = dropout, use_sdpa

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """attn_mask: optional (B|1, 1, T, T) forbidden-mask (e.g. a prefix-LM mask)."""
        B, T, _ = x.shape
        q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        mask = bidirectional_mask(pad_mask)
        if attn_mask is not None:
            mask = attn_mask if mask is None else (mask | attn_mask)
        ctx = _attend(q, k, v, mask, self.dropout, self.training, self.use_sdpa)
        return self.out_proj(ctx.transpose(1, 2).reshape(B, T, self.d_out))


class EncoderBlock(nn.Module):
    """Pre-LN encoder block: x + Attn(LN(x)), x + FFN(LN(x)). Identical to the GPT-2 block of
    ``llmsys.gpt`` except that the attention is bidirectional."""

    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["emb_dim"]
        self.att = BidirectionalAttention(d, d, cfg["n_heads"], cfg["drop_rate"],
                                          cfg.get("qkv_bias", True), cfg.get("use_sdpa", False))
        self.ff = FeedForward(d, cfg.get("hidden_dim"))
        self.norm1, self.norm2 = LayerNorm(d), LayerNorm(d)
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, pad_mask=None, attn_mask=None):
        x = x + self.drop(self.att(self.norm1(x), pad_mask, attn_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class EncoderModel(nn.Module):
    """BERT-shaped encoder: token + learned position embeddings, L bidirectional blocks, final
    LayerNorm. Returns hidden states (B, T, d); heads are attached separately."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg["n_layers"]))
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        # BERT's N(0, 0.02) init: with a tied MLM head, unit-variance embeddings would give
        # initial logits of magnitude sqrt(d) and a loss far above log(V)
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """position_ids: optional (B, T) positions, needed only for packed batches, where each
        sequence restarts at zero inside one long row (see ``pack_sequences``)."""
        T = ids.shape[1]
        if position_ids is None:
            if T > self.cfg["context_length"]:
                raise ValueError("sequence exceeds context_length; truncate the input")
            position_ids = torch.arange(T, device=ids.device)[None]
        x = self.drop_emb(self.tok_emb(ids) + self.pos_emb(position_ids))
        for blk in self.blocks:
            x = blk(x, pad_mask, attn_mask)
        return self.final_norm(x)


class MLMHead(nn.Module):
    """BERT's prediction head: Linear -> GELU -> LayerNorm -> tied vocabulary projection."""

    def __init__(self, emb_dim: int, tok_emb: nn.Embedding):
        super().__init__()
        self.transform = nn.Sequential(nn.Linear(emb_dim, emb_dim), GELU(), LayerNorm(emb_dim))
        self.decoder_weight = tok_emb.weight                      # tied with the input table
        self.bias = nn.Parameter(torch.zeros(tok_emb.num_embeddings))

    def forward(self, hidden: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """With labels, score *only* the selected positions (15% of the tokens): the vocabulary
        projection is the most expensive op in a small encoder, so skipping it elsewhere matters."""
        if labels is not None:
            keep = labels != IGNORE_INDEX
            hidden, labels = hidden[keep], labels[keep]
            if labels.numel() == 0:  # a short batch can draw no masks; cross_entropy would give NaN
                return hidden.new_zeros(0, self.bias.numel()), hidden.sum() * 0.0
        logits = F.linear(self.transform(hidden), self.decoder_weight, self.bias)
        if labels is None:
            return logits
        return logits, F.cross_entropy(logits.float(), labels)


def mlm_loss(encoder: EncoderModel, head: MLMHead, inputs, labels, pad_mask=None):
    hidden = encoder(inputs, pad_mask)
    _, loss = head(hidden, labels)
    return loss


# --------------------------------------------------------------------------- pooling / embeddings
def pool(hidden: torch.Tensor, pad_mask: Optional[torch.Tensor] = None, how: str = "mean"):
    """Turn (B, T, d) token states into (B, d) sequence vectors.

    'cls'  -- position 0 (BERT); 'mean' -- masked mean over real tokens (Sentence-BERT);
    'last' -- the last *real* token (decoder-based embedders, where only it has seen everything).
    """
    if pad_mask is None:
        pad_mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    if how == "cls":
        return hidden[:, 0]
    if how == "mean":
        m = pad_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)
    if how == "last":
        last = pad_mask.long().cumsum(1).argmax(1)                  # index of the last True
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    raise ValueError(f"unknown pooling {how!r}")


class SentenceEmbedder(nn.Module):
    """Encoder + pooling + optional linear projection, L2-normalised (a bi-encoder tower)."""

    def __init__(self, encoder: nn.Module, pooling: str = "mean", proj_dim: Optional[int] = None):
        super().__init__()
        self.encoder, self.pooling = encoder, pooling
        d = encoder.cfg["emb_dim"]
        self.proj = nn.Linear(d, proj_dim, bias=False) if proj_dim else nn.Identity()

    def forward(self, ids: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        e = self.proj(pool(self.encoder(ids, pad_mask), pad_mask, self.pooling))
        return F.normalize(e, dim=-1)


def info_nce_loss(q: torch.Tensor, d: torch.Tensor, temperature: float = 0.05,
                  hard_negatives: Optional[torch.Tensor] = None, symmetric: bool = False):
    """InfoNCE with in-batch negatives. q, d: (B, D) unit vectors, row i of d is the positive
    for row i of q; hard_negatives: optional (B, D) or (N, D) extra documents appended to the
    candidate set. Returns (loss, in-batch retrieval accuracy)."""
    cands = d if hard_negatives is None else torch.cat([d, hard_negatives.reshape(-1, d.shape[-1])])
    logits = q @ cands.T / temperature                              # (B, B [+N])
    target = torch.arange(q.shape[0], device=q.device)
    loss = F.cross_entropy(logits, target)
    if symmetric:                                                   # documents pick their query too
        loss = 0.5 * (loss + F.cross_entropy(logits[:, : q.shape[0]].T, target))
    acc = (logits.argmax(1) == target).float().mean()
    return loss, acc


def matryoshka_loss(q: torch.Tensor, d: torch.Tensor, dims: Sequence[int],
                    temperature: float = 0.05):
    """Matryoshka representation learning: the InfoNCE loss summed over nested prefixes of the
    embedding, so the first ``m`` coordinates are a usable embedding for every m in ``dims``."""
    total = 0.0
    for m in dims:
        qm, dm = F.normalize(q[:, :m], dim=-1), F.normalize(d[:, :m], dim=-1)
        total = total + info_nce_loss(qm, dm, temperature)[0]
    return total / len(dims)


def late_interaction_score(q_tok: torch.Tensor, d_tok: torch.Tensor,
                           q_mask: torch.Tensor, d_mask: torch.Tensor) -> torch.Tensor:
    """ColBERT MaxSim: for every query token take its best-matching document token, then sum.
    q_tok: (Bq, Tq, D), d_tok: (Bd, Td, D) unit vectors. Returns (Bq, Bd) scores."""
    sim = torch.einsum("qid,pjd->qpij", q_tok, d_tok)               # (Bq, Bd, Tq, Td)
    sim = sim.masked_fill(~d_mask[None, :, None, :], float("-inf"))
    best = sim.max(-1).values                                       # (Bq, Bd, Tq)
    best = best.masked_fill(~q_mask[:, None, :], 0.0)
    return best.sum(-1)


# --------------------------------------------------------------------------- cross-encoder
def build_pair_input(q_ids: Sequence[int], d_ids: Sequence[int], cls_id: int, sep_id: int,
                     pad_id: int, max_len: int):
    """[CLS] q [SEP] d [SEP] with padding; returns (ids, pad_mask, segment_ids) as 1-D tensors.
    The document is truncated first (queries are short and must survive intact)."""
    q_ids = list(q_ids)[: max_len - 3]
    d_ids = list(d_ids)[: max_len - 3 - len(q_ids)]
    ids = [cls_id, *q_ids, sep_id, *d_ids, sep_id]
    seg = [0] * (len(q_ids) + 2) + [1] * (len(d_ids) + 1)
    n_pad = max_len - len(ids)
    ids, seg = ids + [pad_id] * n_pad, seg + [0] * n_pad
    pad_mask = [True] * (max_len - n_pad) + [False] * n_pad
    return (torch.tensor(ids), torch.tensor(pad_mask), torch.tensor(seg))


class CrossEncoderReranker(nn.Module):
    """Score a (query, document) pair jointly: encode the concatenation and read a scalar off
    the [CLS] state. Accurate (full token-level interaction) but costs one forward per pair."""

    def __init__(self, encoder: EncoderModel):
        super().__init__()
        self.encoder = encoder
        self.seg_emb = nn.Embedding(2, encoder.cfg["emb_dim"])
        nn.init.zeros_(self.seg_emb.weight)
        self.score = nn.Linear(encoder.cfg["emb_dim"], 1)

    def forward(self, ids, pad_mask, segment_ids):
        enc = self.encoder
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = enc.tok_emb(ids) + enc.pos_emb(pos)[None] + self.seg_emb(segment_ids)
        x = enc.drop_emb(x)
        for blk in enc.blocks:
            x = blk(x, pad_mask)
        return self.score(enc.final_norm(x)[:, 0]).squeeze(-1)      # (B,)


# --------------------------------------------------------------------------- dynamic batching
def length_sorted_batches(lengths: Sequence[int], max_tokens: int,
                          max_batch: Optional[int] = None) -> list[list[int]]:
    """Group sequence indices into batches of similar length, under a budget of padded tokens.

    Sort by length and fill each batch while ``len(batch) * longest <= max_tokens``. The budget
    is on padded tokens because that, not the sequence count, sets the time and memory of the
    forward. A single sequence longer than the budget still gets a batch of its own."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur = [], []
    for i in order:                                 # ascending, so lengths[i] is the longest so far
        full = max_batch is not None and len(cur) == max_batch
        if cur and (full or (len(cur) + 1) * lengths[i] > max_tokens):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def padding_waste(lengths: Sequence[int], batches: Sequence[Sequence[int]]) -> float:
    """Fraction of the padded matrix that is padding, summed over ``batches`` of indices."""
    real = sum(lengths[i] for b in batches for i in b)
    padded = sum(len(b) * max(lengths[i] for i in b) for b in batches)
    return 1.0 - real / padded


def pack_sequences(seqs: Sequence[torch.Tensor]):
    """Concatenate sequences into one row with no padding at all (the "unpadded" layout).

    Returns (ids, position_ids, attn_mask, spans): ids and position_ids have shape (1, sum T_i),
    attn_mask is the block-diagonal (1, 1, sum T_i, sum T_i) forbidden-mask that keeps every
    sequence to itself, and spans lists (start, end) per sequence. A dense mask is the clearest
    statement of the semantics, but it costs (sum T_i)^2 scores; a varlen attention kernel gets
    the same answer from the spans alone in sum T_i^2."""
    device = seqs[0].device
    ids = torch.cat(list(seqs))[None]
    position_ids = torch.cat([torch.arange(len(s), device=device) for s in seqs])[None]
    seq_of_token = torch.cat([torch.full((len(s),), n, device=device) for n, s in enumerate(seqs)])
    attn_mask = (seq_of_token[:, None] != seq_of_token[None, :])[None, None]
    ends = torch.tensor([len(s) for s in seqs]).cumsum(0).tolist()
    spans = list(zip([0] + ends[:-1], ends))
    return ids, position_ids, attn_mask, spans


# --------------------------------------------------------------------------- cost model
def encoder_flops_per_sequence(n_params_nonembed: float, T: int, n_layers: int, d_model: int):
    """Forward FLOPs for one sequence of T tokens: 2*params per token for the GEMMs plus the
    attention score/context products, 4*T*d per token per layer."""
    return 2.0 * n_params_nonembed * T + 4.0 * T * T * d_model * n_layers


def encoder_activation_bytes_per_sequence(T: int, n_layers: int, d_model: int, ffn_mult: float = 4,
                                          bytes_per_el: float = 2) -> float:
    """HBM bytes of activations read and written per sequence, counted one matrix at a time.

    In elements per token per layer: 6d for the Q, K, V projections (read x, write each), 2d for
    QK^T's inputs, 2d for PV (read V, write the context), 2d for the output projection, about 6d
    for the norms and residuals, and (2 + 2m)d for the two MLP GEMMs. The total is (20 + 2m)d."""
    return (20 + 2 * ffn_mult) * d_model * T * n_layers * bytes_per_el


def batch_for_compute_bound(weight_bytes: float, flops_per_seq: float, peak_flops: float,
                            hbm_bw: float, activation_bytes_per_seq: float = 0.0) -> int:
    """Smallest batch at which the forward's intensity reaches the machine balance.

    A batch of b sequences does b*F FLOPs and moves W + b*A bytes (the weights once, activations
    per sequence). It is compute bound when bF / peak >= (W + bA) / bw, which solves to
    b >= W / (F*bw/peak - A). If activations alone exceed that budget no batch size reaches the
    compute roof, and a ValueError says so."""
    slack = flops_per_seq * hbm_bw / peak_flops - activation_bytes_per_seq
    if slack <= 0:
        raise ValueError("activation traffic alone keeps this model memory bound")
    return max(1, math.ceil(weight_bytes / slack))
