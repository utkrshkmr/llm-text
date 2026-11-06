"""Retrieval-augmented generation as a serving system (Chapter 48).

A RAG query is a short pipeline whose stages have very different cost profiles:

    embed(question) -> search(index) -> assemble(prompt) -> generate(engine)
       encoder, ms      memory-bound, ms     Python, us      prefill + decode, 100s of ms

The pieces here are deliberately plain: three chunkers, an embedding cache keyed by content hash
(so re-indexing an edited document re-embeds only the chunks that changed), a toy word tokenizer and a
hashing embedder for CPU tests, mean-pooled hidden states of any ``llmsys`` model as the real
embedder, and ``RAGPipeline``, which assembles every prompt as

    [shared system prefix, padded to a KV block boundary][retrieved chunks][question]

so that the serving engine's prefix cache (:mod:`llmsys.serving`) hits on the system prefix for every
query, and records a latency breakdown per stage. ``long_context_vs_retrieval_cost`` is the cost
model behind the chapter's "stuff the document in the prompt or retrieve from it?" plot.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .index import HNSWIndex

# ----------------------------------------------------------------------------- chunking
Tokenize = Callable[[str], List[str]]


@dataclass
class Chunk:
    """A contiguous span of a document: ``tokens[start:end]`` under the chunker's tokenisation."""
    doc_id: int
    chunk_id: int            # position within the document
    start: int               # token offsets, half-open
    end: int
    text: str

    @property
    def num_tokens(self) -> int:
        return self.end - self.start


def whitespace_tokenize(text: str) -> List[str]:
    return text.split()


def chunk_fixed(text: str, size: int, overlap: int = 0, doc_id: int = 0,
                tokenize: Tokenize = whitespace_tokenize) -> List[Chunk]:
    """Windows of ``size`` tokens with a stride of ``size - overlap``.

    Invariants (the lab tests them): every token is covered; every chunk but the last has exactly
    ``size`` tokens; consecutive chunks share exactly ``overlap`` tokens; the last chunk ends at the
    end of the document. Fixed windows are the cheapest chunker and the one whose cost is easiest to
    model: a document of T tokens becomes about T / (size - overlap) chunks.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("chunk_fixed: need size > 0 and 0 <= overlap < size")
    toks = tokenize(text)
    stride = size - overlap
    chunks: List[Chunk] = []
    start = 0
    while True:
        end = min(start + size, len(toks))
        chunks.append(Chunk(doc_id, len(chunks), start, end, " ".join(toks[start:end])))
        if end >= len(toks):
            break
        start += stride
    return chunks


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """Split on sentence-final punctuation followed by whitespace; good enough for prose."""
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def chunk_sentences(text: str, max_tokens: int, overlap_sentences: int = 0, doc_id: int = 0,
                    tokenize: Tokenize = whitespace_tokenize) -> List[Chunk]:
    """Pack whole sentences greedily up to ``max_tokens``; a chunk never splits a sentence.

    ``overlap_sentences`` repeats the last sentences of a chunk at the start of the next one, so a
    fact stated across a boundary is seen whole by at least one chunk. A single sentence longer than
    ``max_tokens`` is cut into fixed windows, which is the only case where a chunk may exceed the
    limit by zero tokens but the sentence rule is broken.
    """
    if max_tokens <= 0 or overlap_sentences < 0:
        raise ValueError("chunk_sentences: need max_tokens > 0 and overlap_sentences >= 0")
    sents = split_sentences(text)
    lengths = [len(tokenize(s)) for s in sents]
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    chunks: List[Chunk] = []
    i = 0
    while i < len(sents):
        if lengths[i] > max_tokens:                      # oversized sentence: fall back to windows
            for c in chunk_fixed(sents[i], max_tokens, 0, doc_id, tokenize):
                chunks.append(Chunk(doc_id, len(chunks), offsets[i] + c.start, offsets[i] + c.end, c.text))
            i += 1
            continue
        j, total = i, 0
        while j < len(sents) and total + lengths[j] <= max_tokens:
            total += lengths[j]
            j += 1
        chunks.append(Chunk(doc_id, len(chunks), offsets[i], offsets[j], " ".join(sents[i:j])))
        if j >= len(sents):
            break
        i = max(i + 1, j - overlap_sentences)          # always advance, even with a large overlap
    return chunks


# ----------------------------------------------------------------------------- tokenizer / embedders
class ToyTokenizer:
    """Word-level tokenizer for CPU tests: lower-cased words and punctuation, ids below ``vocab_size``.

    Reserved ids: 0 = pad, 1 = unk, 2 = end of text. Real pipelines use the model's BPE tokenizer
    (``llmsys.qwen3.Qwen3Tokenizer``); the interface is the same: ``encode`` / ``decode`` / ``eos_id``.
    """

    PAD, UNK, EOS = 0, 1, 2
    _WORD = re.compile(r"\w+|[^\w\s]")

    def __init__(self, vocab: Sequence[str], vocab_size: int):
        words = list(dict.fromkeys(vocab))[: vocab_size - 3]
        self.itos = ["<pad>", "<unk>", "<eos>"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.vocab_size = vocab_size
        self.eos_id, self.pad_id = self.EOS, self.PAD

    @classmethod
    def from_texts(cls, texts: Sequence[str], vocab_size: int) -> "ToyTokenizer":
        counts: Dict[str, int] = {}
        for t in texts:
            for w in cls._WORD.findall(t.lower()):
                counts[w] = counts.get(w, 0) + 1
        return cls(sorted(counts, key=lambda w: (-counts[w], w)), vocab_size)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(w, self.UNK) for w in self._WORD.findall(text.lower())]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in ids)


class ByteTokenizer:
    """UTF-8 bytes as tokens (ids 3..258), round-trips any text exactly; for tool-call formats in tests.

    Reserved ids match :class:`ToyTokenizer`: 0 = pad, 1 = unk, 2 = end of text. Vocabulary 259.
    """

    PAD, UNK, EOS = 0, 1, 2
    vocab_size = 259

    def __init__(self):
        self.eos_id, self.pad_id = self.EOS, self.PAD

    def encode(self, text: str) -> List[int]:
        return [b + 3 for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(i - 3 for i in ids if i >= 3).decode("utf-8", errors="replace")


class HashEmbedder:
    """Bag-of-words hashed into ``dim`` buckets, L2-normalised: a deterministic stand-in for an encoder.

    It has the one property the pipeline tests need, that texts sharing words land close together,
    at zero model cost. It is also what a lexical (BM25-style) first stage looks like as a vector.
    """

    def __init__(self, dim: int = 64, seed: int = 0):
        self.dim, self.seed = dim, seed

    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        out = torch.zeros(len(texts), self.dim)
        for i, t in enumerate(texts):
            for w in ToyTokenizer._WORD.findall(t.lower()):
                h = int(hashlib.blake2b(f"{self.seed}:{w}".encode(), digest_size=8).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0    # signed buckets cancel collisions
        return torch.nn.functional.normalize(out, dim=1, eps=1e-12)


class MeanPoolEmbedder:
    """Mean-pooled final hidden states of an ``llmsys`` decoder (GPT or Qwen3) as an embedding.

    A decoder's last-layer states are a weak but serviceable embedding; a model trained
    contrastively for the purpose (see the encoders chapter) is what production uses. We hook the
    output of ``final_norm`` rather than copying the forward pass, so any model with that module works.
    """

    def __init__(self, model, tokenizer, max_tokens: int = 64, device: Optional[str] = None):
        self.model, self.tokenizer, self.max_tokens = model.eval(), tokenizer, max_tokens
        self.device = device or next(model.parameters()).device
        self.dim = model.cfg["emb_dim"]

    @torch.no_grad()
    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        ids = [self.tokenizer.encode(t)[: self.max_tokens] or [self.tokenizer.pad_id] for t in texts]
        T = max(len(x) for x in ids)
        pad = self.tokenizer.pad_id
        batch = torch.tensor([x + [pad] * (T - len(x)) for x in ids], device=self.device)
        mask = torch.tensor([[1.0] * len(x) + [0.0] * (T - len(x)) for x in ids], device=self.device)
        captured: List[torch.Tensor] = []
        handle = self.model.final_norm.register_forward_hook(lambda m, i, o: captured.append(o))
        try:
            self.model(batch)
        finally:
            handle.remove()
        h = captured[0].float()                                   # (B, T, d)
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return torch.nn.functional.normalize(pooled, dim=1).cpu()


class EmbeddingCache:
    """Content-addressed cache in front of an embedder: ``embed`` computes only the misses.

    Re-indexing a corpus after small edits is the common case in production (documents change, the
    embedding model does not), and the cache turns it into an O(changed chunks) operation. Keyed by a
    hash of the text and the embedder's identity so a model swap invalidates everything.
    """

    def __init__(self, embedder, tag: str = ""):
        self.embedder = embedder
        self.tag = tag or type(embedder).__name__
        self.store: Dict[str, torch.Tensor] = {}
        self.stats = dict(hits=0, misses=0)

    def key(self, text: str) -> str:
        return hashlib.sha1(f"{self.tag}\0{text}".encode()).hexdigest()

    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        keys = [self.key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.store]
        self.stats["hits"] += len(texts) - len(missing)
        self.stats["misses"] += len(missing)
        if missing:
            fresh = self.embedder.embed([texts[i] for i in missing])
            for i, row in zip(missing, fresh):
                self.store[keys[i]] = row.clone()
        return torch.stack([self.store[k] for k in keys]) if keys else torch.zeros(0, self.embedder.dim)


# ----------------------------------------------------------------------------- prompt assembly
def pad_to_block(ids: List[int], block_size: int, pad_id: int) -> List[int]:
    """Right-pad a token list to a multiple of ``block_size``.

    The engine's prefix cache works in whole blocks (:mod:`llmsys.serving.block_manager`), so a shared
    prefix whose length is not a block multiple leaves its tail block uncacheable; padding the prefix
    once makes every query reuse all of it.
    """
    rem = (-len(ids)) % block_size
    return list(ids) + [pad_id] * rem


def assemble_prompt(system_ids: List[int], chunk_ids: Sequence[List[int]], question_ids: List[int],
                    separator_ids: Optional[List[int]] = None) -> List[int]:
    """Shared prefix first, then the retrieved chunks, then the question (variable part last).

    The order matters for caching, not for the model: anything placed before the retrieved chunks is
    identical across queries and therefore served from the KV cache; anything after them is recomputed.
    """
    sep = separator_ids or []
    out = list(system_ids)
    for c in chunk_ids:
        out += sep + list(c)
    return out + sep + list(question_ids)


# ----------------------------------------------------------------------------- the pipeline
@dataclass
class RAGAnswer:
    text: str
    citations: List[int]                       # global chunk ids (index ids) in rank order
    chunks: List[Chunk]
    distances: List[float]
    prompt_tokens: List[int]
    output_tokens: List[int]
    latency: Dict[str, float] = field(default_factory=dict)   # seconds per stage
    engine_metrics: Dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    """Chunk -> embed (cached) -> HNSW -> prompt with a shared prefix -> ``LLMEngine``.

    ``add_document`` is idempotent per chunk text: re-adding an edited document embeds only the new
    chunks and tombstones the old ones in the index (freshness without a rebuild).
    """

    def __init__(self, embedder, index: HNSWIndex, engine, tokenizer, system_prompt: str,
                 chunker: Callable[[str, int], List[Chunk]] = lambda text, doc: chunk_fixed(text, 32, 8, doc),
                 k: int = 3, ef: int = 64, cite_prefix: str = "[chunk {cid}]", align_prefix: bool = True):
        self.cache = embedder if isinstance(embedder, EmbeddingCache) else EmbeddingCache(embedder)
        self.index, self.engine, self.tok = index, engine, tokenizer
        self.chunker, self.k, self.ef, self.cite_prefix = chunker, k, ef, cite_prefix
        self.chunks: Dict[int, Chunk] = {}          # index id -> chunk
        self.doc_ids: Dict[int, List[int]] = {}     # doc id -> index ids currently live
        system_ids = tokenizer.encode(system_prompt)
        if align_prefix and engine is not None:
            system_ids = pad_to_block(system_ids, engine.block_size, tokenizer.pad_id)
        self.system_ids = system_ids

    # ------------------------------------------------------------------ indexing
    def add_document(self, doc_id: int, text: str) -> List[int]:
        """(Re)index a document; returns the index ids of its chunks."""
        if doc_id in self.doc_ids:                       # edited document: retire the old chunks
            self.index.delete(torch.tensor(self.doc_ids[doc_id]))
            for cid in self.doc_ids[doc_id]:
                self.chunks.pop(cid, None)
        chunks = self.chunker(text, doc_id)
        vecs = self.cache.embed([c.text for c in chunks])
        ids = self.index.add(vecs).tolist()
        for cid, c in zip(ids, chunks):
            self.chunks[cid] = c
        self.doc_ids[doc_id] = ids
        return ids

    # ------------------------------------------------------------------ querying
    def retrieve(self, question: str, k: Optional[int] = None) -> Tuple[List[int], List[float]]:
        q = self.cache.embedder.embed([question])
        ids, dists = self.index.search(q, k or self.k, self.ef)
        keep = ids[0] >= 0
        return ids[0][keep].tolist(), dists[0][keep].tolist()

    def build_prompt(self, question: str, cids: Sequence[int]) -> List[int]:
        chunk_ids = [self.tok.encode(self.cite_prefix.format(cid=cid) + " " + self.chunks[cid].text)
                     for cid in cids]
        return assemble_prompt(self.system_ids, chunk_ids, self.tok.encode(question))

    def answer(self, question: str, max_new_tokens: int = 32, k: Optional[int] = None) -> RAGAnswer:
        from ..serving import SamplingParams

        lat: Dict[str, float] = {}
        t0 = time.perf_counter()
        q = self.cache.embedder.embed([question])
        lat["embed"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        ids, dists = self.index.search(q, k or self.k, self.ef)
        keep = ids[0] >= 0
        cids, ds = ids[0][keep].tolist(), dists[0][keep].tolist()
        lat["search"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        prompt = self.build_prompt(question, cids)
        lat["assemble"] = time.perf_counter() - t2
        t3 = time.perf_counter()
        out = self.engine.generate([prompt], SamplingParams(max_new_tokens=max_new_tokens,
                                                           eos_id=self.tok.eos_id))[0]
        lat["generate"] = time.perf_counter() - t3
        lat["total"] = time.perf_counter() - t0
        return RAGAnswer(self.tok.decode(out), cids, [self.chunks[c] for c in cids], ds, prompt, out,
                         lat, self.engine.metrics())


# ----------------------------------------------------------------------------- cost model
@dataclass
class ServingShape:
    """What a query costs on the generator, in the units of the roofline (see the inference chapter)."""
    params: float = 8e9                 # generator parameters (Qwen3-8B)
    kv_bytes_per_token: float = 147456  # 36 layers x 8 KV heads x 128 x 2 (K,V) x 2 bytes
    attn_flops_per_pair: float = 589824  # 4 x 36 layers x (32 heads x 128): QK^T and PV for one (q, k) pair
    weight_bytes: float = 16e9
    flops: float = 989e12               # H100 bf16 dense
    bandwidth: float = 3.35e12
    mfu: float = 0.5                    # achieved fraction of peak during prefill
    embed_seconds: float = 0.002        # question through a small encoder
    search_seconds: float = 0.001       # one HNSW query on a warm index


def prefill_seconds(tokens: float, s: ServingShape) -> float:
    """Compute-bound: 2 FLOPs per parameter per token, plus the quadratic attention term."""
    dense = 2 * s.params * tokens
    attn = s.attn_flops_per_pair * tokens * tokens / 2        # causal: half of the (q, k) pairs
    return (dense + attn) / (s.flops * s.mfu)


def decode_seconds(context_tokens: float, new_tokens: float, s: ServingShape) -> float:
    """Memory-bound: every step streams the weights and the whole KV cache (batch size 1)."""
    per_step = (s.weight_bytes + context_tokens * s.kv_bytes_per_token) / s.bandwidth
    return new_tokens * per_step


def long_context_vs_retrieval_cost(doc_tokens: float, question_tokens: float = 64, answer_tokens: float = 256,
                                   k: int = 5, chunk_tokens: int = 256, prefix_cached: bool = False,
                                   shape: Optional[ServingShape] = None) -> Dict[str, float]:
    """Seconds per query for "whole document in the prompt" versus "retrieve k chunks".

    Long context pays prefill over D + q tokens (or only q when the document's KV is cached across
    queries) and decodes against a D-token cache; retrieval pays an embedding, a search, prefill over
    k * c + q tokens and a short cache. Memory per query is the KV footprint the engine must hold.
    """
    s = shape or ServingShape()
    ctx_lc = doc_tokens + question_tokens
    lc = prefill_seconds(question_tokens if prefix_cached else ctx_lc, s) + decode_seconds(ctx_lc, answer_tokens, s)
    ctx_r = k * chunk_tokens + question_tokens
    r = s.embed_seconds + s.search_seconds + prefill_seconds(ctx_r, s) + decode_seconds(ctx_r, answer_tokens, s)
    return dict(long_context_seconds=lc, retrieval_seconds=r, ratio=lc / r,
                long_context_kv_bytes=ctx_lc * s.kv_bytes_per_token, retrieval_kv_bytes=ctx_r * s.kv_bytes_per_token)


def crossover_doc_tokens(shape: Optional[ServingShape] = None, **kw) -> float:
    """Document length at which retrieval becomes cheaper than the whole document (bisection)."""
    lo, hi = 1.0, 1e7
    f = lambda d: long_context_vs_retrieval_cost(d, shape=shape, **kw)["ratio"] - 1.0   # noqa: E731
    if f(lo) > 0:
        return lo
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return hi
