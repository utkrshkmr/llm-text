# Text Data, Tokenization, and the Input Pipeline {#ch:text}

A language model never sees text. It sees a stream of integers, and immediately turns each integer
into a vector by indexing into a table. Everything upstream of that table lookup---how raw bytes are
split into pieces, how the pieces are numbered, how the numbered stream is cut into training
examples, and how those examples reach the accelerator---is the subject of this chapter.

It is tempting to treat this as plumbing. It is not. The tokenizer fixes the vocabulary size $V$,
which sets the size of the largest matrices in a small model and the size of the largest
*activation* in a large one (the logits). It fixes how many tokens a document becomes, which sets
sequence length, which sets attention cost and KV-cache size at inference time. And the input
pipeline decides whether a multi-million-dollar training cluster spends its time doing matrix
multiplications or waiting on Python. We build every piece from scratch, then look at each one
through a systems lens.

![Roadmap of this chapter. Steps 1--2 build a word-level tokenizer to expose the ideas, step 3
replaces it with byte-level BPE as used by GPT-2 and Qwen3, step 4 turns a token stream into
next-token-prediction examples, step 5 converts IDs to vectors, and step 6 asks how all of this is
engineered when the corpus has $10^{13}$ tokens.](figures/ch08_text_data-1.svg){#fig:text:roadmap}

## From symbols to vectors: embeddings {#sec:text:embeddings}

Neural networks compute with real-valued tensors; text is a sequence of discrete symbols. The bridge
is an *embedding*: a map from a discrete object to a point in $\mathbb{R}^d$. The same idea is used
for images, audio and video---each modality gets its own embedding model---but for a decoder-only
LLM the object being embedded is a *token*, and the map is a learned lookup table
$\mathbf{E}\in\mathbb{R}^{V\times d}$ whose row $i$ is the vector for token $i$.

Before LLMs, embeddings were often trained as standalone artefacts. *Word2Vec* (Mikolov et al.,
2013) learned one vector per word by training a shallow network to predict a word from its
neighbours (or the neighbours from the word). The resulting space has a useful geometry: words that
occur in similar contexts land near each other, so after projecting to two dimensions one sees
clusters ([@fig:text:embspace]). Larger $d$ can encode more relations, at a cost
proportional to $d$ in every layer that touches the vector.

![A schematic 2-D projection of a word-embedding space (illustrative, not trained). Semantically
related words cluster, and some relations (comparative, superlative) appear as roughly consistent
offsets.](figures/ch08_text_data-2.svg){#fig:text:embspace}

LLMs do *not* reuse such pretrained word vectors. The embedding table is just another weight matrix,
initialised randomly and optimised jointly with the rest of the network for next-token prediction.
Its width is the model width $d_{\text{model}}$: 768 for GPT-2 small, 1024 for Qwen3-0.6B, and in
the thousands for larger models. Because the table is trained end to end, its geometry is whatever
the downstream layers find useful; the clustering of [@fig:text:embspace] tends to emerge
anyway.

::: systemslens
[Embeddings are a gather, not a matmul]{.box-title}

Mathematically $\boldsymbol{e}_i = \boldsymbol{1}_i^\top \mathbf{E}$, a one-hot vector times the
table. Operationally, nobody materialises one-hot vectors: the lookup is a *gather* of $T$ rows,
reading $T\cdot d$ elements from a $V\cdot d$ table---pure memory traffic, zero FLOPs. The backward
pass is a *scatter-add* into those rows, so the gradient of $\mathbf{E}$ is sparse (only rows of
tokens present in the batch are non-zero). Optimisers like AdamW still update every row's moments,
which is why some large-vocabulary systems use sparse or row-wise optimisers for embeddings, and why
the embedding table is a natural candidate for sharding by rows across devices
([@ch:distributed]).
:::

## Tokenizing text {#sec:text:tokenizing}

To build intuition we start with the simplest useful tokenizer: split on whitespace and punctuation.
Our sample text is a short story: Edith Wharton's public-domain story *The Verdict* (1908), about
twenty thousand characters long, which is small enough to train on a laptop yet long enough to be
interesting. Any plain-text file works.

Python's `re.split` with a capturing group returns both the pieces and the separators, which lets us
keep punctuation as tokens of its own:

``` {#lst:text:split .python caption="A regular-expression word splitter ([@lab:text], Part A)."}
import re

WORD_SPLIT = re.compile(r'([,.:;?_!"()\']|--|\s)')

def split_text(text):
    # capture group => separators are returned too; drop pure whitespace
    return [t for t in WORD_SPLIT.split(text) if t.strip()]

split_text('Fused kernels--fast, "lean" ones?')
# ['Fused', 'kernels', '--', 'fast', ',', '"', 'lean', '"', 'ones', '?']
```

Three design decisions are hiding in those two lines:

1.  **Case is preserved.** `Fused` and `fused` become different tokens. This doubles some vocabulary
    entries but lets the model learn capitalisation conventions (sentence starts, proper nouns),
    which it needs in order to *generate* well-formed text.

2.  **Whitespace is discarded.** Fine for a toy; fatal for code, where indentation is syntax.
    Byte-level BPE ([@sec:text:bpe]) keeps whitespace by folding it into tokens.

3.  **Punctuation is separated.** `fast,` becomes `fast` and `,`, so the vocabulary does not need a
    separate entry for every word--punctuation combination.

![Tokenization then ID mapping. The ID values are arbitrary positions in a sorted vocabulary built
from the training text.](figures/ch08_text_data-3.svg){#fig:text:tokids}

## Converting tokens into token IDs {#sec:text:ids}

A *vocabulary* is a bijection between token strings and integers. For a word-level tokenizer we
build it by collecting the unique tokens of the training corpus, sorting them (so the mapping is
deterministic), and enumerating. Encoding maps strings to IDs; decoding maps IDs back and
re-attaches punctuation to the preceding word.

``` {#lst:text:wordtok .python caption="Vocabulary construction and a reversible word-level tokenizer."}
def build_vocab(tokens, specials=("<|endoftext|>", "<|unk|>")):
    words = sorted(set(tokens))
    words += [s for s in specials if s not in words]
    return {w: i for i, w in enumerate(words)}

class WordTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        self.unk = vocab.get("<|unk|>")

    def encode(self, text):
        return [self.str_to_int.get(t, self.unk) for t in split_text(text)]

    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)  # re-attach punctuation
```

On a twenty-thousand-character story this yields a vocabulary of roughly a thousand entries.
Encoding and then decoding a sentence drawn from the training text reproduces it (modulo the
whitespace normalisation of the decoder). Encoding a sentence with a word that never appeared in
training is where things go wrong---which motivates special tokens.

::: pitfall
[Decoding is not always the inverse of encoding]{.box-title}

Any tokenizer that *discards* information during splitting (whitespace runs, tabs, Unicode
normalisation, lower-casing) cannot be inverted exactly. For training data this is a silent data
transformation; for a model that must emit code or JSON it is a correctness bug. A useful unit test
for any tokenizer is `decode(encode(s)) == s` over a fuzzed corpus of strings, including emoji,
combining characters, and invalid UTF-8 boundaries.
:::

## Adding special context tokens {#sec:text:special}

Two problems remain: words outside the vocabulary, and the boundaries between unrelated documents.
Both are solved by reserving vocabulary entries that never occur as ordinary text.

- `<|unk|>` stands in for any out-of-vocabulary word. It makes encoding total, but at a cost: every
  unknown word collapses to the same ID, so information is irreversibly lost.

- `<|endoftext|>` is inserted between independent documents when they are concatenated into one
  training stream. It tells the model "what follows is unrelated to what preceded" and, at inference
  time, serves as a stop signal.

Other conventions exist. Some models use `[BOS]` (beginning of sequence) and `[EOS]` (end of
sequence) separately; many use a `[PAD]` token to right-pad batches to equal length. GPT-2 uses
exactly one special token, `<|endoftext|>` (ID 50256), for document separation *and* as a padding
filler when one is needed; padding positions are then excluded from attention and from the loss by
masks, so what the filler token is does not matter ([@ch:classify; @ch:instruct]). Chat
models add more structure: Qwen3 wraps each turn in `<|im_start|>` ...`<|im_end|>` and marks
reasoning with `<think>` tags ([@ch:inference; @ch:grpo2]).

![Two special tokens in a concatenated training stream. The out-of-vocabulary word is replaced by
`<|unk|>`; the document boundary is made explicit with
`<|endoftext|>`.](figures/ch08_text_data-4.svg){#fig:text:special}

## Byte pair encoding {#sec:text:bpe}

Word-level vocabularies have two fatal flaws for open-domain text: they cannot represent unseen
words without `<|unk|>`, and to cover a large corpus they need hundreds of thousands of entries.
Character-level vocabularies have the opposite problem: tiny vocabulary, but sequences four to five
times longer, and attention cost grows quadratically with length. *Byte pair encoding* (BPE; Gage
1994, adapted to NLP by Sennrich et al., 2016) interpolates between the two, and the byte-level
variant introduced with GPT-2 (Radford et al., 2019) removes the unknown-token problem entirely.

### The algorithm

Byte-level BPE starts from the 256 possible byte values as the initial vocabulary. Any string, in
any language, encodes to UTF-8 bytes, so *every* input is representable---there is no `<|unk|>`.
Training then repeats:

1.  Count every adjacent pair of symbols in the corpus (weighted by frequency).

2.  Take the most frequent pair $(a,b)$ and add a new symbol $c$ whose byte string is the
    concatenation of those of $a$ and $b$.

3.  Replace every occurrence of $(a,b)$ in the corpus by $c$.

until the vocabulary reaches the target size $V$. The ordered list of merges *is* the tokenizer.
GPT-2 performed $50{,}000$ merges on top of 256 bytes, plus one special token:
$256+50{,}000+1=50{,}257$.

Two refinements make this work on real text. First, a *pre-tokenization* regex splits the text into
chunks (words with an optional leading space, digit runs, punctuation runs, whitespace) and merges
are never allowed to cross chunk boundaries; otherwise the most frequent "pair" would quickly become
something like `"e "` + `"the"`. Second, frequent words therefore become single tokens *including
their leading space* (`" the"`), which is how byte-level BPE preserves whitespace exactly.

![Encoding with a learned merge list. Encoding repeatedly applies the *earliest-learned* applicable
merge until none applies.](figures/ch08_text_data-5.svg){#fig:text:bpe}

### Implementation

Listing [-@lst:text:bpetrain] is a complete, if slow, byte-level BPE trainer. Words are
stored as tuples of IDs with their corpus frequency, so a word that appears $10^4$ times is
processed once per merge, not $10^4$ times.

``` {#lst:text:bpetrain .python caption="Naive byte-level BPE training ([@lab:text], Part B)."}
def get_pair_counts(words):          # words: {tuple_of_ids: frequency}
    counts = Counter()
    for ids, freq in words.items():
        for pair in zip(ids, ids[1:]):
            counts[pair] += freq
    return counts

def train_bpe(text, vocab_size):
    words = {tuple(w.encode("utf-8")): f for w, f in pretokenize(text).items()}
    merges = {}
    for new_id in range(256, vocab_size):
        counts = get_pair_counts(words)
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))  # det. ties
        merges[best] = new_id
        words = {tuple(merge_pair(w, best, new_id)): f for w, f in words.items()}
    return merges
```

Encoding a new string applies the learned merges *in training order*: inside each chunk, find the
adjacent pair with the smallest merge ID, merge it, and repeat ([@lst:text:bpeencode]).
Decoding is trivial: concatenate the byte strings of the IDs and decode UTF-8. Because merges can
split a multi-byte UTF-8 character across tokens, a *single* token may not be valid UTF-8 on its
own---streaming detokenizers must buffer bytes until a character completes ([@ch:inference]).

``` {#lst:text:bpeencode .python caption="BPE encoding by merge rank; decoding by byte concatenation."}
def bpe_encode(text, merges):
    out = []
    for chunk in PRETOKENIZE.findall(text):
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break                       # no applicable merge left
            ids = merge_pair(ids, pair, merges[pair])
        out.extend(ids)
    return out

def bpe_decode(ids, merges):
    vocab = build_byte_vocab(merges)        # id -> bytes
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
```

In practice we use a production tokenizer rather than our own: OpenAI's `tiktoken` (written in Rust)
ships GPT-2's exact merges, and Hugging Face `tokenizers` loads Qwen3's $151{,}936$-entry vocabulary
from a `tokenizer.json`. Our `llmsys.tokenizer.ByteBPETokenizer` implements the same algorithm as
[@lst:text:bpetrain; @lst:text:bpeencode] with save/load and special-token handling.

::: systemslens
[Making BPE training fast]{.box-title}

The naive trainer recounts every pair in every word after every merge: cost $O(M \cdot W)$ for $M$
merges over $W$ unique-word symbols. But a merge of $(a,b)$ only changes words that *contain*
$(a,b)$---usually a tiny fraction. An incremental trainer keeps (i) global pair counts and (ii) an
inverted index *pair $\to$ words containing it*; for each merge it visits only the affected words,
subtracts their old pairs, rewrites them, and adds the new pairs ([@lab:text], Part C). On a $485$ kB
synthetic corpus in this book's build environment, the incremental version learned the same 344
merges $6\times$ faster than the naive one (2.6 s $\to$ 0.4 s, single CPU core, shared machine).
Production trainers go further: a max-heap with lazy deletion makes "find the best pair" $O(\log P)$
instead of $O(P)$, pre-tokenization runs in parallel over shards, and everything is in Rust or C++.
:::

::: napkinmath
[What the vocabulary costs]{.box-title}

The vocabulary appears twice in the model: the input embedding $\mathbf{E}\in\mathbb{R}^{V\times d}$
and the output projection to logits (often tied to $\mathbf{E}$).

- GPT-2 small: $V\cdot d = 50{,}257\times 768 = 38.6$M parameters---31% of the 124M total. Each
  token's logits cost $2Vd = 77$ MFLOP, a quarter of the $\approx 2N$ forward FLOPs.

- Qwen3-0.6B: $151{,}936\times 1024 = 155.6$M parameters (tied), $\approx 26\%$ of the model.

- The logits *activation* during training is $B\cdot T\cdot V$ values. For $B=8$, $T=4096$,
  $V=151{,}936$ in fp32 that is $8\cdot 4096\cdot 151{,}936\cdot 4
    \approx 19.9$ GB---more than the weights, optimizer state and all other activations of a 0.6B
  model combined. This is why fused "linear + cross-entropy" kernels that compute the loss
  chunk-by-chunk without ever storing the full logits matter for large-vocabulary models.

Conversely, a larger vocabulary compresses text into fewer tokens (more bytes per token), shortening
sequences and so reducing attention FLOPs and KV-cache size at inference time. Vocabulary size is
therefore a genuine systems trade-off, not a modelling detail.
:::

![Parameters in one $V\times d$ embedding table (analytic). Untied models pay this twice. For small
models the vocabulary dominates the parameter budget; for large models it is a few percent, but the
per-token logits computation and the training-time logits tensor still scale with
$V$.](figures/ch08_text_data-6.svg){#fig:text:vocabcost}

## Tokenizers in the wild {#sec:text:wild}

Byte-level BPE is conceptually simple, but production tokenizers are full of details that change
model behaviour, cost, and even security. This section collects the ones a systems engineer is most
likely to trip over.

### UTF-8 and streaming detokenization

UTF-8 encodes a code point in one to four bytes: ASCII in one, most European and Middle-Eastern
scripts in two, most CJK characters in three, emoji in four. Because BPE operates on bytes, a token
boundary can fall *inside* a character ([@fig:text:utf8]). This is invisible when a whole
sequence is decoded at once, but an inference server streaming tokens to a client one at a time must
not emit half a character. The fix is a small incremental decoder that holds back bytes until they
form complete code points ([@lst:text:stream]); Python's `codecs` module provides exactly
this. Hugging Face's and vLLM's detokenizers implement the same idea with an offset-based
"prefix/read" window.

![Byte-level tokens need not align with character boundaries. Hex bytes shown; the token split is
illustrative.](figures/ch08_text_data-7.svg){#fig:text:utf8}

``` {#lst:text:stream .python caption="Streaming detokenization that never emits partial characters."}
import codecs

class StreamingDetokenizer:
    def __init__(self, vocab):                 # vocab: id -> bytes
        self.vocab = vocab
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token_id):                  # call once per generated token
        return self.dec.decode(self.vocab[token_id])   # "" until a char completes

    def flush(self):
        return self.dec.decode(b"", final=True)
```

### Numbers, whitespace and other artefacts

Pre-tokenization rules matter for what a model finds easy. GPT-2's regex lets BPE merge arbitrary
digit strings, so `"1234"` and `"12345"` may be split inconsistently, which makes digit-level
arithmetic harder to learn. Many newer tokenizers split numbers into single digits or into groups of
at most three digits so that place value is represented consistently. Similarly, `" the"` and
`"the"` are different tokens; a prompt that ends with a trailing space forces the model to continue
from an unusual token boundary and can visibly degrade completions---one reason serving APIs
sometimes "heal" the last prompt token. Finally, tokens that were frequent in the tokenizer's
training data but rare in the model's training data (the infamous "glitch tokens") have nearly
untrained embeddings and can trigger erratic behaviour.

### Special tokens are a security boundary

Chat models delimit roles with special tokens ([@ch:inference]). If a tokenizer parsed the
*string* `<|im_start|>` followed by `system` inside user-supplied text into the real special-token
ID, a user could forge a system message. Production tokenizers therefore distinguish "special tokens
allowed in this text" from ordinary text: `tiktoken`'s `allowed_special` and `disallowed_special`
arguments exist for exactly this reason, and our `ByteBPETokenizer.encode` takes an
`allowed_special` tuple. Serving stacks should tokenize templates and user content separately and
concatenate IDs, never concatenate strings.

::: napkinmath
[The cost of a language]{.box-title}

API pricing and context limits are in *tokens*, but users write in *characters*. English prose with
a modern BPE tokenizer averages roughly 4 bytes (about 4 characters) per token. A script that needs
3 bytes per character and is underrepresented in the merge list may average closer to 1--1.5 bytes
per token, i.e. more than one token per character. The same sentence then consumes several times
more tokens: several times the price, several times the KV cache, and a context window that holds
several times less text. Measuring bytes/token per language on a held-out multilingual set ([@lab:text],
`compression_ratio`) is the right first step before choosing or extending a tokenizer.
:::

## Data sampling with a sliding window {#sec:text:sliding}

An LLM is trained to predict the next token. Given a tokenized corpus $t_1,t_2,\dots,t_N$, a
training example is a window of $T$ consecutive tokens as input, $x=(t_{s+1},\dots,t_{s+T})$, and
the same window shifted one position to the right as target, $y=(t_{s+2},\dots,t_{s+T+1})$. One
window thus yields $T$ prediction tasks at once: from $t_{s+1}$ predict $t_{s+2}$; from
$t_{s+1..s+2}$ predict $t_{s+3}$; ...; from all $T$ inputs predict $t_{s+T+1}$. Causal masking
([@ch:attention]) ensures position $i$ cannot peek at targets to its right, so all $T$ losses
are computed in one forward pass.

![Sliding-window sampling with context length $T=4$ and stride
3.](figures/ch08_text_data-8.svg){#fig:text:sliding}

The *stride* controls how far consecutive windows start apart. Stride $=T$ partitions the stream
into disjoint windows---each token is an input exactly once per epoch. Stride $<T$ creates overlap:
more examples, but they are highly correlated and the model sees the same text many times per epoch,
which encourages memorisation of a small corpus. Stride $=1$ creates the maximum $N-T$ examples at a
$T$-fold redundancy. For pretraining at scale, disjoint windows (stride $=T$) are standard.

PyTorch structures this as a `Dataset` (how to fetch example $i$) plus a `DataLoader` (how to batch,
shuffle and parallelise fetching). Our dataset precomputes window views over a single token tensor
([@lst:text:dataset]); the loader stacks `batch_size` windows into $(B,T)$ input and target
tensors. `drop_last=True` discards a final short batch, which would otherwise produce a noisy,
small-batch gradient step and a different tensor shape (bad for compiled graphs).

``` {#lst:text:dataset .python caption="A sliding-window dataset and loader (core: `llmsys/data.py`)."}
class SlidingWindowDataset(Dataset):
    def __init__(self, token_ids, max_length, stride):
        ids = torch.as_tensor(token_ids, dtype=torch.long)
        n = 0
        if len(ids) > max_length:
            n = (len(ids) - max_length - 1) // stride + 1
        self.inputs = [ids[i*stride : i*stride + max_length] for i in range(n)]
        self.targets = [ids[i*stride + 1 : i*stride + max_length + 1]
                        for i in range(n)]
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, i):
        return self.inputs[i], self.targets[i]

loader = DataLoader(SlidingWindowDataset(ids, 256, 256),
                    batch_size=8, shuffle=True, drop_last=True, num_workers=0)
x, y = next(iter(loader))        # both (8, 256), torch.long
```

It is worth printing what the loader actually yields for tiny settings, because off-by-one errors in
the target shift are the most common bug in hand-written pretraining code and they do not
crash---the model simply learns to copy its input.

``` {#lst:text:strides caption="Loader output for a toy stream `0..11` with $T=4$: stride 1 vs. stride 4."}
stride=1, batch_size=2                 stride=4, batch_size=2
x = [[0, 1, 2, 3],                     x = [[0, 1, 2, 3],
     [1, 2, 3, 4]]                          [4, 5, 6, 7]]
y = [[1, 2, 3, 4],                     y = [[1, 2, 3, 4],
     [2, 3, 4, 5]]                          [5, 6, 7, 8]]
-> 8 windows per epoch                 -> 2 windows per epoch (tokens 8..11
   (each token reused up to 4x)           only appear as the last targets)
```

A simple invariant catches the shift bug: for every batch, `x[:, 1:] == y[:, :-1]` must hold
elementwise. Lab 3's tests check it, and it is cheap enough to leave as an assertion in a real
training loop's first step.

::: pitfall
[Slices are views]{.box-title}

`ids[a:b]` is a *view* sharing storage with `ids`, so the list of windows costs almost nothing
beyond the original tensor. If a worker process pickles such a view to send it to the main process,
however, PyTorch serialises the *whole* underlying storage. With `num_workers>0` and large corpora,
either use memory-mapped files ([@sec:text:pipeline]) or `.clone()` each window before
returning it.
:::

## Creating token embeddings {#sec:text:tokemb}

The last step converts a batch of IDs into vectors with `nn.Embedding(V, d)`---a trainable
$V\times d$ matrix whose forward pass is a row gather. A $(B,T)$ `long` tensor becomes a $(B,T,d)$
float tensor. [@fig:text:gather] shows the equivalence with a one-hot matrix product, which
is useful for reasoning about gradients: since $\boldsymbol{e} = \boldsymbol{1}_i^\top\mathbf{E}$,
$\partial \mathcal{L}/\partial\mathbf{E} = \boldsymbol{1}_i\,(\partial\mathcal{L}/\partial\boldsymbol{e})$,
i.e., the gradient flows only into row $i$.

![An embedding layer is a gather of rows; its gradient is a
scatter-add.](figures/ch08_text_data-9.svg){#fig:text:gather}

``` {#lst:text:onehot .python caption="Embedding lookup equals a one-hot product."}
torch.manual_seed(0)
emb = torch.nn.Embedding(num_embeddings=6, embedding_dim=3)
ids = torch.tensor([5, 2, 5])
a = emb(ids)                                        # gather: (3, 3)
b = F.one_hot(ids, 6).float() @ emb.weight          # same numbers, (3,6)@(6,3)
assert torch.allclose(a, b)
```

For a GPT-2-sized model the table is $50{,}257\times768$; a batch of $8\times1024$ IDs produces an
$(8,1024,768)$ tensor of 6.3M floats (25 MB in fp32). In a real model the table is initialised
randomly and learned; our tests initialise it with a fixed seed so results are reproducible.

## Encoding word positions {#sec:text:positions}

Self-attention, which we build in [@ch:attention], is *permutation-equivariant*: shuffling
the input vectors shuffles the outputs the same way. Without extra information the model cannot
distinguish "dog bites man" from "man bites dog". Token embeddings alone do not help, since the same
ID always maps to the same vector regardless of where it appears. Position information must be
injected explicitly, and there are two families of approaches.

##### Absolute position embeddings.

Each position $p\in\{0,\dots,T_{\max}-1\}$ gets its own vector $\boldsymbol{p}_p$, which is *added*
to the token embedding: $\boldsymbol{h}_p = \mathbf{E}[t_p] + \boldsymbol{p}_p$. The original
Transformer used fixed sinusoids, $\boldsymbol{p}_{p,2i}=\sin(p/10000^{2i/d})$,
$\boldsymbol{p}_{p,2i+1}=\cos(p/10000^{2i/d})$; GPT-2 uses a *learned* table
`nn.Embedding(context_length, d)`, optimised like any other weight. Learned absolute positions are
simple and effective, but the model has no vectors for positions beyond $T_{\max}$ (1024 for GPT-2),
so it cannot run on longer inputs at all.

![Sinusoidal position encodings for $d=64$ (analytic): each pair of dimensions oscillates at a
geometrically decreasing frequency, from period $2\pi$ (dimension 0) to very long periods. A
position is identified by the joint phase of all frequencies, like the hands of a clock. RoPE
([@ch:modern]) uses the same frequency ladder but applies it as a rotation of queries and
keys.](figures/ch08_text_data-10.svg){#fig:text:sinusoid}

The sinusoidal scheme has a property worth noticing: for any fixed offset $k$, the encoding of
position $p+k$ is a fixed rotation (a linear function independent of $p$) of the encoding of $p$,
because $\sin(\omega(p+k))$ and $\cos(\omega(p+k))$ are linear combinations of $\sin(\omega p)$ and
$\cos(\omega p)$. Relative offsets are therefore easy for a linear layer to read out. RoPE takes
this observation to its conclusion by applying the rotation to queries and keys directly, so that
attention scores depend on $i-j$ *exactly*.

##### Relative position encodings.

Here what matters is the *distance* between two tokens, not their absolute index. Variants add a
learned bias depending on $i-j$ to attention scores (T5), add a linear distance penalty (ALiBi;
Press et al., 2022), or---most popular today---rotate query and key vectors by position-dependent
angles so their dot product depends only on $i-j$ (RoPE; Su et al., 2021), used by Llama, Qwen3 and
most modern open models. [@ch:modern] implements RoPE and its context-extension tricks.

![GPT-2's input layer: token embeddings plus learned absolute position embeddings, broadcast across
the batch. The shapes in small print are the chapter's running
example.](figures/ch08_text_data-11.svg){#fig:text:posemb}

``` {#lst:text:inputemb .python caption="The complete input layer of GPT-2 ([@lab:text], Part E)."}
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, context_length, emb_dim):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, emb_dim)
        self.pos = nn.Embedding(context_length, emb_dim)

    def forward(self, ids):                       # ids: (B, T)
        T = ids.shape[1]
        pos = self.pos(torch.arange(T, device=ids.device))   # (T, d)
        return self.tok(ids) + pos[None]          # (B, T, d)

emb = InputEmbedding(vocab_size=50257, context_length=4, emb_dim=256)
x, y = next(iter(loader))           # loader with max_length=4, batch_size=8
emb(x).shape                        # torch.Size([8, 4, 256])
```

::: systemslens
[Position encoding and the serving stack]{.box-title}

The choice of positional scheme leaks into systems in non-obvious ways. (i) With learned absolute
positions, a *left-padded* batch shifts every row's positions, changing outputs; with RoPE only
relative offsets matter, so a uniform shift is harmless ([@ch:serving] and
`llmsys.generate.generate_batch`). (ii) Prefix caching across requests ([@ch:serving])
requires that a cached prefix's keys do not depend on what follows---true for all causal schemes,
but KV entries encoded with absolute or rotary positions are only reusable at the *same* positions.
(iii) Context length extension ([@ch:modern]) is a pure positional-encoding problem: RoPE can
be rescaled (position interpolation, NTK, YaRN) without retraining from scratch; a learned table
cannot.
:::

## The input pipeline at scale {#sec:text:pipeline}

Everything so far fits in memory: tokenize a 20 kB story, keep the IDs in a tensor, slice windows.
Pretraining corpora are measured in trillions of tokens. This section describes what changes.

##### Tokenize once, offline.

Tokenization is CPU work that does not depend on model weights, so it is done once in a
preprocessing job and the result stored as flat binary arrays of token IDs, a format popularised by
projects like nanoGPT and Megatron-LM. If $V\le 2^{16}$ (GPT-2), each token fits in `uint16`;
Qwen3's $151{,}936$-entry vocabulary needs `uint32`.

##### Memory-map the shards.

`np.memmap` maps a file into the address space without reading it; the OS pages in only the bytes a
worker touches. A dataset item is then an offset computation plus a small copy
([@lst:text:memmap]). This scales to files far larger than RAM, and many worker processes
can share the page cache.

``` {#lst:text:memmap .python caption="Writing and reading a token shard ([@lab:text], Part D)."}
def write_shard(ids, path, vocab_size):
    dtype = np.uint16 if vocab_size <= 2**16 else np.uint32
    arr = np.asarray(ids, dtype=dtype)
    arr.tofile(path)                     # raw little-endian array, no header
    return arr.nbytes

class ShardDataset(torch.utils.data.Dataset):
    def __init__(self, path, block_size, vocab_size):
        dtype = np.uint16 if vocab_size <= 2**16 else np.uint32
        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.block_size = block_size
    def __len__(self):
        return (len(self.data) - 1) // self.block_size
    def __getitem__(self, i):
        s = i * self.block_size
        raw = self.data[s:s + self.block_size + 1]      # touches only these pages
        chunk = torch.from_numpy(raw.astype(np.int64))
        return chunk[:-1], chunk[1:]
```

##### Pack documents.

Real corpora are collections of documents of wildly different lengths. Padding each to $T$ wastes
compute on padding tokens. Instead, documents are concatenated with `<|endoftext|>` separators into
one stream and cut into fixed windows---*packing*. A window may then contain the end of one document
and the start of another. Standard GPT-style pretraining lets tokens attend across the boundary (the
separator teaches the model to ignore what came before); more careful setups use block-diagonal
"document masks" so attention never crosses a boundary, which FlashAttention's variable-length
interface supports at no extra cost ([@ch:attention; @ch:instruct]).

##### Shuffle at several levels.

A global random permutation of $10^{9}$ windows is expensive to store and destroys locality.
Practical pipelines shuffle the *order of shards*, then the order of windows within a shard, and
sometimes add a streaming *shuffle buffer*. Determinism matters: for checkpoint/restart
([@ch:pretrain]) the loader must be able to resume at exactly the sample where it stopped,
which is easy with index arithmetic over memmapped shards and hard with stateful streaming
iterators.

##### Overlap loading with compute.

A loader created with `num_workers=k`, `pin_memory=True` and `prefetch_factor=2` runs $k$ worker
processes that fetch batches ahead of time into page-locked host memory, from which a non-blocking
`.to("cuda", non_blocking=True)` copy can overlap the previous step's compute.

### Variable-length batches: padding versus packing {#sec:text:padding}

Fine-tuning and serving work with *individual* examples of different lengths (an SMS, an
instruction--response pair, a chat request) rather than an endless stream. There are two ways to
batch them ([@fig:text:padpack]). *Padding* right-pads every sequence to the longest in the
batch with a filler ID and masks the padding out of attention and loss. It is simple and keeps
examples independent, but the padded positions still flow through every matmul. *Packing*
concatenates several examples into each row, separated by end-of-text tokens, and uses
document-aware masks (or position resets) so examples do not attend to each other. Packing wastes
almost nothing but complicates masking, position IDs, and per-example loss normalisation.

![The same four examples (lengths 9, 4, 12, 6) batched by padding and by packing. Grey cells are
wasted compute.](figures/ch08_text_data-12.svg){#fig:text:padpack}

::: napkinmath
[Expected padding waste]{.box-title}

If lengths are i.i.d. uniform on $\{1,\dots,T\}$ and every batch is padded to $T$, the expected
fraction of real tokens is $\mathbb{E}[\ell]/T = (T+1)/(2T)\approx 50\%$: half the FLOPs of every
linear layer are spent on padding (attention is worse, since its cost is quadratic in the padded
length). Padding only to the longest sequence *in the batch* helps less than one might hope for
large batches: with $B=32$, the batch maximum of uniform lengths is close to $T$ already
($\mathbb{E}[\max]\approx T\cdot B/(B+1)$). Sorting or *bucketing* examples by length before
batching ([@ch:classify]) recovers most of the waste without changing the model; packing
recovers nearly all of it.
:::

![A production pretraining input pipeline. Tokenization happens once, offline; the per-step path is
reduced to offset arithmetic, page-cache reads and an asynchronous host-to-device
copy.](figures/ch08_text_data-13.svg){#fig:text:pipeline}

::: napkinmath
[Can the data pipeline keep up?]{.box-title}

Suppose a cluster trains a 7B model at 40% MFU on 256 H100s. Using $6N$ FLOPs per token, the token
throughput is $0.4\cdot 256\cdot 989\times10^{12}/(6\cdot 7\times10^9)\approx 2.4$M tokens/s.

- *Reading pre-tokenized shards*: $2.4$M tokens/s $\times$ 4 bytes (`uint32`) $\approx 9.6$ MB/s for
  the whole cluster---negligible against any file system or NIC.

- *Tokenizing on the fly*: at $\sim$`<!-- -->`{=html}4 bytes of text per token, the cluster needs
  $\sim$`<!-- -->`{=html}9.6 MB/s of text tokenized. Our pure-Python encoder ran at
  $\approx 0.3$ MB/s per core in this environment, so it would need about 30 dedicated cores; a Rust
  tokenizer at tens of MB/s per core needs one or two. Either is feasible, but CPU spent tokenizing
  competes with loader workers, and offline tokenization also makes runs reproducible.

The lesson: once data is pre-tokenized, I/O bandwidth is almost never the bottleneck for text LLMs;
Python-level per-sample overhead (and, for multimodal models, image decoding) is.
:::

::: deepdive
[Data quality is a systems problem too]{.box-title}

Before tokenization, web-scale corpora go through language identification, heuristic quality
filters, model-based classifiers, PII scrubbing, and deduplication. Exact dedup hashes documents;
near-duplicate detection uses MinHash signatures with locality-sensitive hashing over $n$-gram
shingles, which turns an $O(n^2)$ comparison into a bucketed join. These jobs run on CPU clusters
with Spark/Ray-style frameworks and dominate the engineering effort of datasets such as FineWeb
(Penedo et al., 2024). Benchmark *contamination*---test questions leaking into pretraining data---is
detected with the same $n$-gram machinery ([@ch:eval]).
:::

## Putting it together: from a file to a $(B,T,d)$ tensor {#sec:text:e2e}

[@lst:text:e2e] strings the chapter's components together using the shared `llmsys`
package. It trains a small BPE tokenizer on a text file, tokenizes the file, builds a loader of
disjoint windows, and produces input embeddings ready for the attention layers of
[@ch:attention]. [@tab:text:shapes] traces the shapes.

``` {#lst:text:e2e .python caption="End-to-end input pipeline with the `llmsys` core modules."}
import torch
from llmsys.tokenizer import ByteBPETokenizer
from llmsys.data import create_dataloader

text = open("the-verdict.txt", encoding="utf-8").read()
tok = ByteBPETokenizer.train(text, vocab_size=1024)          # 256 bytes + merges
loader = create_dataloader(text, tok, batch_size=8, max_length=4, stride=4,
                           shuffle=False)
x, y = next(iter(loader))                     # (8, 4) inputs, (8, 4) targets

d, T = 256, 4
tok_emb = torch.nn.Embedding(tok.vocab_size, d)
pos_emb = torch.nn.Embedding(T, d)
h = tok_emb(x) + pos_emb(torch.arange(T))     # (8, 4, 256)
```

::: {#tab:text:shapes .table-float}
| stage               | object               | shape                                    | dtype     |
|:--------------------|:---------------------|:-----------------------------------------|:----------|
| raw text            | `str`                | $\approx 2\times10^4$ characters         | ---       |
| token IDs           | `list[int]`          | $N\approx 5\times 10^3$ (depends on $V$) | int       |
| one batch           | `x`, `y`             | $(B,T) = (8,4)$                          | `int64`   |
| token embeddings    | `tok_emb(x)`         | $(8,4,256)$                              | `float32` |
| position embeddings | `pos_emb(arange(T))` | $(4,256)$, broadcast                     | `float32` |
| input to block 1    | `h`                  | $(8,4,256)$                              | `float32` |

: Shapes through the input pipeline for the running example ($B=8$, $T=4$, $d=256$).
:::

::: deepdive
[Do we need tokenizers at all?]{.box-title}

Tokenization is a fixed, non-learned compression stage bolted onto an otherwise end-to-end model,
and many artefacts above stem from it. Byte-level models avoid it by operating directly on bytes, at
the cost of sequences about four times longer. Hierarchical architectures recover efficiency by
grouping bytes into patches processed by a large "global" model and small "local" models,
e.g. MegaByte (Yu et al., 2023), or by forming patches dynamically from next-byte entropy, as in the
Byte Latent Transformer (Pagnoni et al., 2024). For now, every widely deployed LLM---including
everything in this book---uses a subword tokenizer.
:::

## Summary {#sec:text:summary}

- LLMs consume integer token IDs, which an embedding table turns into $d$-dimensional vectors. The
  table is learned jointly with the model; its forward pass is a gather and its gradient a sparse
  scatter-add.

- A tokenizer splits text and maps pieces to IDs through a vocabulary. Word-level tokenizers need
  `<|unk|>` for unseen words and a huge vocabulary for coverage.

- Byte-level BPE starts from 256 bytes and greedily merges frequent pairs; it can encode any input,
  keeps whitespace, and is what GPT-2 (50,257 tokens) and Qwen3 (151,936) use. Encoding applies
  merges in training order.

- Vocabulary size trades parameters and logit cost against sequence length; for large vocabularies
  the training-time logits tensor can exceed all other memory.

- Training examples are windows of $T$ tokens with targets shifted by one; stride controls overlap.
  Datasets/DataLoaders batch them into $(B,T)$ tensors.

- Position must be injected explicitly. GPT-2 adds learned absolute position embeddings; modern
  models use rotary embeddings ([@ch:modern]).

- At scale: tokenize offline into binary shards, memory-map them, pack documents with separators,
  shuffle hierarchically, and overlap loading with compute.

## Exercises {#sec:text:exercises}

::: exercise
[stride arithmetic]{.box-title}

A corpus has $N$ tokens. Give a closed form for the number of $(x,y)$ windows produced by
`SlidingWindowDataset` with context length $T$ and stride $s$, and for the total number of *input
token positions* processed per epoch. How many times is a typical token seen as an input per epoch
when $s=1$? when $s=T/4$?
:::

::: exercise
[tokenizer round-trip]{.box-title}

Construct three strings for which the word-level `WordTokenizer` of [@lst:text:wordtok]
fails `decode(encode(s)) == s` even when every word is in the vocabulary. Explain why byte-level BPE
does not have this problem.
:::

::: exercise
[unknown words in BPE]{.box-title}

Using a BPE tokenizer trained on English prose, encode a string of random lowercase letters of
length 20 and a 20-character word in a non-Latin script. Report bytes per token for each and explain
the difference in terms of the merge list.
:::

::: exercise
[merge rank, not frequency]{.box-title}

Explain why `bpe_encode` must apply merges in the order they were learned rather than, say, greedily
merging the most frequent pair *in the input string*. Give a concrete merge list and input where the
two strategies produce different token sequences.
:::

::: exercise
[complexity of incremental BPE]{.box-title}

Let $W$ be the total number of symbols across unique pre-token chunks, $M$ the number of merges, and
$a_m$ the number of chunks affected by merge $m$. Give the time complexity of the naive and
incremental trainers. What additional data structure reduces the "find the best pair" step from
$O(P)$ to $O(\log P)$, and what complication do decreasing counts introduce?
:::

::: exercise
[vocabulary trade-off]{.box-title}

For a model with $d=2048$, $L=24$ layers, and a fixed budget of $10^{11}$ training tokens of English
text, compare $V=32\text{k}$ and $V=256\text{k}$, assuming the larger vocabulary compresses text 15%
better. Estimate (a) embedding parameters (tied), (b) logits FLOPs per token, (c) the number of
training *tokens* needed to cover the same text, and (d) the size of the fp32 logits tensor for
$B=4$, $T=8192$. Which choice would you make for a 0.5B model? For a 70B model?
:::

::: exercise
[embedding gradients]{.box-title}

Write a test that shows `emb.weight.grad` is non-zero only in rows whose IDs occur in the batch, and
that a row whose ID occurs $k$ times receives the sum of $k$ gradient vectors. How does
`nn.Embedding(..., sparse=True)` change the gradient's representation, and which optimisers accept
it?
:::

::: exercise
[positional permutation test]{.box-title}

Build a single self-attention layer without positional information and verify numerically that
permuting the input rows permutes the output rows identically. Then add learned position embeddings
and show the property disappears.
:::

::: exercise
[sinusoids]{.box-title}

Show that for the sinusoidal encoding there is a linear map $\mathbf{M}_k$, independent of $p$, with
$\boldsymbol{p}_{p+k} = \mathbf{M}_k \boldsymbol{p}_p$. Why does this make it plausible for a
network to learn relative-position behaviour from absolute sinusoidal encodings?
:::

::: exercise
[document masks]{.box-title}

Packing places several documents into one window. Write a function that, given a window's token IDs
and the end-of-text ID, returns the $(T,T)$ boolean mask that is causal *and* block-diagonal by
document. How many attention scores does it save, relative to the plain causal mask, for a window
with four documents of equal length?
:::

::: exercise
[pipeline measurement]{.box-title}

Using `labs/lab08_tokenizer/bench.py`, measure random-access window reads per second from a
memmapped shard with (a) a warm page cache and (b) after dropping caches (Linux:
`echo 3 > /proc/sys/vm/drop_caches`, as root). Relate the cold-cache number to your disk's
random-read IOPS.
:::

## Lab 4: Tokenizers and the input pipeline {#sec:text:lab}

::: labbox
[Tokenizers and the input pipeline]{.box-title}

**Code:** `code/labs/lab08_tokenizer`

**Goals**

Build every stage between a text file and a $(B,T,d)$ tensor yourself, and measure the costs that
matter at scale: tokenizer training time, encode throughput, compression, and shard access rate.

**Setup**

``` bash
cd code
pytest -q labs/lab08_tokenizer                    # runs against starter.py
LAB_IMPL=solution pytest -q labs/lab08_tokenizer  # reference
```

Each task corresponds to one stubbed function in `starter.py`; docstrings give the contract and a
hint.

**Tasks**

1.  **Word-level tokenizer.** Implement `split_text`, `build_vocab`, `WordTokenizer.encode/decode`
    with `<|unk|>` fallback. The tests check punctuation handling and round-trips.

2.  **BPE primitives.** Implement `get_pair_counts` (frequency-weighted) and `merge_pair`
    (left-to-right, non-overlapping: `[1,1,1]` with pair `(1,1)` becomes `[7,1]`).

3.  **BPE training and coding.** Implement `train_bpe`, `bpe_encode` (apply merges by rank),
    `bpe_decode`, `compression_ratio`. Round-trip must be exact for arbitrary Unicode.

4.  **Incremental trainer.** Implement `train_bpe_fast` with pair counts plus an inverted index. It
    must return *exactly* the same merges as `train_bpe` (deterministic tie-breaking is provided by
    `_best_pair`).

5.  **Sampling and storage.** Implement `sliding_windows`, `pack_documents`, `write_shard`
    (narrowest dtype) and `ShardDataset.__getitem__`.

6.  **Input layer.** Implement `InputEmbedding.forward` and `embedding_as_matmul`.

**Measurements**

Run `bench.py` (optionally `--text` on a real corpus such as a public-domain book):

- training time of naive vs. incremental BPE as a function of `--vocab`; plot both. Expect the gap
  to grow with vocabulary size and corpus diversity;

- encode throughput (MB/s) and bytes/token of your tokenizer, and of `tiktoken`'s GPT-2 encoding if
  available. Expect the Rust tokenizer to be one to two orders of magnitude faster;

- random window reads per second from the memmapped shard.

**Deliverables**

All tests passing; a short table of the measurements; a two-sentence answer to: *at what cluster
token throughput would on-the-fly tokenization with your implementation need more CPU cores than a
typical 8-GPU node has?*

**Stretch goals**

- Replace `_best_pair`'s linear scan with a max-heap using lazy invalidation; verify identical
  merges.

- Add document-boundary-aware window sampling that returns, with each window, the block-diagonal
  mask from [@sec:text:exercises].

- Load GPT-2's real merges (from `tiktoken`'s files) into `ByteBPETokenizer` and check
  token-for-token agreement on a test corpus (requires network access on your machine).
:::
