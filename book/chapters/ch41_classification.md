# Fine-Tuning for Classification {#ch:classify}

A pretrained language model is a general-purpose engine for predicting the next token. Most deployed
models, though, have a narrower job: route a support ticket, flag a message as spam, decide whether
a code change touches a security-sensitive file. *Classification fine-tuning* is the most direct way
to turn a pretrained model into such a component. We keep the transformer trunk, throw away the
50,257-way vocabulary projection, put a tiny $C$-way projection in its place, and train on labelled
examples.

The modelling change is small, but it raises several systems questions that recur for the rest of
the book. Which parameters do we update, and what does freezing the rest save in memory and in
FLOPs? How much of each batch is padding, and how do we stop paying for it? Which token's hidden
state should the head read, and does that answer change with padding? A classifier is also the
simplest example of a *prefill-only* inference workload: one forward pass, no decoding loop, no KV
cache that outlives the request. That makes it a useful contrast to the generative serving systems
of [@ch:inference; @ch:serving].

![Chapter roadmap. The top row is the modelling pipeline for turning a pretrained GPT into a spam
classifier; the dashed boxes are the systems questions we attach to each
stage.](figures/ch41_classification-1.svg){#fig:classify:roadmap}

## The fine-tuning landscape {#sec:classify:landscape}

Once pretraining ([@ch:pretrain]) is done, there are two classic ways to adapt a language
model to a task.

Instruction fine-tuning

: trains the model to read a natural-language request and answer in natural language. The output
  space stays the full vocabulary, so a single model can handle a wide range of tasks: summarise,
  translate, fix the grammar, explain a stack trace. The cost is that it needs diverse, carefully
  formatted data, and the outputs are free text that must be parsed. This is [@ch:instruct].

Classification fine-tuning

: trains the model to emit one label from a fixed set. The output space shrinks to $C$ classes
  (spam/not-spam, sentiment, intent, toxicity category). The model is useless for anything else, but
  it is cheap to train, its output needs no parsing, and its accuracy is easy to measure.

In a production pipeline you often need both. An instruction-tuned assistant may sit behind a
battery of small classifiers that screen inputs, route requests to specialised models, or detect
policy violations. These classifiers run on *every* request, so their cost per token matters more
than their absolute size suggests.

![Ways to adapt a pretrained model. *What* we optimise (labels, responses, rewards) is one axis;
*which parameters* we update (all, the last few layers, or low-rank adapters) is a separate axis
that applies to every method.](figures/ch41_classification-2.svg){#fig:classify:landscape}

A second axis, separate from the first, is *how many parameters move*. You can update all of them
(full fine-tuning), only the top few layers plus the new head (the default in this chapter), or
small low-rank adapters (LoRA, [@sec:instruct:lora]). The choice sets the memory footprint,
the per-step compute, and how much capacity the model has to adapt.

::: systemslens
[What each adaptation regime keeps in memory]{.box-title}

For a model with $N$ parameters trained with AdamW in mixed precision, the persistent state is about
$16N$ bytes ([@ch:pretrain]): bf16 weights (2), bf16 gradients (2), fp32 master weights (4),
and two fp32 Adam moments (8). Freezing a parameter removes everything except the 2-byte weight. If
only a fraction $f$ of the parameters is trainable, the persistent state is roughly $2N + 14fN$
bytes. For GPT-2 small with only the last block, the final LayerNorm and the head trainable,
$f\approx 0.057$ ([@sec:classify:head]). The optimizer state then becomes a rounding error,
and the memory budget is set by weights and activations.
:::

::: napkinmath
[Full vs. partial fine-tuning of GPT-2 small]{.box-title}

GPT-2 small has $N\approx124\text{M}$ unique parameters. Full fine-tuning with AdamW:
$16 \times 124\text{M} \approx 2.0\,\mathrm{GB}$ of persistent state. Training only the last block,
the final norm and a 2-way head ($\approx 7.1$M parameters):
$2 \times 124\text{M} + 14 \times 7.1\text{M} \approx 248 + 99 = 347\,\mathrm{MB}$. The six-fold
reduction matters little on a datacenter GPU. On a laptop, or when one GPU hosts dozens of
task-specific classifiers, it decides whether the job fits.
:::

## Preparing the dataset {#sec:classify:data}

Our running example is the *SMS Spam Collection*, a public corpus of 5,574 short text messages. Each
is labelled `ham` (legitimate, 4,827 messages) or `spam` (747 messages). It is distributed as a
tab-separated file, one `label<TAB>text` pair per line. The data is small enough to fine-tune GPT-2
small on a laptop in minutes, and realistic enough to show the usual problems: class imbalance,
long-tailed message lengths, and near-duplicate messages.

##### Class imbalance.

Only $747/5574 \approx 13.4\%$ of messages are spam. A model that always predicts `ham` already
scores 86.6% accuracy, which makes accuracy a poor signal early in training. The simplest fix is
*undersampling*: keep every spam message and draw an equal-sized random subset of 747 ham messages,
giving a balanced dataset of 1,494 examples. Undersampling throws data away, which is acceptable
here because we want a fast, interpretable experiment. In production you would more likely keep all
the data and use one of the following:

- a class-weighted loss, where the weight of class $c$ is proportional to $1/\text{freq}(c)$;

- oversampling the minority class, either by repetition or by generating paraphrases with an LLM;

- reporting precision, recall and F1 (or the area under the precision--recall curve) instead of
  accuracy, and choosing a decision threshold for the cost you care about. Here that cost is usually
  a false positive: a real message hidden as spam.

``` {#lst:classify:balance .python caption="Undersampling to balance classes, then a seeded three-way split (from `llmsys/finetune.py`)."}
def balance_by_undersampling(labels, seed=123):
    by_class = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)
    n_min = min(len(v) for v in by_class.values())
    rng = random.Random(seed)
    keep = []
    for y in sorted(by_class):
        idx = by_class[y]
        keep.extend(idx if len(idx) == n_min else rng.sample(idx, n_min))
    rng.shuffle(keep)
    return keep

def split_indices(n, train_frac=0.7, val_frac=0.1, seed=123):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    a = int(n * train_frac)
    b = a + int(n * val_frac)
    return idx[:a], idx[a:b], idx[b:]       # test = the remainder
```

##### Splitting.

We shuffle once with a fixed seed and cut the balanced set 70/10/20 into training (1,045),
validation (149) and test (300) examples. Labels are mapped to integers, `ham`$\to 0$ and
`spam`$\to 1$, because `cross_entropy` expects class indices.

![From the raw SMS collection to a balanced, three-way split, drawn to scale. Undersampling discards
about 85% of the ham messages.](figures/ch41_classification-3.svg){#fig:classify:split}

::: pitfall
[Leakage through duplicates]{.box-title}

SMS corpora contain many exact and near-exact duplicates, especially among spam campaigns that send
the same template to thousands of numbers. If a message appears in both training and test, test
accuracy measures memorisation rather than generalisation. Deduplicate *before* splitting: hash a
normalised form (lowercased, whitespace collapsed, digits masked) and keep one copy per hash. The
same discipline applies to the much larger contamination problem in LLM evaluation
([@ch:eval]).
:::

## Creating data loaders {#sec:classify:loaders}

Text messages vary in length, but a tensor is a rectangle. The SMS messages, tokenised with the
GPT-2 BPE tokenizer ([@ch:text]), run from a handful of tokens to well over a hundred. There
are two simple ways to make them rectangular:

1.  **Truncate** every message to the length of the shortest one. This is cheap but destroys
    information, because most spam cues sit beyond the first few tokens.

2.  **Pad** every message with a filler token up to some length $T_{\max}$. Since GPT-2 has no
    dedicated padding token, we reuse `<|endoftext|>` (id 50256). That is safe as long as the model
    never reads the hidden states at padded positions; see [@sec:classify:head].

The simplest padding policy fixes $T_{\max}$ to the longest training message (120 tokens for this
dataset) and pads the validation and test sets to the same length, truncating anything longer. Every
batch then has the same shape, which is convenient for debugging and for compilers that specialise
on shapes.

![Tokenise, then pad to a common length. Every grey cell costs as much compute as a real token and,
in attention, adds a row and a column to the score
matrix.](figures/ch41_classification-4.svg){#fig:classify:pad}

##### Dynamic padding.

A better default is to pad each *batch* to its own longest member. The dataset stores
variable-length token lists, and the collate function pads them when the batch is assembled. It also
returns the true lengths, which we will need to find the last real token.

``` {#lst:classify:collate .python caption="A dataset that stores token lists, and a collate function that pads per batch and returns lengths."}
class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=None):
        self.encoded = [tokenizer.encode(t)[:max_length] if max_length
                        else tokenizer.encode(t) for t in texts]
        self.labels = [int(y) for y in labels]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return self.encoded[i], self.labels[i]

def pad_batch(seqs, pad_id, side="right", max_length=None):
    if max_length is not None:
        seqs = [list(s)[:max_length] for s in seqs]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        t = torch.as_tensor(list(s), dtype=torch.long)
        if side == "right": ids[i, :len(s)] = t
        else:               ids[i, T - len(s):] = t
    return ids, torch.tensor([len(s) for s in seqs])

def classification_collate(batch, pad_id, side="right"):
    seqs, labels = zip(*batch)
    ids, lengths = pad_batch(seqs, pad_id, side)
    return ids, lengths, torch.tensor(labels)
```

We wrap the three splits in `DataLoader`s with batch size 8, shuffling and `drop_last=True` for
training only. Dropping the last incomplete batch keeps every optimisation step at the same batch
size. Validation and test keep all their examples, so every example is scored. A quick sanity check
is to iterate once and print the shapes: with batch size 8 and global padding, input batches are
$8 \times 120$ and label batches are $8$. With dynamic padding the second dimension varies from
batch to batch.

::: systemslens
[The real price of padding]{.box-title}

Padding costs more than the extra tokens suggest. For a batch of $B$ sequences padded to $T$, the
MLP and projection FLOPs scale with $BT$, while attention scores scale with $BT^2$. A single
120-token outlier in a batch of 10-token messages makes that batch about $12\times$ more expensive
in the linear layers and about $144\times$ more expensive in the attention score matrix. Two levers
fix this:

- **Length bucketing.** Group sequences of similar length into the same batch. Sorting the whole
  dataset by length destroys the randomness SGD relies on. A standard compromise is to shuffle, cut
  the data into *mega-batches* of, say, $50\times$ the batch size, sort within each mega-batch,
  slice into batches, and shuffle the batch order ([@lst:classify:bucket]).

- **Token-budget batching.** Fix the number of *tokens* per batch instead of the number of
  sequences, so batches of short messages hold more of them. This keeps GPU utilisation and memory
  use flat. Large-scale training and serving systems almost always batch by tokens.

The next step after bucketing is *packing* ([@sec:instruct:packing]), which removes padding
completely by concatenating sequences and masking attention between them.
:::

``` {#lst:classify:bucket .python caption="Sort-within-mega-batch bucketing and a padding-waste meter."}
def length_bucketed_batches(lengths, batch_size, seed=0, bucket_mult=50):
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    mega = batch_size * bucket_mult
    batches = []
    for s in range(0, len(idx), mega):
        chunk = sorted(idx[s:s + mega], key=lambda i: lengths[i])
        batches.extend(chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size))
    rng.shuffle(batches)                  # randomise batch order
    return batches

def padding_stats(lengths, batches):
    real = slots = 0
    for b in batches:
        L = [lengths[i] for i in b]
        real += sum(L)
        slots += max(L) * len(L)
    return dict(real_tokens=real, padded_slots=slots, waste=1 - real / max(slots, 1))
```

[@fig:classify:waste] shows what bucketing buys. We generated 4,000 synthetic message
lengths with a log-normal distribution (median 13 tokens, maximum 108), similar in shape to real SMS
data, and measured the fraction of computed token slots that are padding. Padding everything to the
global maximum wastes 86% of all slots. Dynamic padding with random batches wastes 40--76%, rising
with batch size because a larger batch is more likely to contain an outlier. Bucketing keeps waste
below 10% at every batch size.

![Padding waste versus batch size, measured in this environment on 4,000 synthetic log-normal
message lengths (`labs/lab41_classify`). Bucketing with mega-batches of $50\times$ the batch size
removes nearly all of the waste.](figures/ch41_classification-5.svg){#fig:classify:waste}

## Initializing a model with pretrained weights {#sec:classify:init}

We start from GPT-2 small (124M) and load the released weights into the `GPTModel` of
[@ch:gpt] with `load_gpt2_hf_state_dict` ([@ch:pretrain]). Two settings must match the
checkpoint: the released model has biases on the query/key/value projections (`qkv_bias=True`), and
it uses a 1,024-token context. We also set dropout to zero, since we will not fine-tune long enough
for regularisation to matter.

``` {#lst:classify:load .python caption="Loading GPT-2 small for fine-tuning."}
from safetensors.torch import load_file
from llmsys.gpt import GPTModel, gpt2_config, load_gpt2_hf_state_dict

cfg = gpt2_config("gpt2-small (124M)", drop_rate=0.0)   # qkv_bias=True, ctx 1024
model = GPTModel(cfg)
load_gpt2_hf_state_dict(model, load_file("gpt2/model.safetensors"))
model.eval()
```

Before changing anything, run two sanity checks. First, generate a continuation of a neutral prompt
("Every effort moves you") with greedy decoding and confirm the text is fluent English. A garbled
continuation means the weight mapping is wrong, most often because a Conv1D weight was not
transposed. Second, see whether the base model can already classify when prompted:

    Is the following text 'spam'? Answer with 'yes' or 'no': 'You are a winner you
    have been specially selected to receive $1000 cash or a $2000 award.'

A base GPT-2 typically repeats the question or wanders off instead of answering. It was never
trained to follow instructions ([@ch:instruct]). That is the motivation for replacing the
output layer and training it directly.

::: keyidea
Fine-tuning works because the pretrained trunk already produces hidden states that encode much of
what a classifier needs: topic, register, sentiment, and the presence of URLs, prize language or
urgency cues. The new head only has to find a linear boundary in that representation space.
Unfreezing the last block lets the features themselves bend slightly toward the task.
:::

### Where does the task information live? {#sec:classify:probe}

Before deciding how many blocks to unfreeze, it helps to measure how linearly separable the classes
already are at each depth of the frozen network. A *linear probe* trains only a logistic-regression
layer on the last-real-token hidden state taken after block $\ell$, for each $\ell$. Forward hooks
collect all twelve representations in a single pass over the data:

``` {#lst:classify:probe .python caption="Collecting per-layer last-token features with forward hooks."}
@torch.no_grad()
def collect_layer_features(model, loader, device="cpu"):
    feats = {i: [] for i in range(len(model.trf_blocks))}
    current = {}
    hooks = [blk.register_forward_hook(
                 lambda m, inp, out, i=i: current.__setitem__(i, out))
             for i, blk in enumerate(model.trf_blocks)]
    labels = []
    for ids, lengths, y in loader:
        model(ids.to(device))
        rows = torch.arange(ids.shape[0])
        for i, h in current.items():
            feats[i].append(h[rows, lengths - 1].float().cpu())  # (B, d)
        labels.append(y)
    for h in hooks:
        h.remove()
    return {i: torch.cat(v) for i, v in feats.items()}, torch.cat(labels)
```

The resulting curve of probe accuracy against depth is a cheap guide to where to cut the network. If
accuracy is already high at block 8, you can drop blocks 9--12 entirely for this task, which cuts
serving cost by a third. Or you can freeze through block 8 and cache its outputs
([@sec:classify:head]). The memory cost of probing is the number of examples $\times$
layers $\times$ $d$ floats: $1{,}045\times12\times768\times4$ bytes $\approx39\,\mathrm{MB}$, small
because only one position per message is kept. [@ex:classify:probe] asks you to produce the
curve for GPT-2 small on the SMS data.

## Adding a classification head {#sec:classify:head}

The pretrained `out_head` maps the final 768-dimensional hidden state to 50,257 vocabulary logits.
We replace it with `nn.Linear(768, 2)`: one logit for `ham` and one for `spam`. With two classes we
could get away with a single logit and a sigmoid. Keeping $C$ logits and the softmax cross-entropy
loss means the same code handles any number of classes.

##### Which layers to train.

Lower transformer layers in pretrained LMs tend to encode general lexical and syntactic features
that transfer across tasks. Upper layers are more specialised to next-token prediction. A common,
cheap recipe is therefore:

1.  freeze every parameter;

2.  replace `out_head` (new parameters are trainable by default);

3.  unfreeze the last transformer block and the final LayerNorm.

Training more blocks usually helps a little and costs more; training all of them is full
fine-tuning. [@ex:classify:layers] asks you to measure the trade-off.

``` {#lst:classify:head .python caption="Swapping the head and choosing what to train."}
def add_classification_head(model, num_classes, trainable_blocks=1,
                            train_final_norm=True):
    for p in model.parameters():
        p.requires_grad = False
    emb_dim = model.out_head.in_features
    model.out_head = nn.Linear(emb_dim, num_classes)     # trainable
    if trainable_blocks > 0:
        for blk in model.trf_blocks[-trainable_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
    if train_final_norm:
        for p in model.final_norm.parameters():
            p.requires_grad = True
    return model
```

![The classifier: a frozen GPT-2 trunk, a trainable top block and final norm, and a new two-way head
in place of the vocabulary projection.](figures/ch41_classification-6.svg){#fig:classify:frozen}

With this recipe, 7,090,946 of the model's parameters are trainable: the last block (7,087,872), the
final norm (1,536) and the head ($768\times2+2 = 1{,}538$). That is 5.7% of the network.

::: systemslens
[Freezing saves backward compute and activation memory, not just optimizer state]{.box-title}

Autograd records a node only for operations whose inputs require gradients. With blocks 1--11
frozen, their inputs (the embedding outputs) do not require gradients, so no graph is recorded for
them and their intermediate activations are freed right after the forward pass. Backward stops at
the input of block 12. Two consequences:

- **Compute.** A training step costs roughly $2N$ FLOPs per token forward plus
  $4N_{\text{trainable}}$ backward, instead of $6N$. With one of twelve blocks trainable, that is
  about $(2 + 4/12)/6 \approx 0.39$ of a full fine-tuning step, or $2.6\times$ cheaper.

- **Trunk caching.** If the bottom 11 blocks never change, their output for a given message never
  changes either. You can run them once over the dataset, store the $T\times 768$ hidden states, and
  train only block 12 and the head on the cache. The stored hidden states fit easily: 1,045 messages
  $\times$ 120 positions $\times$ 768 dimensions $\times$ 2 bytes $\approx 193\,\mathrm{MB}$. This
  is a big win when you sweep hyperparameters over many epochs.

LoRA ([@sec:instruct:lora]) is a different case. Adapters sit in *every* layer, so the
activation gradient must flow through the whole network, and the savings come only from skipping the
weight-gradient GEMMs of the frozen weights.
:::

### Which token should the head read?

The model returns logits of shape $(B, T, C)$, one $C$-vector per position. We need one vector per
message. Recall the causal mask from [@ch:attention]: position $t$ can attend only to
positions $0,\dots,t$. Only the *last* position has attended to the whole message, so its hidden
state is the natural summary to classify. The first position has seen a single token.

![Causal attention over a six-token message. Only the final position aggregates information from the
whole sequence, so its output is the one we
classify.](figures/ch41_classification-7.svg){#fig:classify:lasttoken}

With *right* padding, the last position of a padded row is a padding token, not the last real token.
A naive implementation reads `logits[:, -1, :]` anyway. That works reasonably well, because the
padding token at the end still attends to every real token before it. But it makes the prediction
depend on *how much* padding follows the message, which in turn depends on which other messages
happen to share the batch. Two cleaner options:

1.  **Gather the last real token.** With the lengths returned by the collate function, pick row $b$
    at position $\ell_b - 1$. The prediction is then exactly invariant to padding (the lab tests
    this).

2.  **Left padding with a key-padding mask.** Put the padding before the message and mask padded
    keys (the `pad_mask` argument of the core models). The last position is now always real, so
    `logits[:, -1]` is correct. This is what batched generation does ([@ch:serving]). With
    learned absolute positions (GPT-2) the real tokens shift to higher position ids, which changes
    the outputs slightly; with RoPE it does not.

``` {#lst:classify:select .python caption="Selecting the per-message logits."}
def select_class_logits(logits, lengths=None, side="right"):
    """logits: (B, T, C) -> (B, C)."""
    if lengths is None or side == "left":
        return logits[:, -1, :]
    idx = (lengths - 1).clamp(min=0).to(logits.device)
    return logits[torch.arange(logits.shape[0], device=logits.device), idx]
```

::: deepdive
[Other pooling choices]{.box-title}

Encoder models such as BERT classify from a dedicated `[CLS]` token placed at the *start*. That
works only because encoders use bidirectional attention. In a causal decoder you can also mean-pool
the hidden states of the real tokens. This is sometimes more robust for long inputs, but it dilutes
the strongest signal, which tends to sit at the end. Embedding models built from decoders
(e5-mistral-style) typically use last-token pooling with an appended end-of-sequence token, which is
the recipe of this chapter.
:::

## Calculating the classification loss and accuracy {#sec:classify:loss}

For a message $x$ with selected logits $\boldsymbol{z}\in\mathbb{R}^{C}$, the predicted class
distribution is $p(c\mid x) = \operatorname{softmax}(\boldsymbol{z})_c$. The predicted label is
$\hat{y} = \operatorname*{arg\,max}_c z_c$. Softmax is monotonic, so we can take the
$\operatorname*{arg\,max}$ of the logits directly. Training minimises the mean cross-entropy over a
batch, $$\begin{equation}
\mathcal{L} = -\frac{1}{B}\sum_{b=1}^{B} \log \operatorname{softmax}(\boldsymbol{z}_b)_{y_b}
            = \frac{1}{B}\sum_{b=1}^{B}\Bigl(\log\sum_{c} e^{z_{b,c}} - z_{b,y_b}\Bigr),
\label{eq:classify:ce}
\end{equation}$$ the same loss as in pretraining ([@ch:pretrain]). It is applied once per
message instead of once per token, over $C=2$ classes instead of 50,257. We cannot optimise accuracy
directly because it is piecewise constant, so its gradient is zero almost everywhere. Cross-entropy
is a smooth surrogate for it.

``` {#lst:classify:acc .python caption="Loss on a batch and accuracy over a loader."}
def classification_loss(model, ids, lengths, labels, device="cpu", use_lengths=True):
    ids, labels = ids.to(device), labels.to(device)
    logits = select_class_logits(model(ids), lengths if use_lengths else None)
    return F.cross_entropy(logits.float(), labels)

@torch.no_grad()
def accuracy_loader(model, loader, device="cpu", num_batches=None, use_lengths=True):
    model.eval()
    correct = total = 0
    for i, (ids, lengths, labels) in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break
        logits = select_class_logits(model(ids.to(device)),
                                     lengths if use_lengths else None)
        correct += (logits.argmax(-1).cpu() == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)
```

Evaluated before any fine-tuning, the new head is random, so accuracy on the balanced splits sits
near 50%, which is chance. The `num_batches` argument estimates accuracy from a few batches, which
is useful for frequent cheap checks during training.

::: pitfall
[Computing the loss in low precision]{.box-title}

Under bf16 autocast the logits come out in bf16, which has only 8 bits of mantissa. A log-sum-exp
over large vocabularies loses accuracy in that format. With two classes the damage is small, but the
habit of casting logits to fp32 (`logits.float()`) before `cross_entropy` costs nothing here and
matters a lot in [@ch:pretrain; @ch:grpo].
:::

## Fine-tuning the model on supervised data {#sec:classify:train}

The training loop is the pretraining loop of [@ch:pretrain] with three changes: the loss is
computed on the selected logits, only trainable parameters are passed to the optimizer, and we track
accuracy alongside loss. We use AdamW with learning rate $5\times10^{-5}$ and weight decay $0.1$ for
about five epochs. On a modern laptop CPU this takes several minutes for GPT-2 small; on a GPU, well
under a minute.

``` {#lst:classify:train .python caption="Classification fine-tuning loop (lab reference)."}
def train_classifier(model, train_loader, val_loader, epochs=3, lr=5e-4,
                     weight_decay=0.1, device="cpu", use_lengths=True, log=print):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    history = []
    for ep in range(epochs):
        model.train()
        total, n, correct, seen = 0.0, 0, 0, 0
        for ids, lengths, labels in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = select_class_logits(model(ids.to(device)),
                                         lengths if use_lengths else None)
            loss = F.cross_entropy(logits.float(), labels.to(device))
            loss.backward()
            opt.step()
            total += loss.item(); n += 1
            correct += (logits.argmax(-1).cpu() == labels).sum().item()
            seen += len(labels)
        va_acc = accuracy_loader(model, val_loader, device, use_lengths=use_lengths)
        history.append(dict(epoch=ep + 1, loss=total / n,
                            train_acc=correct / seen, val_acc=va_acc))
    return history
```

Two details are worth noticing. First, the training accuracy is accumulated from forward passes we
already pay for, instead of a second pass over the training set. That pass would cost as much as a
full extra epoch of forward compute. The number is slightly stale (it mixes predictions from the
start and end of the epoch), which is fine for monitoring. Second, `loss.item()` forces a
host--device synchronisation on every step. On a GPU, it is better to accumulate the loss in a
device tensor and read it once per logging interval ([@ch:machine]).

[@fig:classify:curves] shows a training run on the lab's synthetic spam task: a 2-layer,
64-dimensional GPT trained from scratch, measured on CPU in this environment. The shape is what you
will see with real GPT-2 on real SMS data. The loss falls quickly, training accuracy saturates near
100%, and validation accuracy plateaus a few points lower. With a pretrained GPT-2 on the SMS data,
expect training and validation accuracies in the high 90s and test accuracy in the mid 90s.

![Loss and accuracy per epoch on the synthetic spam task of Lab 24 (600 messages, 40% spam before
balancing, tiny GPT, AdamW at $2\times10^{-3}$, length-bucketed batches of 16). Measured on CPU in
this environment.](figures/ch41_classification-8.svg){#fig:classify:curves}

##### Reading the curves.

If validation accuracy lags training accuracy by more than a few points and the gap keeps widening,
the model is memorising. The usual fixes, in rough order of cost, are fewer epochs (early stopping
on validation accuracy), higher weight decay, dropout, more data, and training fewer layers. If both
curves stall near 50%, the usual causes are (i) the head is reading padding positions while the
padding amount is correlated with the label, (ii) the learning rate is too low for a freshly
initialised head, or (iii) the trainable flags are wrong. Print
`sum(p.numel() for p in model.parameters() if p.requires_grad)` to check.

::: napkinmath
[How long should this take on a GPU?]{.box-title}

Each training message costs about $2N + 4N_{\text{tr}}$ FLOPs per token
([@sec:classify:head]), where $N$ counts only the non-embedding weights that do work. For
GPT-2 small, $N\approx 85$M and $N_{\text{tr}}\approx 7.1$M, giving $\approx 1.98\times10^8$ FLOPs
per token. Five epochs over 1,045 messages padded to 120 tokens is $6.3\times10^5$ token-positions,
or about $1.2\times10^{14}$ FLOPs. At a pessimistic 20% of an A100's 312 , that is two seconds of
arithmetic. In practice a run takes much longer: kernel launches for small batches, Python overhead
and evaluation passes dominate. This chapter's workload is *latency-bound*, not FLOP-bound. Larger
batches, `torch.compile` and CUDA graphs ([@ch:inference]) are the relevant levers.
:::

## Using the LLM as a spam classifier {#sec:classify:use}

To classify a new message we tokenise it, truncate it to the model's context length (or to the
training $T_{\max}$, if we trained with global padding and want identical conditions), run one
forward pass, and take the $\operatorname*{arg\,max}$ of the selected logits.

``` {#lst:classify:infer .python caption="Inference with the fine-tuned classifier."}
@torch.no_grad()
def classify(model, tokenizer, text, device="cpu", max_length=None,
             labels=("not spam", "spam")):
    ctx = model.cfg["context_length"]
    ids = tokenizer.encode(text)[: min(max_length or ctx, ctx)]
    logits = model(torch.tensor([ids], device=device))[:, -1, :]   # no padding
    return labels[int(logits.argmax(-1))]

classify(model, tok, "You are a winner! Reply now to claim your free prize.")
# -> 'spam'
classify(model, tok, "Are we still on for dinner tonight?")
# -> 'not spam'

torch.save(model.state_dict(), "spam_classifier.pth")   # 2-way head included
```

To reload, rebuild the architecture *with* the 2-way head before calling `load_state_dict`.
Otherwise the stored `out_head.weight` of shape $(2, 768)$ will not fit a $(50257, 768)$ parameter.
Storing the full model costs about $0.5\,\mathrm{GB}$ in fp32. If many tasks share one frozen trunk,
store only the trainable parameters per task ($\approx$`<!-- -->`{=html}28  here) together with a
pointer to the base checkpoint, which is the same idea as LoRA adapters in [@ch:instruct].

::: systemslens
[Serving a classifier is a pure-prefill workload]{.box-title}

Generative serving alternates a compute-bound *prefill* with a memory-bound *decode* loop
([@ch:inference]). A classifier has only the prefill. Every request is a single forward pass
over its own tokens, the output is $C$ numbers, and the KV cache is discarded immediately.
Consequences:

- Throughput is governed by FLOPs, not memory bandwidth, so large batches and tensor-core-friendly
  shapes pay off directly.

- The vocabulary head disappears. In GPT-2 small the $768\times50257$ projection is 38.6M
  parameters, nearly half the size of all twelve blocks. Removing it saves those FLOPs for every
  request.

- Batching policy is all about padding. Bucket by length, or use a varlen attention kernel that
  accepts a packed batch with per-sequence offsets ([@sec:instruct:packing]).

- Latency is easy to bound: there is no generation length to predict, so SLOs can be met with simple
  admission control.
:::

::: napkinmath
[Classifier throughput on one H100]{.box-title}

Forward FLOPs per token for GPT-2 small without the vocabulary head:
$2\times 85\text{M} = 1.7\times10^8$ (attention terms are negligible at 30 tokens). At 50% of the
H100's 989 : $4.9\times10^{14} / 1.7\times10^8 \approx 2.9$ million tokens per second. At about 30
tokens per SMS, that is roughly 97,000 messages per second from one GPU, if batches are large and
padding-free. With random batching at 60% padding waste ([@fig:classify:waste]), the same
GPU delivers about 40% of that.
:::

::: deepdive
[When not to fine-tune a 124M-parameter decoder]{.box-title}

For a binary task with a few thousand examples, a linear probe on frozen sentence embeddings, or a
small encoder (DistilBERT/ModernBERT-class), is often as accurate and 10--100$\times$ cheaper to
serve. A decoder LLM classifier makes most sense when (i) you already serve that trunk and can share
it, (ii) the inputs are long or need world knowledge, or (iii) you will later turn the same model
into an instruction follower. An LLM can also act as a *zero-shot* classifier by scoring the
log-probabilities of label words. [@ch:refine] develops this kind of probability-based
scoring.
:::

## From one classifier to a fleet {#sec:classify:fleet}

Getting one spam classifier to work is only the beginning. In production a single classifier is
rarely the whole story: you need to decide where to put the decision threshold, you may run several
classifiers on every request, and you will want an expensive model to see only the inputs the cheap
one is unsure about. This section collects the systems patterns that follow from that.

### Metrics, thresholds and calibration

The $\operatorname*{arg\,max}$ rule implicitly thresholds $p(\text{spam}\mid x)$ at $0.5$. The costs
of the two error types are rarely equal. Hiding a real message from a user (a false positive) is
usually worse than letting a spam through (a false negative). With
$\mathrm{TP},\mathrm{FP},\mathrm{FN},\mathrm{TN}$ counted at a threshold $\tau$, $$\begin{equation}
\text{precision}(\tau) = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\qquad
\text{recall}(\tau) = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\qquad
F_1 = \frac{2\,\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}.
\label{eq:classify:prf}
\end{equation}$$ Raising $\tau$ trades recall for precision. The operating point is chosen on the
*validation* split, for example the smallest $\tau$ whose validation precision is at least 99%, and
then reported once on the test split.

![The confusion matrix with the predicted class as columns. The two error cells cost different
amounts, so the threshold $\tau$ on $p(\text{spam}\mid x)$ is a product decision, not a modelling
default.](figures/ch41_classification-9.svg){#fig:classify:confusion}

A threshold only means something if the probabilities are *calibrated*: among messages scored $0.9$,
about 90% should really be spam. Fine-tuned networks are often over-confident. A standard check is
the expected calibration error. Bin the $n$ test predictions by confidence into bins $B_1,\dots,B_M$
and compute $$\begin{equation}
\mathrm{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{n}\,\bigl|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\bigr|.
\end{equation}$$ *Temperature scaling* (Guo et al., 2017) is the cheapest fix. It divides the logits
by a single scalar $T$ fitted on the validation set to minimise the negative log-likelihood. Because
it does not change the $\operatorname*{arg\,max}$, accuracy is untouched. This is the same
temperature that controls sampling diversity in [@ch:pretrain; @ch:scaling], used here for
the opposite purpose of making probabilities honest.

``` {#lst:classify:temp .python caption="Fitting a calibration temperature on cached validation logits."}
def fit_temperature(val_logits, val_labels, steps=200, lr=0.01):
    # val_logits: (n, C), collected once under torch.no_grad()
    log_t = torch.zeros(1, requires_grad=True)          # T = exp(log_t) > 0
    opt = torch.optim.Adam([log_t], lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(val_logits / log_t.exp(), val_labels)
        opt.zero_grad(); loss.backward(); opt.step()
    return log_t.exp().item()
```

Note the systems shape of this step. We run the expensive model *once* over the validation set,
cache an $n\times C$ matrix of logits, and fit on that. The fit takes milliseconds. The same pattern
of caching model outputs and tuning cheap post-processing on top appears again with verifiers and
scorers in [@ch:eval; @ch:refine].

### Cascades: spend big-model compute only on hard inputs

Suppose a large instruction-tuned model ([@ch:instruct]) classifies more accurately than our
fine-tuned GPT-2, but costs $R$ times more per token. A *cascade* runs the small classifier on
everything and forwards a request to the large model only when the small model's calibrated
confidence $\max_c p(c\mid x)$ is below a threshold. If a fraction $e$ of traffic is escalated, the
expected cost per request in units of the small model is $$\begin{equation}
\text{cost}(e) = 1 + e\,R,\qquad \frac{\text{cost}(e)}{\text{cost of large only}} = \frac{1+eR}{R}.
\label{eq:classify:cascade}
\end{equation}$$ Calibration matters here: with an over-confident small model, the escalated set
misses exactly the inputs it gets wrong.

::::: {#fig:classify:cascade .figure}
::: minipage
![](figures/ch41_classification-10.svg)
:::

::: minipage
![](figures/ch41_classification-11.svg)
:::

[Left: a two-stage classifier cascade. Right: the relative cost from
[@eq:classify:cascade]. When the large model is $R\approx94\times$ more expensive per token
(an 8B-parameter model versus GPT-2 small's 85M non-embedding parameters, prefill only), escalating
5% of traffic costs about 6% of the large-only budget.]{.figure-caption}
:::::

::: napkinmath
[Pricing a cascade]{.box-title}

Prefill FLOPs per token scale with non-embedding parameters: $2\times 85\text{M}$ for GPT-2 small
versus about $2\times 8\text{B}$ for an 8B model, so $R\approx94$. If the small model is confident
on 95% of messages, $\text{cost}=1+0.05\times94=5.7$ small-model units, against 94 for the large
model alone: a $16\times$ saving. And if the large model must *generate* a verdict instead of
scoring label tokens in a single prefill, $R$ grows further, because decoding is memory-bound
([@ch:inference]).
:::

### One trunk, many heads

Platforms often run several classifiers over every message: spam, abuse, language, intent, PII. If
they all fine-tune only the top of the same base model, the frozen bottom of the network computes
*identical* activations for each of them. Sharing that computation is pure profit.

![Serving several fine-tuned classifiers that share a frozen trunk. Each task owns only its top
block and head (about 14  in bf16 for GPT-2
small).](figures/ch41_classification-12.svg){#fig:classify:multihead}

With $K$ tasks that each fine-tune only the last of $L$ blocks, separate models evaluate $KL$ blocks
per token, while a shared trunk evaluates $(L-1)+K$. For $L=12$ and $K=8$ that is $96$ versus $19$,
a $5\times$ saving in FLOPs *and* in weight memory traffic. If the tasks only train heads
(`trainable_blocks=0`), the saving approaches $K\times$. LoRA gives the same economics for adapters
spread through the whole network, using the batched multi-adapter kernels of
[@sec:instruct:multilora].

### Classifying with a modern decoder

Nothing in this chapter is specific to GPT-2. The Qwen3 model of [@ch:modern] uses the same
attribute names (`trf_blocks`, `final_norm`, `out_head`), so `add_classification_head` applies
unchanged. Three differences deserve care:

- **Tied embeddings.** In Qwen3-0.6B, `out_head.weight` *is* `tok_emb.weight`. Replacing `out_head`
  breaks the tie, which is what we want. The input embedding stays frozen and the new head is
  independent. But parameter counts and checkpoint sizes change, and code that re-ties weights on
  load must be skipped.

- **A real padding token.** Qwen's tokenizer has distinct `<|endoftext|>` and `<|im_end|>` tokens,
  so padding never collides with an end-of-turn marker that the model treats as meaningful.

- **RoPE makes left padding exact.** Attention scores depend only on relative positions, so left
  padding plus a key mask gives the same last-token logits as an unpadded forward pass, up to
  floating-point summation order. The core test suite checks this property for batched generation.

``` {#lst:classify:qwen .python caption="The same recipe on a Qwen3-shaped model (tiny config shown)."}
from llmsys.qwen3 import Qwen3Model, QWEN3_CONFIG_TINY
from llmsys.finetune import add_classification_head, pad_batch

model = add_classification_head(Qwen3Model(QWEN3_CONFIG_TINY), num_classes=2)
ids, lengths = pad_batch([[5, 6, 7, 8], [9, 10]], pad_id=0, side="left")
pad_mask = ids != 0                              # True for real tokens
logits = model(ids, pad_mask=pad_mask)[:, -1]   # (2, 2): last position is real
```

## Summary {#sec:classify:summary}

- Classification fine-tuning replaces the vocabulary projection with a $C$-way head and trains on
  labelled examples. It is cheaper and easier to evaluate than instruction fine-tuning, but the
  resulting model does only that one task.

- Data preparation matters: balance or reweight classes, deduplicate before splitting, and split
  with a fixed seed into train/validation/test (70/10/20 here).

- Variable-length inputs must be padded. Padding to the global maximum wastes most of the compute;
  dynamic padding plus length bucketing removes nearly all of it.

- Under a causal mask only the last position has seen the whole input, so the head reads that
  position. With right padding, gather the last *real* token, or use left padding with a key mask.

- Freezing the trunk removes optimizer state and, when the frozen layers are at the bottom, also
  their backward pass and stored activations. It even allows caching trunk features across epochs.

- Serving a classifier is pure prefill: compute-bound, no decode loop, no vocabulary head.
  Padding-aware batching is the main lever on throughput.

- Thresholds should be chosen on calibrated probabilities. Calibrated small classifiers enable
  cascades and shared-trunk multi-task serving, which cut cost by an order of magnitude.

## Exercises {#sec:classify:exercises}

::: exercise
[Global versus dynamic padding]{.box-title}

Train the classifier twice: once with every split padded to the longest training message (120
tokens), and once with per-batch dynamic padding and last-real-token pooling. Compare accuracy and
wall-clock time per epoch. Then pad everything to 1,024 tokens and explain what happens to time and
to accuracy when the head reads position $-1$.
:::

::: {#ex:classify:layers .exercise}
\[How many layers to unfreeze\] Sweep `trainable_blocks`$\in\{0,1,2,4,12\}$ (0 = head and final norm
only). For each setting, report test accuracy, trainable parameter count, peak memory, and time per
step. Predict the time per step from the $2N + 4N_{\text{tr}}$ model before measuring, and explain
any discrepancy.
:::

::: exercise
[Trunk caching]{.box-title}

With `trainable_blocks=1`, precompute the outputs of blocks 1--11 for the whole training set and
train block 12 and the head on the cached tensors. Verify that the loss trajectory matches uncached
training (dropout off). How large is the cache, and how much faster is each epoch?
:::

::: exercise
[Class weights instead of undersampling]{.box-title}

Use all 5,574 messages with the original 87/13 class ratio. Compare (a) plain cross-entropy, (b)
cross-entropy with class weights inversely proportional to class frequency, and (c) undersampling.
Report accuracy, precision, recall and F1 for the spam class. Which metric would you put on a
dashboard, and why?
:::

::: exercise
[Padding invariance]{.box-title}

Prove that with right padding, causal attention, and last-real-token pooling, a message's logits do
not depend on how much padding follows it. Then show with a counterexample that reading position
$-1$ does depend on padding amount. (Hint: what does the padding position attend to?)
:::

::: exercise
[Left padding and absolute positions]{.box-title}

Classify a batch with left padding and a key-padding mask using GPT-2's learned absolute position
embeddings. Measure how much the logits differ from the unpadded single-message logits, and explain
why. Would the result change for a RoPE model?
:::

::: exercise
[Token-budget batching]{.box-title}

Implement a sampler that yields batches with at most $K$ tokens after padding (e.g. $K=2048$).
Compare the number of batches, padding waste, and step-time variance against fixed-size bucketed
batches.
:::

::: exercise
[Zero-shot baseline]{.box-title}

Without any fine-tuning, classify messages by comparing $\log p(\text{`` spam''})$ and
$\log p(\text{`` ham''})$ after a prompt such as "Message: ...\\nCategory:". Report accuracy on the
test split. How does the gap to the fine-tuned model change with model size (GPT-2 small, medium,
large)?
:::

::: exercise
[Serving capacity]{.box-title}

A service must classify 20,000 messages per second (mean 35 tokens, p99 160 tokens) with p99 latency
under 50 . Using the napkin model of [@sec:classify:use], estimate how many L4 GPUs are
needed with (a) random batching and (b) bucketing. Which assumption in your estimate is most likely
to be wrong?
:::

::: exercise
[Deduplication]{.box-title}

Count exact duplicates in the SMS collection, and near-duplicates after lowercasing and replacing
digit runs with a placeholder. Re-split after deduplication and compare test accuracy to the
original split. What does the difference tell you?
:::

::: exercise
[Calibration]{.box-title}

Collect validation logits once, fit a temperature with [@lst:classify:temp], and report ECE
(15 bins) on the test split before and after. Show analytically that temperature scaling cannot
change accuracy.
:::

::: exercise
[Cascade design]{.box-title}

Using your calibrated classifier, plot escalation rate $e(\tau)$ and cascade accuracy against
$\tau$, assuming the large model is always right. Find the cheapest $\tau$ that reaches 99.5%
accuracy and compute its cost with [@eq:classify:cascade].
:::

::: {#ex:classify:probe .exercise}
\[Layer-wise probes\] Use [@lst:classify:probe] to collect last-real-token features after
every block of pretrained GPT-2 small on the balanced SMS splits, fit a logistic-regression probe
per layer, and plot validation accuracy against depth. At which depth would you cut the network for
serving, and what is the predicted FLOP saving?
:::

## Lab 28: Classification fine-tuning and padding-aware batching {#sec:classify:lab}

::: labbox
[Classification fine-tuning and padding-aware batching]{.box-title}

**Code:** `code/labs/lab41_classify`

**Goals**

Build the full classification fine-tuning pipeline: balancing, splitting, dynamic padding, the head
swap with selective freezing, padding-invariant pooling, and the training loop. Then measure how
much compute padding wastes and how much bucketing recovers.

**Setup**

`cd code; LAB_IMPL=starter python -m pytest -q labs/lab41_classify`. The tests run on CPU in under a
minute using a tiny GPT and a synthetic spam task (`make_synthetic_spam`: "spam" messages contain
marker tokens, and lengths are log-normal). `run.py` repeats the experiment with GPT-2 small on the
real SMS Spam Collection on your own machine.

**Tasks**

1.  `balance_by_undersampling`, `split_indices`: seeded, exact sizes, every index exactly once.

2.  `pad_batch`: left/right padding, optional truncation, returns lengths.

3.  `add_classification_head`: freeze everything, swap the head, unfreeze the last $k$ blocks and
    the final norm.

4.  `select_class_logits` and `accuracy_loader`: last-real-token pooling. The test checks that
    predictions do not change when more padding follows a message.

5.  `length_bucketed_batches` and `padding_stats`: sort-within-mega-batch bucketing and a waste
    meter. The test requires bucketing to cut waste by at least $3\times$.

6.  `train_classifier`: AdamW over trainable parameters, training accuracy accumulated for free,
    validation accuracy per epoch. The test requires $>80\%$ test accuracy on the synthetic task.

**Measurements**

- Reproduce [@fig:classify:waste] for your own length distribution (synthetic and, if you
  have it, the real SMS data with the GPT-2 tokenizer).

- With `run.py` on a GPU: time per epoch and peak memory for `trainable_blocks`$\in\{1,12\}$, with
  and without `--bucketed`. Expect bucketing to cut epoch time roughly in proportion to the waste
  reduction once batches are large enough to be compute-bound. Expect `trainable_blocks=1` to be
  roughly $2$--$3\times$ faster than full fine-tuning per step.

- Accuracy before and after fine-tuning: near 50% before, mid-to-high 90s after on real SMS data.

**Deliverables**

Passing tests; a short table (accuracy, time/epoch, peak memory, trainable params) for the
configurations above; one paragraph explaining the measured speed-up from freezing in terms of
forward and backward FLOPs.

**Stretch goals**

- Implement trunk caching and token-budget batching (Exercises 16.3 and 9.7).

- Serve the classifier behind a minimal HTTP endpoint with a micro-batching queue: collect requests
  for up to 5  or 64 messages, bucket by length, and run one forward pass. Plot throughput against
  p99 latency as the batching window varies.
:::
