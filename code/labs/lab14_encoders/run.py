"""Lab 14: train a tiny encoder with MLM, turn it into a sentence embedder, and measure throughput.

    python run.py --impl solution                     # MLM, then InfoNCE, then retrieval accuracy
    python run.py --impl solution --gpu               # throughput vs batch size, sorted batching
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, CODE)
from llmsys.encoder import (EncoderModel as RefEncoderModel, MLMHead,  # noqa: E402
                            encoder_activation_bytes_per_sequence, encoder_flops_per_sequence,
                            batch_for_compute_bound, length_sorted_batches, padding_waste)

PAD, CLS, SEP, MASK, UNK = 0, 1, 2, 3, 4
SPECIALS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
A100_FLOPS, A100_HBM = 312e12, 1.555e12

# Each inner tuple is one concept. A paraphrase picks other words from the same concepts and may
# change the voice, so plain word overlap between a sentence and its paraphrase is close to zero.
SUBJECTS = [("engineer", "developer", "programmer"), ("student", "learner", "pupil"),
            ("manager", "supervisor", "boss"), ("doctor", "physician", "medic"),
            ("teacher", "instructor", "tutor"), ("driver", "chauffeur", "motorist"),
            ("farmer", "grower", "rancher"), ("lawyer", "attorney", "counsel"),
            ("cook", "chef", "baker"), ("guard", "sentry", "watchman")]
VERBS = [("fixed", "repaired", "mended"), ("bought", "purchased", "acquired"),
         ("sold", "traded", "auctioned"), ("found", "located", "discovered"),
         ("moved", "carried", "transported"), ("cleaned", "washed", "scrubbed"),
         ("broke", "smashed", "shattered"), ("checked", "inspected", "examined")]
OBJECTS = [("car", "automobile", "vehicle"), ("server", "machine", "computer"),
           ("report", "document", "paper"), ("house", "home", "residence"),
           ("boat", "ship", "vessel"), ("phone", "handset", "telephone"),
           ("shoes", "boots", "sneakers"), ("chair", "seat", "stool"),
           ("bag", "sack", "satchel"), ("road", "street", "highway")]
SYNTHETIC_WORDS = sorted({w for group in SUBJECTS + VERBS + OBJECTS for w in group}
                         | {"the", "was", "by", "yesterday", "today", "someone", "saw", "that"})


def load_impl(name: str):
    spec = importlib.util.spec_from_file_location(f"lab14_{name}", os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- data
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+|[0-9]+|[.,]", text.lower())


def build_vocab(words: list[str]) -> dict[str, int]:
    vocab = {w: i for i, w in enumerate(SPECIALS)}
    for w in sorted(set(words) | set(SYNTHETIC_WORDS)):
        vocab.setdefault(w, len(vocab))
    return vocab


def encode(words: list[str], vocab: dict[str, int]) -> list[int]:
    return [CLS] + [vocab.get(w, UNK) for w in words] + [SEP]


def collate(seqs: list[list[int]], device) -> tuple[torch.Tensor, torch.Tensor]:
    tmax = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), tmax), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    return ids.to(device), (ids != PAD).to(device)


def surface(meaning: tuple[int, int, int], rng: random.Random) -> list[str]:
    """One random wording of (subject concept, verb concept, object concept)."""
    s, v, o = (rng.choice(g[i]) for g, i in zip((SUBJECTS, VERBS, OBJECTS), meaning))
    return rng.choice([
        ["the", s, v, "the", o],
        ["the", o, "was", v, "by", "the", s],
        ["yesterday", "the", s, v, "the", o],
        ["someone", "saw", "that", "the", s, v, "the", o],
        ["today", "the", o, "was", v, "by", "the", s],
    ])


def split_meanings(seed: int, test_frac: float = 0.2):
    """Held-out meanings are never seen in training, though all their words are."""
    all_m = [(s, v, o) for s in range(len(SUBJECTS)) for v in range(len(VERBS))
             for o in range(len(OBJECTS))]
    random.Random(seed).shuffle(all_m)
    n_test = int(test_frac * len(all_m))
    return all_m[n_test:], all_m[:n_test]


# --------------------------------------------------------------------------- stages
def train_mlm(lab, model, head, windows, steps, batch, lr, device, generator):
    params = list(model.parameters()) + [p for n, p in head.named_parameters()
                                         if n != "decoder_weight"]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    first, ema = None, None
    model.train()
    for step in range(steps):
        idx = torch.randint(len(windows), (batch,), generator=generator).tolist()
        ids, pad = collate([windows[i] for i in idx], device)
        gen = torch.Generator().manual_seed(int(torch.randint(1 << 30, (1,), generator=generator)))
        inputs, labels = lab.mlm_corrupt(ids.cpu(), MASK, model.cfg["vocab_size"],
                                         (PAD, CLS, SEP), generator=gen)
        _, loss = head(model(inputs.to(device), pad), labels.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
    return first, ema


def embed(lab, model, seqs, how, device, batch: int = 128) -> torch.Tensor:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            ids, pad = collate(seqs[i: i + batch], device)
            out.append(F.normalize(lab.pool(model(ids, pad), pad, how), dim=-1))
    return torch.cat(out)


def retrieval_accuracy(lab, model, meanings, vocab, how, device, seed=123, gallery=200):
    """Top-1 accuracy: each query is a wording of a held-out meaning, each document another."""
    rng = random.Random(seed)
    meanings = rng.sample(meanings, min(gallery, len(meanings)))
    q = embed(lab, model, [encode(surface(m, rng), vocab) for m in meanings], how, device)
    d = embed(lab, model, [encode(surface(m, rng), vocab) for m in meanings], how, device)
    top1 = (q @ d.T).argmax(1).cpu()
    return (top1 == torch.arange(len(meanings))).float().mean().item(), len(meanings)


def train_contrastive(lab, model, meanings, vocab, how, steps, batch, lr, temperature, device, seed):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    first, ema = None, None
    model.train()
    for _ in range(steps):
        picks = rng.sample(meanings, batch)                  # distinct, so no false in-batch negative
        q_ids, q_pad = collate([encode(surface(m, rng), vocab) for m in picks], device)
        d_ids, d_pad = collate([encode(surface(m, rng), vocab) for m in picks], device)
        q = F.normalize(lab.pool(model(q_ids, q_pad), q_pad, how), dim=-1)
        d = F.normalize(lab.pool(model(d_ids, d_pad), d_pad, how), dim=-1)
        loss, _ = lab.info_nce_loss(q, d, temperature)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
    return first, ema


def text_stage(args, lab) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    words = tokenize(open(args.data, encoding="utf-8").read())
    vocab = build_vocab(words)
    cfg = dict(vocab_size=len(vocab), context_length=args.window + 2, emb_dim=args.d, n_heads=4,
               n_layers=args.layers, drop_rate=0.1, qkv_bias=True)
    model = lab.EncoderModel(cfg)
    model.apply(RefEncoderModel._init_weights)    # small init: the MLM head is tied to tok_emb
    model.to(device)
    head = MLMHead(args.d, model.tok_emb).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocab {len(vocab)}, {len(words)} running words, {n_params / 1e3:.0f}k parameters, "
          f"device {device}")
    train_m, test_m = split_meanings(args.seed)
    # MLM reads both the prose and wordings of the *training* meanings, so that the synonyms
    # are not unseen words when the contrastive stage starts.
    rng = random.Random(args.seed)
    windows = [encode(words[i: i + args.window], vocab)
               for i in range(0, len(words) - args.window, args.window // 2)]
    windows += [encode(surface(rng.choice(train_m), rng), vocab) for _ in range(len(windows))]
    results = dict(vocab=len(vocab), params=n_params, device=str(device),
                   held_out_meanings=len(test_m), windows=len(windows))

    acc, gallery = retrieval_accuracy(lab, model, test_m, vocab, args.pooling, device)
    results.update(acc_random_init=acc, gallery=gallery)
    print(f"random init      : nearest-neighbour accuracy {acc:.3f}  (chance {1 / gallery:.3f})")

    t0 = time.perf_counter()
    gen = torch.Generator().manual_seed(args.seed)
    first, last = train_mlm(lab, model, head, windows, args.mlm_steps, args.batch, args.mlm_lr,
                            device, gen)
    acc, _ = retrieval_accuracy(lab, model, test_m, vocab, args.pooling, device)
    results.update(mlm_first=first, mlm_last=last, acc_after_mlm=acc,
                   mlm_seconds=time.perf_counter() - t0)
    print(f"after MLM        : loss {first:.2f} -> {last:.2f}, accuracy {acc:.3f} "
          f"({results['mlm_seconds']:.1f} s)")

    t0 = time.perf_counter()
    first, last = train_contrastive(lab, model, train_m, vocab, args.pooling, args.steps,
                                    args.batch, args.lr, args.temperature, device, args.seed)
    acc, _ = retrieval_accuracy(lab, model, test_m, vocab, args.pooling, device)
    results.update(info_nce_first=first, info_nce_last=last, acc_after_contrastive=acc,
                   contrastive_seconds=time.perf_counter() - t0)
    print(f"after contrastive: loss {first:.2f} -> {last:.2f}, accuracy {acc:.3f} "
          f"({results['contrastive_seconds']:.1f} s)")
    return results


# --------------------------------------------------------------------------- GPU measurements
def hardware_string() -> str:
    driver = "unknown"
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True).stdout.split()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        pass
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
    return f"{gpu}; torch {torch.__version__}; CUDA {torch.version.cuda}; driver {driver}"


def timed(fn, warmup: int = 5, reps: int = 20) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    q = statistics.quantiles(samples, n=10)
    return dict(median=statistics.median(samples), p10=q[0], p90=q[-1])


def make_model(cfg: dict, device) -> RefEncoderModel:
    return RefEncoderModel(dict(cfg, use_sdpa=True)).to(device=device, dtype=torch.bfloat16).eval()


def nonembed_params(model: RefEncoderModel) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if "_emb" not in n)


def batch_sweep(cfg: dict, T: int, batches: list[int], device) -> dict:
    model = make_model(cfg, device)
    n_all = sum(p.numel() for p in model.parameters())
    ffn_mult = cfg.get("hidden_dim", 4 * cfg["emb_dim"]) / cfg["emb_dim"]
    flops = encoder_flops_per_sequence(nonembed_params(model), T, cfg["n_layers"], cfg["emb_dim"])
    act = encoder_activation_bytes_per_sequence(T, cfg["n_layers"], cfg["emb_dim"], ffn_mult)
    weight_bytes = 2 * nonembed_params(model)
    rows = []
    for b in batches:
        ids = torch.randint(5, cfg["vocab_size"], (b, T), device=device)
        def step():
            with torch.no_grad():
                model(ids)
        try:
            t = timed(step)
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {b}: out of memory, sweep stops here")
            torch.cuda.empty_cache()
            break
        rows.append(dict(batch=b, seq_per_s=b / t["median"], seq_per_s_p10=b / t["p90"],
                         seq_per_s_p90=b / t["p10"], ms=1e3 * t["median"],
                         peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9))
        print(f"  batch {b:5d}: {rows[-1]['seq_per_s']:9.1f} seq/s "
              f"(p10 {rows[-1]['seq_per_s_p10']:.1f}, p90 {rows[-1]['seq_per_s_p90']:.1f}), "
              f"{rows[-1]['ms']:8.2f} ms/batch")
    del model
    torch.cuda.empty_cache()
    return dict(params=n_all, params_nonembed=weight_bytes // 2, T=T, flops_per_seq=flops,
                activation_bytes_per_seq=act, roof_seq_per_s=A100_FLOPS / flops,
                compute_bound_batch=batch_for_compute_bound(weight_bytes, flops, A100_FLOPS,
                                                            A100_HBM, act), rows=rows)


def sampled_lengths(n: int, tmax: int, seed: int) -> list[int]:
    """A skewed length distribution: most texts are short and a few are long (log-normal)."""
    rng = random.Random(seed)
    return [min(tmax, max(4, int(rng.lognormvariate(4.0, 0.8)))) for _ in range(n)]


def batching_sweep(cfg: dict, n: int, batch: int, device, seed: int) -> dict:
    """Same sequences, three schedules: fixed-size random, fixed-size sorted, token-budget sorted."""
    model = make_model(cfg, device)
    lengths = sampled_lengths(n, cfg["context_length"], seed)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    random_batches = [order[i: i + batch] for i in range(0, n, batch)]
    by_len = sorted(order, key=lambda i: lengths[i])
    sorted_batches = [by_len[i: i + batch] for i in range(0, n, batch)]
    budget = batch * int(statistics.mean(lengths))
    budget_batches = length_sorted_batches(lengths, budget)
    data = [torch.randint(5, cfg["vocab_size"], (t,)) for t in lengths]
    results = dict(n=n, batch=batch, mean_len=statistics.mean(lengths), max_len=max(lengths),
                   token_budget=budget, schedules={})

    def run(batches):
        for b in batches:
            tmax = max(lengths[i] for i in b)
            ids = torch.zeros(len(b), tmax, dtype=torch.long)
            pad = torch.zeros(len(b), tmax, dtype=torch.bool)
            for row, i in enumerate(b):
                ids[row, : lengths[i]] = data[i]
                pad[row, : lengths[i]] = True
            with torch.no_grad():
                model(ids.to(device), pad.to(device))

    for name, batches in (("random", random_batches), ("sorted", sorted_batches),
                          ("token budget", budget_batches)):
        t = timed(lambda: run(batches), warmup=1, reps=5 if n > 1000 else 20)
        results["schedules"][name] = dict(
            n_batches=len(batches), waste=padding_waste(lengths, batches),
            seconds=t["median"], seq_per_s=n / t["median"], real_tok_per_s=sum(lengths) / t["median"])
        r = results["schedules"][name]
        print(f"  {name:13s}: {r['n_batches']:4d} batches, padding {100 * r['waste']:5.1f}%, "
              f"{r['seq_per_s']:8.1f} seq/s, {r['real_tok_per_s'] / 1e3:8.1f}k real tokens/s")
    del model
    torch.cuda.empty_cache()
    return results


def gpu_stage(args) -> dict:
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    results = dict(hardware=hardware_string())
    print(results["hardware"])
    bert = dict(vocab_size=30522, context_length=512, emb_dim=768, n_heads=12, n_layers=12,
                drop_rate=0.0, qkv_bias=True)
    batches = [int(b) for b in args.batches.split(",")]
    print("BERT-base shape, T = 128, bf16:")
    results["bert_base"] = batch_sweep(bert, 128, batches, device)
    print("BERT-base shape, schedules on a skewed length distribution:")
    results["batching"] = batching_sweep(bert, args.n_seqs, 32, device, args.seed)
    if not args.no_7b:
        print("7B decoder shape, T = 128, bf16:")
        big = dict(vocab_size=32000, context_length=4096, emb_dim=4096, n_heads=32, n_layers=32,
                   drop_rate=0.0, qkv_bias=False, hidden_dim=18048)
        big_batches = [b for b in batches if b <= 128]
        results["decoder_7b"] = batch_sweep(big, 128, big_batches, device)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impl", default="starter", choices=["starter", "solution"])
    ap.add_argument("--data", default=os.path.join(CODE, "data", "encoder_sample.txt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--mlm-steps", type=int, default=800)
    ap.add_argument("--mlm-lr", type=float, default=2e-3)
    ap.add_argument("--steps", type=int, default=600, help="contrastive steps")
    ap.add_argument("--lr", type=float, default=5e-4, help="contrastive learning rate")
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--gpu", action="store_true", help="throughput measurements only")
    ap.add_argument("--batches", default="1,2,4,8,16,32,64,128,256,512")
    ap.add_argument("--n-seqs", type=int, default=2000)
    ap.add_argument("--no-7b", action="store_true", help="skip the 14 GB decoder-shaped model")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    results = gpu_stage(args) if args.gpu else text_stage(args, load_impl(args.impl))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
