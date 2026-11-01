"""Distillation for reasoning models (Chapter 20).

Hard distillation  = supervised fine-tuning on teacher-generated text (answer-only CE).
Soft distillation  = match the teacher's per-token distribution (KL), optionally from a
                     stored top-k slice of the teacher logits.

Data path: raw teacher records -> formatted target -> token ids + prompt_len ->
length filter -> train/val split -> (per-example or padded/packed) training.
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

IGNORE = -100

# --------------------------------------------------------------------------- formatting
PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason carefully, then give the final "
    "result on its own line in the form \\boxed{{ANSWER}}.\n\n"
    "Problem:\n{problem}\n\nSolution:"
)


def render_math_prompt(problem: str) -> str:
    return PROMPT_TEMPLATE.format(problem=problem.strip())


def format_distilled_answer(record: Dict, use_think: bool = True) -> str:
    """Turn a teacher record into the student's target text.

    ``record`` has ``reasoning`` (the teacher's thinking trace) and ``answer`` (its
    final, user-facing response). With ``use_think`` the trace is wrapped in
    <think>...</think> so downstream UIs can hide it.
    """
    answer = str(record.get("answer", "")).strip()
    if not answer:
        raise ValueError("record has an empty 'answer' field")
    reasoning = str(record.get("reasoning", "")).strip()
    if use_think:
        return f"<think>{reasoning}</think>\n\n{answer}"
    return f"{reasoning}\n\n{answer}" if reasoning else answer


def build_examples(records: Iterable[Dict], encode_prompt: Callable[[str], List[int]],
                   encode_answer: Callable[[str], List[int]], eos_id: int,
                   use_think: bool = True,
                   render: Callable[[str], str] = render_math_prompt) -> Tuple[List[Dict], int]:
    """Tokenise every record once. Returns (examples, n_skipped).

    Each example: {"token_ids": prompt + answer + [eos], "prompt_len": len(prompt)}.
    ``encode_prompt`` should apply the chat template; ``encode_answer`` must NOT
    (otherwise a second assistant header would be inserted mid-sequence).
    """
    examples, skipped = [], 0
    for rec in records:
        try:
            p = encode_prompt(render(rec["problem"]))
            a = encode_answer(format_distilled_answer(rec, use_think))
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        ids = p + a + [eos_id]
        if len(ids) < 2 or len(p) == 0:
            skipped += 1
            continue
        examples.append({"token_ids": ids, "prompt_len": len(p)})
    return examples, skipped


def length_stats(examples: Sequence[Dict], answer_only: bool = False) -> Dict[str, float]:
    lens = [len(e["token_ids"]) - (e["prompt_len"] if answer_only else 0) for e in examples]
    srt = sorted(lens)
    q = lambda f: srt[min(len(srt) - 1, int(f * len(srt)))]
    return dict(n=len(lens), mean=sum(lens) / len(lens), min=srt[0], max=srt[-1],
                p50=q(0.5), p90=q(0.9), p99=q(0.99),
                argmin=lens.index(srt[0]), argmax=lens.index(srt[-1]))


def filter_by_max_len(examples: Sequence[Dict], max_len: int) -> List[Dict]:
    return [e for e in examples if len(e["token_ids"]) <= max_len]


def split_train_val(examples: Sequence[Dict], val_size: int, seed: int = 123):
    ex = list(examples)
    random.Random(seed).shuffle(ex)
    return ex[val_size:], ex[:val_size]


# --------------------------------------------------------------------------- losses
def answer_only_loss(model, example: Dict, device="cpu") -> torch.Tensor:
    """Mean CE over the answer tokens of one example (= -mean log p(answer | prompt)).

    Position t of the logits predicts token t+1, so the first answer token
    (index prompt_len) is predicted by logit index prompt_len - 1.
    """
    ids = torch.tensor(example["token_ids"], dtype=torch.long, device=device)
    logits = model(ids[:-1][None])[0]
    start = max(example["prompt_len"] - 1, 0)
    return F.cross_entropy(logits[start:].float(), ids[1:][start:])


def collate_answer_only(batch: Sequence[Dict], pad_id: int, device="cpu"):
    """Right-pad a batch; targets are -100 on prompt and padding positions.

    Returns (inputs (B, T-1), targets (B, T-1), n_answer_tokens).
    """
    T = max(len(e["token_ids"]) for e in batch)
    inp = torch.full((len(batch), T - 1), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), T - 1), IGNORE, dtype=torch.long)
    for i, e in enumerate(batch):
        ids = torch.tensor(e["token_ids"], dtype=torch.long)
        n = len(ids)
        inp[i, : n - 1] = ids[:-1]
        tgt[i, : n - 1] = ids[1:]
        tgt[i, : max(e["prompt_len"] - 1, 0)] = IGNORE
    return inp.to(device), tgt.to(device), int((tgt != IGNORE).sum())


def batch_answer_only_loss(model, batch: Sequence[Dict], pad_id: int, device="cpu",
                           reduction: str = "token") -> torch.Tensor:
    """Batched answer-only loss.

    reduction="token":    sum of token losses / number of answer tokens in the batch
                          (long answers weigh more; matches concatenating the data).
    reduction="sequence": mean over sequences of each sequence's mean token loss
                          (matches averaging ``answer_only_loss`` over examples).
    Right padding is safe for causal LMs: real tokens never attend to later pads.
    """
    inp, tgt, n_tok = collate_answer_only(batch, pad_id, device)
    logits = model(inp).float()
    tok = F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), ignore_index=IGNORE,
                          reduction="none").view_as(tgt)
    valid = (tgt != IGNORE).float()
    if reduction == "token":
        return tok.sum() / valid.sum().clamp(min=1)
    per_seq = tok.sum(1) / valid.sum(1).clamp(min=1)
    return per_seq.mean()


def chunked_cross_entropy(hidden: torch.Tensor, out_head: torch.nn.Module, targets: torch.Tensor,
                          chunk: int = 1024) -> torch.Tensor:
    """CE over a long sequence without materialising all (T, V) logits at once.

    hidden: (N, d) final hidden states, targets: (N,) with -100 for ignored positions.
    Peak logit memory is chunk*V instead of N*V. (Fused kernels such as Liger's
    FusedLinearCrossEntropy also avoid storing the chunk for backward.)
    """
    total = hidden.new_zeros((), dtype=torch.float32)
    count = (targets != IGNORE).sum().clamp(min=1)
    for s in range(0, hidden.shape[0], chunk):
        logits = out_head(hidden[s:s + chunk]).float()
        total = total + F.cross_entropy(logits, targets[s:s + chunk], ignore_index=IGNORE,
                                        reduction="sum")
    return total / count


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            mask: Optional[torch.Tensor] = None, temperature: float = 1.0,
            direction: str = "forward") -> torch.Tensor:
    """Token-level KL distillation loss, averaged over unmasked positions.

    forward: KL(p_T || p_S)  -- mass-covering (classic Hinton KD, needs teacher dist.)
    reverse: KL(p_S || p_T)  -- mode-seeking (MiniLLM / on-policy distillation)
    Multiplied by T^2 so gradient scale is independent of the temperature.
    """
    T = temperature
    ls = F.log_softmax(student_logits.float() / T, dim=-1)
    lt = F.log_softmax(teacher_logits.float() / T, dim=-1)
    if direction == "forward":
        kl = (lt.exp() * (lt - ls)).sum(-1)
    elif direction == "reverse":
        kl = (ls.exp() * (ls - lt)).sum(-1)
    else:
        raise ValueError(direction)
    if mask is None:
        mask = torch.ones_like(kl, dtype=torch.bool)
    m = mask.float()
    return (kl * m).sum() / m.sum().clamp(min=1) * T * T


def topk_teacher(teacher_logits: torch.Tensor, k: int):
    """Compress teacher logits for storage: top-k log-probs and their token ids."""
    lp = F.log_softmax(teacher_logits.float(), dim=-1)
    vals, idx = torch.topk(lp, k, dim=-1)
    return vals.to(torch.float16), idx.to(torch.int32)


def sparse_kd_loss(student_logits: torch.Tensor, topk_logp: torch.Tensor, topk_idx: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Forward KL against a top-k teacher distribution renormalised over its support."""
    ls = F.log_softmax(student_logits.float(), dim=-1)
    ls_k = ls.gather(-1, topk_idx.long())
    pt = torch.softmax(topk_logp.float(), dim=-1)            # renormalise over k tokens
    kl = (pt * (pt.clamp_min(1e-12).log() - ls_k)).sum(-1)
    if mask is None:
        mask = torch.ones_like(kl, dtype=torch.bool)
    m = mask.float()
    return (kl * m).sum() / m.sum().clamp(min=1)


# --------------------------------------------------------------------------- training
@torch.no_grad()
def evaluate_examples(model, examples: Sequence[Dict], device="cpu") -> float:
    was = model.training
    model.eval()
    tot = sum(answer_only_loss(model, e, device).item() for e in examples)
    if was:
        model.train()
    return tot / max(1, len(examples))


def append_csv(path: str, row: Dict):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def train_distillation(model, train_examples, val_examples, device="cpu", epochs=2, lr=5e-6,
                       grad_clip: Optional[float] = 1.0, seed=123, log_every=50,
                       batch_size: int = 1, pad_id: int = 0, checkpoint_dir: Optional[str] = None,
                       csv_path: Optional[str] = None, log=print) -> List[Dict]:
    """Hard-distillation SFT loop: shuffle each epoch, answer-only CE, clip, AdamW.

    ``batch_size > 1`` switches to padded batches with token-level normalisation.
    Returns the list of logged metric rows.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)
    model.train()
    steps_per_epoch = (len(train_examples) + batch_size - 1) // batch_size
    total, step, rows = epochs * steps_per_epoch, 0, []
    t0 = time.perf_counter()
    tokens = 0
    for epoch in range(1, epochs + 1):
        order = list(train_examples)
        rng.shuffle(order)
        for s in range(0, len(order), batch_size):
            batch = order[s:s + batch_size]
            step += 1
            opt.zero_grad(set_to_none=True)
            if batch_size == 1:
                loss = answer_only_loss(model, batch[0], device)
            else:
                loss = batch_answer_only_loss(model, batch, pad_id, device, "token")
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tokens += sum(len(e["token_ids"]) for e in batch)
            if log_every and step % log_every == 0:
                val = evaluate_examples(model, val_examples, device)
                row = dict(epoch=epoch, step=step, train_loss=round(loss.item(), 6),
                           val_loss=round(val, 6),
                           tok_per_s=round(tokens / (time.perf_counter() - t0), 1))
                rows.append(row)
                if csv_path:
                    append_csv(csv_path, row)
                log(f"[epoch {epoch}/{epochs} step {step}/{total}] train {loss.item():.4f} "
                    f"val {val:.4f}")
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, f"distill-step{step:06d}-epoch{epoch}.pth"))
    return rows


# --------------------------------------------------------------------------- data generation
def generate_teacher_dataset(problems: Sequence[Dict], teacher_fn: Callable[[str], Dict],
                             out_path: str, verify_fn: Optional[Callable[[str, str], bool]] = None,
                             max_retries: int = 2) -> Dict[str, int]:
    """Resumable offline teacher-trace generation (JSONL, one record per line).

    ``teacher_fn(prompt) -> {"reasoning": str, "answer": str}`` wraps whatever serves
    the teacher (an API client, a local vLLM server ...). Already-finished problem ids
    are skipped, so a crashed job can simply be restarted. With ``verify_fn`` only
    traces whose final answer verifies against ``gtruth`` are kept (rejection sampling).
    """
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    stats = dict(skipped_existing=0, written=0, rejected=0, failed=0)
    with open(out_path, "a") as f:
        for i, prob in enumerate(problems):
            pid = prob.get("id", i)
            if pid in done:
                stats["skipped_existing"] += 1
                continue
            for _ in range(max_retries + 1):
                try:
                    out = teacher_fn(render_math_prompt(prob["problem"]))
                    break
                except Exception:  # network errors, rate limits ...
                    out = None
            if out is None:
                stats["failed"] += 1
                continue
            if verify_fn is not None and not verify_fn(out["answer"], prob["answer"]):
                stats["rejected"] += 1
                continue
            rec = dict(id=pid, problem=prob["problem"], gtruth=prob["answer"], **out)
            f.write(json.dumps(rec) + "\n")
            stats["written"] += 1
    return stats


def normalize_record(rec: Dict) -> Dict:
    """Map common teacher-trace layouts onto {problem, gtruth, reasoning, answer}.

    Handles e.g. the public ``rasbt/math_distill`` JSON layout
    (problem / gtruth_answer / message_thinking / message_content).
    """
    return dict(problem=rec["problem"],
                gtruth=rec.get("gtruth", rec.get("gtruth_answer")),
                reasoning=rec.get("reasoning", rec.get("message_thinking", "")),
                answer=rec.get("answer", rec.get("message_content", "")))


def load_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
