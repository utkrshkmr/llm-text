"""Fine-tune GPT-2 small as an SMS spam classifier (Chapter 28, end-to-end).

Requirements (run on your own machine):
  * the SMS Spam Collection file (tab-separated "label<TAB>text" lines), e.g. from the
    UCI ML repository (archive "sms+spam+collection.zip" -> SMSSpamCollection)
  * GPT-2 weights as safetensors: ``huggingface-cli download gpt2 --local-dir gpt2``
  * ``pip install tiktoken safetensors``

Example:
  python run.py --data SMSSpamCollection --gpt2-dir gpt2 --epochs 5 --bucketed
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from llmsys.finetune import padding_stats  # noqa: E402
from llmsys.gpt import GPTModel, gpt2_config, load_gpt2_hf_state_dict  # noqa: E402
from llmsys.utils import count_parameters, get_device, set_seed  # noqa: E402
import solution as lab  # noqa: E402


def read_sms(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "\t" not in line:
                continue
            lab_, text = line.rstrip("\n").split("\t", 1)
            texts.append(text)
            labels.append(1 if lab_ == "spam" else 0)
    return texts, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gpt2-dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--trainable-blocks", type=int, default=1)
    ap.add_argument("--bucketed", action="store_true")
    ap.add_argument("--max-length", type=int, default=None)
    args = ap.parse_args()

    import tiktoken
    from safetensors.torch import load_file

    set_seed(123)
    device = get_device()
    tok = tiktoken.get_encoding("gpt2")
    texts, labels = read_sms(args.data)
    seqs = [tok.encode(t)[: args.max_length or 1024] for t in texts]
    print(f"{len(seqs)} messages, spam fraction {sum(labels)/len(labels):.3f}")

    lab.PAD_ID = 50256  # <|endoftext|> doubles as padding
    train, val, test = lab.make_loaders(seqs, labels, args.batch_size, 123, args.bucketed)
    lengths = [len(s) for s in seqs]
    print("padding waste (train batches):",
          f"{padding_stats([len(x[0]) for x in train.dataset], list(train.batch_sampler))['waste']:.1%}")

    model = GPTModel(gpt2_config("gpt2-small (124M)", drop_rate=0.0))
    load_gpt2_hf_state_dict(model, load_file(os.path.join(args.gpt2_dir, "model.safetensors")))
    lab.add_classification_head(model, 2, args.trainable_blocks)
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {trainable:,} / {count_parameters(model):,}")
    print("accuracy before fine-tuning:", lab.accuracy_loader(model, val, device))

    t0 = time.perf_counter()
    lab.train_classifier(model, train, val, args.epochs, args.lr, 0.1, device)
    print(f"training took {time.perf_counter()-t0:.1f}s")
    print(f"test accuracy {lab.accuracy_loader(model, test, device):.3f}")
    torch.save(model.state_dict(), "spam_classifier.pth")

    def classify(text):
        ids = torch.tensor([tok.encode(text)], device=device)
        with torch.no_grad():
            pred = lab.select_class_logits(model(ids)).argmax(-1).item()
        return "spam" if pred else "not spam"

    print(classify("You are a winner! Reply now to claim your free prize."))
    print(classify("Are we still on for dinner tonight?"))


if __name__ == "__main__":
    main()
