#!/usr/bin/env python3
"""Evaluate ProsusAI/finbert on the Financial PhraseBank benchmark.

Produces a real, reproducible sentiment-accuracy number for the FinBERT claim.
The Financial PhraseBank (Malo et al., 2014) is the standard public benchmark
for financial-news sentiment: sentences labelled negative / neutral / positive
by finance experts, split by how strongly the annotators agreed.

Robust loading: it first tries the `datasets` library, and if that fails (newer
`datasets`/`huggingface_hub` dropped the script-based `financial_phrasebank`
dataset) it falls back to downloading the official zip from HuggingFace and
parsing the `sentence@label` text files directly. Either way you get the same
numbers.

Usage:
    python eval_finbert.py                       # 75%-agreement split (default)
    python eval_finbert.py --config sentences_allagree
    python eval_finbert.py --limit 200           # quick smoke test

Writes finbert_metrics.json next to this file. Needs internet on first run to
pull the dataset + model; both are cached afterwards.
"""
import argparse
import io
import json
import os
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

# Financial PhraseBank label order: 0=negative, 1=neutral, 2=positive
LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}

# Map the HF config name -> the text file inside the official zip.
SPLIT_FILE = {
    "sentences_50agree": "Sentences_50Agree.txt",
    "sentences_66agree": "Sentences_66Agree.txt",
    "sentences_75agree": "Sentences_75Agree.txt",
    "sentences_allagree": "Sentences_AllAgree.txt",
}

# Mirrors of the official Financial PhraseBank zip on the HuggingFace Hub.
ZIP_URLS = [
    "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip",
    "https://huggingface.co/datasets/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip",
]
ZIP_MEMBER = "FinancialPhraseBank-v1.0/{name}"


def _download_zip_bytes() -> bytes:
    """Fetch the PhraseBank zip, caching it in the system temp dir."""
    cache = Path(tempfile.gettempdir()) / "FinancialPhraseBank-v1.0.zip"
    if cache.exists() and cache.stat().st_size > 10_000:
        return cache.read_bytes()
    last_err = None
    for url in ZIP_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            cache.write_bytes(data)
            return data
        except Exception as exc:  # try the next mirror
            last_err = exc
    raise RuntimeError(
        f"Could not download Financial PhraseBank from HuggingFace: {last_err}"
    )


def _parse_phrasebank_txt(raw: str):
    """Parse 'sentence@label' lines into (sentences, label_ids)."""
    sentences, labels = [], []
    for line in raw.splitlines():
        if "@" not in line:
            continue
        sentence, label = line.rsplit("@", 1)
        label = label.strip().lower()
        if label not in LABEL_TO_ID:
            continue
        sentences.append(sentence.strip())
        labels.append(LABEL_TO_ID[label])
    return sentences, labels


def load_phrasebank(config: str):
    """Return (sentences, label_ids). Tries `datasets`, falls back to raw zip."""
    # 1) Preferred: the datasets library (works on older versions).
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "financial_phrasebank", config, split="train", trust_remote_code=True
        )
        return list(ds["sentence"]), list(ds["label"])
    except Exception as exc:
        print(f"datasets load unavailable ({type(exc).__name__}); "
              f"falling back to the official zip ...")

    # 2) Fallback: download + parse the official zip directly.
    if config not in SPLIT_FILE:
        raise ValueError(f"Unknown config {config!r}; expected one of {list(SPLIT_FILE)}")
    data = _download_zip_bytes()
    member = ZIP_MEMBER.format(name=SPLIT_FILE[config])
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        raw = z.read(member).decode("latin-1")
    sentences, labels = _parse_phrasebank_txt(raw)
    if not sentences:
        raise RuntimeError(f"Parsed 0 sentences from {member}")
    return sentences, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--config",
        default="sentences_75agree",
        choices=list(SPLIT_FILE),
        help="PhraseBank annotator-agreement split (default: sentences_75agree)",
    )
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N sentences (smoke test)")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    # Heavy imports here so --help works without them installed.
    from transformers import pipeline
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
    )

    print(f"Loading Financial PhraseBank ({args.config}) ...")
    sentences, gold = load_phrasebank(args.config)
    if args.limit:
        sentences, gold = sentences[:args.limit], gold[:args.limit]

    print(f"Loading ProsusAI/finbert and scoring {len(sentences)} sentences ...")
    clf = pipeline(
        "sentiment-analysis", model="ProsusAI/finbert",
        truncation=True, batch_size=args.batch_size,
    )

    t0 = time.time()
    preds_raw = clf(sentences)
    elapsed = time.time() - t0
    pred = [LABEL_TO_ID[p["label"].lower()] for p in preds_raw]

    acc = accuracy_score(gold, pred)
    report = classification_report(
        gold, pred, target_names=LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(gold, pred, labels=[0, 1, 2]).tolist()

    metrics = {
        "model": "ProsusAI/finbert",
        "dataset": f"financial_phrasebank/{args.config}",
        "n_samples": len(gold),
        "accuracy": round(acc, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "per_class": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix": cm,
        "throughput_sentences_per_sec": round(len(gold) / elapsed, 1) if elapsed else None,
    }

    out_path = Path(__file__).with_name("finbert_metrics.json")
    out_path.write_text(json.dumps(metrics, indent=2))

    print()
    print(f"FinBERT on {metrics['dataset']}  (n={metrics['n_samples']})")
    print(f"  accuracy    : {acc * 100:5.1f}%")
    print(f"  macro F1    : {metrics['macro_f1'] * 100:5.1f}%")
    print(f"  weighted F1 : {metrics['weighted_f1'] * 100:5.1f}%")
    for label in LABELS:
        pc = metrics["per_class"][label]
        print(f"    {label:8s}  P={pc['precision']:.2f}  R={pc['recall']:.2f}  "
              f"F1={pc['f1']:.2f}  (n={pc['support']})")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
