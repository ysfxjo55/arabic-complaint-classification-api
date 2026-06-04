"""
Benchmark the complaint pipeline on a public Arabic sentiment dataset.

The sentiment model was fine-tuned on Saudi government complaints. We
intentionally benchmark on a *different* distribution
(arbml/Arabic_Sentiment_Twitter_Corpus) to measure out-of-distribution
generalization, which is what production deployments actually face.

Usage:
    HF_TOKEN=... python -m benchmarks.run_benchmark --n 500

Output:
    benchmarks/results.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import List, Tuple

from datasets import load_dataset

from services.action_service import predict_action_service
from services.model_loader import ModelLoader
from services.sentiment_service import predict_sentiment_service
from services.topic_service import predict_topic_service

DATASET_ID = "arbml/Arabic_Sentiment_Twitter_Corpus"
OUTPUT_PATH = Path(__file__).parent / "results.csv"
RANDOM_SEED = 42


def _stratified_sample(rows: List[dict], n: int) -> List[dict]:
    """Take roughly n samples balanced by label."""
    by_label: dict[int, List[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    per_class = max(1, n // len(by_label))
    rng = random.Random(RANDOM_SEED)
    out: List[dict] = []
    for label, items in by_label.items():
        rng.shuffle(items)
        out.extend(items[:per_class])
    rng.shuffle(out)
    return out


def _load_dataset_rows(limit: int) -> List[dict]:
    """Load full dataset and return all valid rows for stratified sampling."""
    print(f"[bench] loading {DATASET_ID}...")
    ds = load_dataset(DATASET_ID, split="train")
    rows: List[dict] = []
    for row in ds:
        text = (row.get("tweet") or "").strip()
        if not text or len(text) < 5:
            continue
        rows.append({"text": text, "label": int(row["label"])})
    label_counts = {0: 0, 1: 0}
    for r in rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    print(f"[bench] loaded {len(rows)} rows, label counts: {label_counts}")
    return rows


def _predict_one(loader: ModelLoader, text: str) -> Tuple[dict, float, str | None]:
    """Run sentiment + topic + action; return predictions, total_ms, error."""
    t0 = time.perf_counter()
    try:
        sent = predict_sentiment_service(text, loader.sentiment_model)
        topic = predict_topic_service(text, loader.topic_model)
        action = predict_action_service(text, loader.action_model)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return (
            {
                "sentiment": sent.label.value,
                "sentiment_conf": sent.confidence,
                "topic": topic.label.value,
                "topic_conf": topic.confidence,
                "action": action.label.value,
                "action_conf": action.confidence,
            },
            elapsed_ms,
            None,
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return ({}, elapsed_ms, f"{type(e).__name__}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="approximate sample size")
    args = parser.parse_args()

    raw = _load_dataset_rows(args.n)
    samples = _stratified_sample(raw, args.n)
    print(f"[bench] stratified to {len(samples)} samples")

    print("[bench] loading models (this can take ~30s on first run)...")
    loader = ModelLoader()
    loader.load_models()
    print("[bench] models ready")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "text",
                "true_label",
                "pred_sentiment",
                "sentiment_conf",
                "pred_topic",
                "topic_conf",
                "pred_action",
                "action_conf",
                "latency_ms",
                "error",
            ]
        )

        for i, row in enumerate(samples, start=1):
            pred, elapsed_ms, err = _predict_one(loader, row["text"])
            writer.writerow(
                [
                    row["text"],
                    row["label"],
                    pred.get("sentiment", ""),
                    pred.get("sentiment_conf", ""),
                    pred.get("topic", ""),
                    pred.get("topic_conf", ""),
                    pred.get("action", ""),
                    pred.get("action_conf", ""),
                    f"{elapsed_ms:.2f}",
                    err or "",
                ]
            )
            if i % 25 == 0:
                print(f"[bench] {i}/{len(samples)}")

    print(f"[bench] done -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
