"""
Analyze benchmark results and produce REPORT.md.

Reads benchmarks/results.csv (written by run_benchmark.py) and computes:
- Binary sentiment accuracy / precision / recall / F1 (excluding NEU abstentions)
- Abstention (NEU) rate
- Confusion matrix (NEG vs POS)
- Confidence distribution
- Latency p50 / p95 / p99
- Error rate

Usage:
    python -m benchmarks.analyze
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results.csv"
REPORT_PATH = Path(__file__).parent / "REPORT.md"


def _pct(num: float, denom: float) -> str:
    if denom == 0:
        return "n/a"
    return f"{(num / denom) * 100:.1f}%"


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit(
            f"results.csv not found at {RESULTS_PATH}. "
            f"Run `python -m benchmarks.run_benchmark` first."
        )

    rows = []
    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    total = len(rows)
    errors = sum(1 for r in rows if r["error"])
    ok = [r for r in rows if not r["error"]]

    sentiments = Counter(r["pred_sentiment"] for r in ok)
    topics = Counter(r["pred_topic"] for r in ok)
    actions = Counter(r["pred_action"] for r in ok)

    neu_count = sentiments.get("NEU", 0)
    decided = [r for r in ok if r["pred_sentiment"] in ("NEG", "POS")]
    label_map = {"NEG": 0, "POS": 1}

    correct = 0
    confusion = {
        ("NEG", 0): 0,
        ("NEG", 1): 0,
        ("POS", 0): 0,
        ("POS", 1): 0,
    }
    for r in decided:
        pred = label_map[r["pred_sentiment"]]
        true = int(r["true_label"])
        confusion[(r["pred_sentiment"], true)] += 1
        if pred == true:
            correct += 1

    tp = confusion[("POS", 1)]
    fp = confusion[("POS", 0)]
    fn = confusion[("NEG", 1)]
    tn = confusion[("NEG", 0)]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    sent_confs = [float(r["sentiment_conf"]) for r in ok if r["sentiment_conf"]]
    topic_confs = [float(r["topic_conf"]) for r in ok if r["topic_conf"]]
    action_confs = [float(r["action_conf"]) for r in ok if r["action_conf"]]
    latencies = [float(r["latency_ms"]) for r in rows if r["latency_ms"]]
    latencies.sort()

    def pctile(values, p):
        if not values:
            return 0.0
        idx = max(0, min(len(values) - 1, int(round((p / 100) * len(values))) - 1))
        return values[idx]

    md = []
    md.append("# Benchmark Report")
    md.append("")
    md.append(
        "Out-of-distribution sentiment evaluation. The fine-tuned sentiment "
        "model targets Saudi government complaints; this benchmark runs it on "
        "general Arabic tweets to measure real-world generalization."
    )
    md.append("")
    md.append("## Dataset")
    md.append("")
    md.append("- **Source:** `arbml/Arabic_Sentiment_Twitter_Corpus` (Hugging Face)")
    md.append("- **Labels:** binary (`0` = negative, `1` = positive)")
    md.append("- **Sampling:** stratified by label, fixed seed (42) for reproducibility")
    md.append(f"- **Sample size:** {total}")
    md.append("")
    md.append("## Sentiment performance (binary)")
    md.append("")
    md.append("The model can also predict `NEU`. Those predictions are counted as")
    md.append("*abstentions* and excluded from accuracy. They are reported separately")
    md.append("because abstaining from an ambiguous tweet is the correct production")
    md.append("behavior (the pipeline routes them to manual review).")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---|")
    md.append(f"| Decided predictions (NEG or POS) | {len(decided)} / {total} ({_pct(len(decided), total)}) |")
    md.append(f"| Abstentions (NEU) | {neu_count} ({_pct(neu_count, total)}) |")
    md.append(f"| Pipeline errors | {errors} ({_pct(errors, total)}) |")
    md.append(f"| **Accuracy on decided** | **{_pct(correct, len(decided))}** |")
    md.append(f"| Precision (POS) | {precision:.3f} |")
    md.append(f"| Recall (POS) | {recall:.3f} |")
    md.append(f"| F1 (POS) | {f1:.3f} |")
    md.append("")
    md.append("### Confusion matrix")
    md.append("")
    md.append("| | True NEG | True POS |")
    md.append("|---|---|---|")
    md.append(f"| Pred NEG | {tn} | {fn} |")
    md.append(f"| Pred POS | {fp} | {tp} |")
    md.append("")
    md.append("## Latency")
    md.append("")
    md.append("Wall-clock latency per request, measured end-to-end through all three")
    md.append("classifiers (sentiment + topic + action). CPU inference unless noted.")
    md.append("")
    md.append("| Percentile | Latency (ms) |")
    md.append("|---|---|")
    md.append(f"| p50 | {pctile(latencies, 50):.0f} |")
    md.append(f"| p95 | {pctile(latencies, 95):.0f} |")
    md.append(f"| p99 | {pctile(latencies, 99):.0f} |")
    md.append(f"| mean | {statistics.mean(latencies):.0f} |")
    md.append("")
    md.append("## Confidence distribution")
    md.append("")
    md.append("Mean confidence per classifier (higher = model is more certain).")
    md.append("Predictions below the configured threshold (0.7) route to manual review.")
    md.append("")
    md.append("| Classifier | Mean confidence | Predictions |")
    md.append("|---|---|---|")
    if sent_confs:
        md.append(f"| Sentiment | {statistics.mean(sent_confs):.3f} | {len(sent_confs)} |")
    if topic_confs:
        md.append(f"| Topic | {statistics.mean(topic_confs):.3f} | {len(topic_confs)} |")
    if action_confs:
        md.append(f"| Action | {statistics.mean(action_confs):.3f} | {len(action_confs)} |")
    md.append("")
    md.append("## Label distribution (predicted)")
    md.append("")
    md.append("### Sentiment")
    md.append("")
    for label, count in sentiments.most_common():
        md.append(f"- `{label}`: {count} ({_pct(count, sum(sentiments.values()))})")
    md.append("")
    md.append("### Topic")
    md.append("")
    for label, count in topics.most_common():
        md.append(f"- `{label}`: {count} ({_pct(count, sum(topics.values()))})")
    md.append("")
    md.append("### Action")
    md.append("")
    for label, count in actions.most_common():
        md.append(f"- `{label}`: {count} ({_pct(count, sum(actions.values()))})")
    md.append("")
    md.append("## Honest caveats")
    md.append("")
    md.append("- This is an **out-of-distribution** benchmark. The model trained on")
    md.append("  formal complaint text, the dataset contains tweets with slang,")
    md.append("  emoji, and code-switching.")
    md.append("- Twitter labels are binary; we excluded `NEU` predictions from")
    md.append("  accuracy. Reading the `NEU` rate as low quality vs. correct")
    md.append("  abstention depends on the operator's risk tolerance.")
    md.append("- Topic and action models have no ground-truth labels in this dataset,")
    md.append("  so only label/confidence distributions are reported for them.")
    md.append("- Latency was measured on a development machine; production hardware")
    md.append("  will differ.")
    md.append("")
    md.append("## Reproducing")
    md.append("")
    md.append("```bash")
    md.append("HF_TOKEN=... python -m benchmarks.run_benchmark --n 500")
    md.append("python -m benchmarks.analyze")
    md.append("```")
    md.append("")

    REPORT_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"[analyze] wrote {REPORT_PATH}")
    print(f"[analyze] accuracy on decided: {_pct(correct, len(decided))}")
    print(f"[analyze] f1 (POS): {f1:.3f}")
    print(f"[analyze] latency p50/p95: {pctile(latencies, 50):.0f}/{pctile(latencies, 95):.0f} ms")


if __name__ == "__main__":
    main()
