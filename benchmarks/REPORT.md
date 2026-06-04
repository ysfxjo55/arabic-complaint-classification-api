# Benchmark Report

Out-of-distribution sentiment evaluation. The fine-tuned sentiment model targets Saudi government complaints; this benchmark runs it on general Arabic tweets to measure real-world generalization.

## Dataset

- **Source:** `arbml/Arabic_Sentiment_Twitter_Corpus` (Hugging Face)
- **Labels:** binary (`0` = negative, `1` = positive)
- **Sampling:** stratified by label, fixed seed (42) for reproducibility
- **Sample size:** 500

## Sentiment performance (binary)

The model can also predict `NEU`. Those predictions are counted as
*abstentions* and excluded from accuracy. They are reported separately
because abstaining from an ambiguous tweet is the correct production
behavior (the pipeline routes them to manual review).

| Metric | Value |
|---|---|
| Decided predictions (NEG or POS) | 186 / 500 (37.2%) |
| Abstentions (NEU) | 314 (62.8%) |
| Pipeline errors | 0 (0.0%) |
| **Accuracy on decided** | **68.3%** |
| Precision (POS) | 0.701 |
| Recall (POS) | 0.649 |
| F1 (POS) | 0.674 |

### Confusion matrix

| | True NEG | True POS |
|---|---|---|
| Pred NEG | 66 | 33 |
| Pred POS | 26 | 61 |

## Latency

Wall-clock latency per request, measured end-to-end through all three
classifiers (sentiment + topic + action). CPU inference unless noted.

| Percentile | Latency (ms) |
|---|---|
| p50 | 96 |
| p95 | 253 |
| p99 | 5586 |
| mean | 215 |

## Confidence distribution

Mean confidence per classifier (higher = model is more certain).
Predictions below the configured threshold (0.7) route to manual review.

| Classifier | Mean confidence | Predictions |
|---|---|---|
| Sentiment | 0.890 | 500 |
| Topic | 0.909 | 500 |
| Action | 0.948 | 500 |

## Label distribution (predicted)

### Sentiment

- `NEU`: 314 (62.8%)
- `NEG`: 99 (19.8%)
- `POS`: 87 (17.4%)

### Topic

- `FINANCIAL`: 255 (51.0%)
- `TECHNICAL`: 125 (25.0%)
- `POLICY_SECURITY`: 68 (13.6%)
- `CONTENT`: 52 (10.4%)

### Action

- `REPORT_BUG`: 290 (58.0%)
- `GENERAL_NOTE`: 192 (38.4%)
- `USER_REQUEST`: 18 (3.6%)

## Honest caveats

- This is an **out-of-distribution** benchmark. The model trained on
  formal complaint text, the dataset contains tweets with slang,
  emoji, and code-switching.
- Twitter labels are binary; we excluded `NEU` predictions from
  accuracy. Reading the `NEU` rate as low quality vs. correct
  abstention depends on the operator's risk tolerance.
- Topic and action models have no ground-truth labels in this dataset,
  so only label/confidence distributions are reported for them.
- Latency was measured on a development machine; production hardware
  will differ.

## Reproducing

```bash
HF_TOKEN=... python -m benchmarks.run_benchmark --n 500
python -m benchmarks.analyze
```
