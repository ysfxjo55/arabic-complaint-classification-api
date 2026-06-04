# Benchmarks

Empirical evaluation of the complaint pipeline on a public Arabic dataset.

## Why this exists

Anyone can ship an API. This folder answers the harder questions:

- How well does the model generalize beyond its training distribution?
- What is the request latency under realistic load?
- How often does the model abstain (return low-confidence or `NEU`)?

Results are written to [`REPORT.md`](./REPORT.md) and consumed by the public
UI to display real numbers, not marketing claims.

## Dataset

[`arbml/Arabic_Sentiment_Twitter_Corpus`](https://huggingface.co/datasets/arbml/Arabic_Sentiment_Twitter_Corpus)
— general Arabic sentiment (binary labels). The sentiment model was trained
on Saudi government complaints, so this is an **out-of-distribution** test
on purpose. It measures how the system behaves when its inputs do not match
the training data, which is the realistic production case.

## Running

```bash
# from repo root, with HF_TOKEN in .env
python -m benchmarks.run_benchmark --n 500
python -m benchmarks.analyze
```

Two artifacts are produced:

- `benchmarks/results.csv` — per-sample predictions and latency
- `benchmarks/REPORT.md`   — aggregated metrics and confusion matrix

## What is *not* measured here

- Topic and action labels are not in the dataset, so only their predicted
  distributions are reported (no accuracy claim).
- Throughput under concurrent load is not measured here. The latency
  numbers reflect single-request sequential calls.
- LLM explanation latency is excluded; only the deterministic classifier
  path is benchmarked.

## Reproducibility

- Stratified sampling uses a fixed random seed (`42`).
- Model checkpoints are pinned via `HF_MODEL_*` env vars in `.env.example`.
