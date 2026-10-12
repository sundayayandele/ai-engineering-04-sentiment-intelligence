# Transformer experiment — CPU/GPU capacity and TCO

These values are **planning estimates**, not observed benchmark results. Record hardware, workload and dataset size when publishing results.

| Scenario | vCPU | Host RAM | GPU VRAM | Free disk | Expected use |
|---|---:|---:|---:|---:|---|
| TF-IDF + Logistic Regression (1,200 samples) | 2–4 | 4–8 GiB | None | 5–10 GiB | Classical reference |
| DistilBERT training on CPU, 256–1,200 samples | 4–8 | 8–16 GiB | None | 10–20 GiB | Slow but no GPU cost |
| DistilBERT training on NVIDIA CUDA | 4–8 | 16 GiB | 6–8 GiB recommended | 15–25 GiB | Batch 4–8, sequence 96–128 |
| DistilBERT full-dataset experimentation | 8+ | 32+ GiB | 12–24 GiB | 30–60 GiB | Measure before scaling |
| GitHub-hosted CPU experiment | Runner-provided | Runner-provided | None | Ephemeral | Small, capped smoke runs |

## What the script measures

- Classical and transformer training wall-clock seconds.
- Accuracy, macro-F1, per-class precision/recall and confusion matrix on the **same held-out test subset**.
- Mean, p50 and p95 warm single-request inference latency (up to 50 test examples).
- Peak process resident memory on supported operating systems.
- CUDA peak allocated tensor memory when a GPU is used; this excludes driver/runtime overhead.

## TCO comparison

| Option | Monthly/training cost formula | Limitation |
|---|---|---|
| CPU classical | CPU hourly price × training hours + inference replica cost | Simpler features |
| CPU transformer | CPU hourly price × fine-tune hours + CPU inference replicas | Higher latency |
| GPU transformer | GPU hourly price × fine-tune hours + GPU inference service | Utilization matters |
| Desktop | Power + amortized hardware + labor | Not production HA |
| GitHub runner | Runner minutes × applicable billing rate | Ephemeral; plan-dependent |

Monthly serving TCO = 730 × hourly instance price × replicas + storage + traffic + logs + platform overhead + operations. Do not claim price leadership without actual run duration, region, currency and utilization.

## Decision gates

Compare models at an agreed minimum macro-F1 and p95 latency. Repeat measurements on representative production text lengths and concurrency. Pin dataset/model revisions and record Python/PyTorch/CUDA versions. DistilBERT pretraining cost is not included in fine-tuning TCO.

For reproducible commands see [Transformer Experiment](../mlops/TRANSFORMER-EXPERIMENT.md).
