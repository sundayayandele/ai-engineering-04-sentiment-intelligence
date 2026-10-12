# Verified TweetEval CPU comparison — 12 October 2026

**Provenance:** [GitHub Actions transformer-experiment run 38305611278](https://github.com/sundayayandele/ai-engineering-04-sentiment-intelligence/actions/runs/38305611278), successful on commit `adca583b9f4907485bb6600f3ffa9f24d8c4c358`. [Download original comparison.json artifact](https://github.com/sundayayandele/ai-engineering-04-sentiment-intelligence/actions/runs/38305611278/artifacts/11773086224) (GitHub artifact retention applies). These figures were read directly from that JSON and corroborated with the runner log.

Dataset: `cardiffnlp/tweet_eval`, sentiment, 3 classes; seed 42; 256 training, 64 validation, 96 test samples. Runner: Linux Azure x86_64, Python 3.12.15. DistilBERT checkpoint: `distilbert/distilbert-base-uncased`; CPU, 1 epoch, batch size 4, maximum token length 96.

| Measured quantity | TF-IDF + Logistic Regression | DistilBERT fine-tuned CPU |
|---|---:|---:|
| Test accuracy | 0.427083 | 0.416667 |
| Test macro-F1 | 0.308924 | 0.196078 |
| Train duration | 0.52 seconds | 26.66 seconds |
| Mean inference latency, 50 requests | 0.78 ms | 34.49 ms |
| Median inference latency | 0.73 ms | 34.03 ms |
| p95 inference latency | 0.94 ms | 43.87 ms |

Observed process peak resident memory: **2,279,682,048 bytes ≈ 2.12 GiB** for the experiment process (not a separate per-model figure). GPU memory: **not measured** because the run was CPU-only.

Confusion matrices use rows = true label, columns = predicted label, ordered `negative, neutral, positive`.

- Baseline: `[[0,25,7],[0,33,7],[0,16,8]]`
- DistilBERT: `[[0,32,0],[0,40,0],[0,24,0]]`

**Interpretation:** DistilBERT predicted only neutral in this extremely small training run. The TF-IDF baseline also never predicted negative. Neither model is acceptable for production. The baseline's mean inference latency was **44.2× lower** (34.49 / 0.78); the transformer took **51.3× longer** to train (26.66 / 0.52). Timing excludes dependency installation, dataset/model download, cloud provisioning and service network overhead. Latency measurements are 50 local prediction calls, not load-tested p95 under concurrency.

## CPU versus GPU and TCO

| Deployment option | Measured in this run? | Resource and cost conclusion |
|---|---|---|
| CPU TF-IDF baseline | Yes | Low latency; small CPU/RAM footprint; poor classification on this subset |
| CPU DistilBERT | Yes | 34.49 ms mean inference, 26.66 s fine-tuning; process peak 2.12 GiB across experiment |
| GPU DistilBERT | **No** | GPU throughput, GPU memory and training duration require a separate controlled GPU run |
| Cloud GPU cost | **No** | Must use region/date-specific hourly prices and measured runtime/utilization; no verified GPU savings claim |

For illustrative **compute-only** comparisons, let `R_cpu` and `R_gpu` be the hourly VM prices, and `t_cpu`, `t_gpu` be measured training durations in seconds:

`CPU_training_cost = R_cpu × t_cpu / 3600`

`GPU_training_cost = R_gpu × t_gpu / 3600`

`GPU_break_even = (t_cpu / t_gpu) × R_cpu` (GPU hourly rate must be below this value to win on training compute cost alone).

This run establishes `t_cpu = 26.66 s` **for only 256 examples**, but no `t_gpu` or verified VM rate. Consequently **no defensible numeric GPU/CPU cost ratio is available**. A hosted runner's wall-clock billing, package installation and model downloads can dominate this tiny training step. See [TCO planning](../infrastructure/TRANSFORMER-CPU-GPU-TCO.md).

## Next validation gate

Run a larger stratified, fixed-seed experiment with an approved dataset, 3+ training epochs, multiple seeds, per-class recall, confidence intervals, latency under concurrency, model artifact/version capture and separately benchmarked GPU hardware. Until then treat this as **pipeline validation, not model-quality validation**.
