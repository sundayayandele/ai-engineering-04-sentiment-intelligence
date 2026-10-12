# Project 04 — Real dataset, classical baseline versus DistilBERT

## Reproducible experiment

Dataset: [TweetEval sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval), **sentiment subset only**. The dataset card identifies this subset as **Creative Commons Attribution 3.0 Unported (CC BY 3.0)**. Attribute TweetEval and the underlying SemEval sentiment task when publishing results. The aggregate dataset metadata says "unknown" because other TweetEval tasks have different or undefined licensing; do **not** treat the whole collection as CC BY.

Models:
- **Classical:** TF-IDF (1–2 grams) + Logistic Regression.
- **Transformer:** `distilbert/distilbert-base-uncased` with 3-way classification head, fine-tuned on the *same* selected training examples.
- Official train, validation and test splits remain disjoint. No test labels are used for model selection. Fixed shuffle seed 42, equal held-out test subset.
- Three labels: negative, neutral, positive. Both models report accuracy, macro-F1, per-class precision/recall/F1, confusion matrix and per-request latency.
- Outputs are JSON under `artifacts/tweet_eval_experiment/comparison.json` and ignored by Git.

## Local CPU or GPU setup

Python 3.12 recommended. Hugging Face downloads require internet access and adequate disk space.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[experiment]"
python scripts/compare_transformer.py --train-samples 1200 --validation-samples 300 --test-samples 300 --epochs 1 --batch-size 8 --device cpu
```

For a **baseline-only** quick run:

```bash
python scripts/compare_transformer.py --train-samples 1200 --validation-samples 300 --test-samples 300 --skip-transformer
```

For a compatible NVIDIA CUDA GPU, install the PyTorch wheel recommended for your CUDA/driver combination from [PyTorch installation guidance](https://pytorch.org/get-started/locally/) and run with `--device cuda`. GPU is optional; do not assume Windows/macOS GPU compatibility.

For a small CPU smoke run: `--train-samples 256 --validation-samples 64 --test-samples 96 --epochs 1 --batch-size 4 --max-length 96`. Results from 256 samples are for workflow validation, not quality claims.

## GitHub-hosted runner

In **Actions → transformer-experiment → Run workflow**, choose `baseline` first (default). Then optionally choose `transformer`; it can take considerably longer on CPU. The workflow caps train/test sample sizes, uploads a JSON artifact, and does **not** deploy a model. It is separate from regular CI to avoid repeated large downloads and excessive runner costs.

## Fair evaluation and limitations

- The transformer begins from a pretrained checkpoint, while the TF-IDF model is trained from scratch. Report that distinction.
- Same training subset and test subset, but transformer uses pretraining data and compute that are not counted in fine-tuning TCO.
- Report mean, p50 and p95 single-request latency; for accurate service sizing, add concurrent load tests, warm/cold starts and repeated runs.
- CPU and GPU timings are hardware-specific and must not be compared across unlike hosts without caveats.
- Review sarcasm, dialect, code-switching, privacy and subgroup performance before production.
- The experiment intentionally does not upload model weights, downloaded datasets or customer content.

## Evidence checklist

Record dataset and model revision IDs, package versions, seed, split sizes, host CPU/GPU, RAM, training duration, macro-F1, per-class metrics, latency and peak process RSS/GPU memory. The script currently records model/dataset names and platform; for rigorous publication, additionally pin and record immutable revisions and run multiple seeds.

See [capacity sizing](../infrastructure/CAPACITY-PLANNING.md) and [TCO](../COST-TCO.md).
