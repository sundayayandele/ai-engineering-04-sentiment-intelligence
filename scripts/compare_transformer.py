"""Reproducible TweetEval sentiment comparison; never commits downloaded data/models."""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

LABELS = ["negative", "neutral", "positive"]
DATASET_ID = "cardiffnlp/tweet_eval"
MODEL_ID = "distilbert/distilbert-base-uncased"


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-samples", type=int, default=1200)
    p.add_argument("--validation-samples", type=int, default=300)
    p.add_argument("--test-samples", type=int, default=300)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratified", action="store_true", help="Stratify each official split by label")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--output-dir", default="artifacts/tweet_eval_experiment")
    p.add_argument("--skip-transformer", action="store_true")
    p.add_argument("--save-model", action="store_true", help="Export local safetensors model/tokenizer for staging")
    return p


def sample_split(ds, count: int, seed: int):
    if count < 1:
        raise ValueError("sample counts must be positive")
    count = min(count, len(ds))
    # Fixed seed and official split: no test leakage.
    return ds.shuffle(seed=seed).select(range(count))


def stratified_split(ds, count: int, seed: int):
    """Sample each class within the supplied official split, without cross-split leakage."""
    if count < 3:
        raise ValueError("Stratified sample count must be at least 3")
    count = min(count, len(ds))
    labels = sorted(set(ds["label"]))
    if count < len(labels):
        raise ValueError("Sample too small to cover every label")
    shuffled = ds.shuffle(seed=seed)
    groups = {label: [] for label in labels}
    for i, label in enumerate(shuffled["label"]):
        groups[label].append(i)
    # Allocate roughly equal class quotas, redistributing exhausted capacity.
    quotas = {label: 0 for label in labels}
    while sum(quotas.values()) < count:
        candidates = [label for label in labels if quotas[label] < len(groups[label])]
        if not candidates:
            break
        chosen = min(candidates, key=lambda label: (quotas[label], label))
        quotas[chosen] += 1
    indices = [i for label in labels for i in groups[label][:quotas[label]]]
    return shuffled.select(indices).shuffle(seed=seed)


def metric_report(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification": classification_report(
            y_true, y_pred, labels=list(range(3)), target_names=LABELS,
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(3))).tolist(),
    }


def benchmark(predict_fn, texts, warmup=2):
    if not texts:
        raise ValueError("Cannot benchmark an empty test set")
    for item in texts[:warmup]:
        predict_fn([item])
    times = []
    for item in texts:
        start = time.perf_counter()
        predict_fn([item])
        times.append((time.perf_counter() - start) * 1000)
    return {
        "requests": len(times),
        "mean_ms": round(statistics.mean(times), 2),
        "p50_ms": round(float(np.percentile(times, 50)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
    }


def main():
    from datasets import load_dataset

    cfg = args_parser().parse_args()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # TweetEval sentiment subset: CC BY 3.0 according to the dataset card.
    # Keep the official train, validation and test partitions separate.
    ds = load_dataset(DATASET_ID, "sentiment")
    sampler = stratified_split if cfg.stratified else sample_split
    train = sampler(ds["train"], cfg.train_samples, cfg.seed)
    valid = sampler(ds["validation"], cfg.validation_samples, cfg.seed)
    test = sampler(ds["test"], cfg.test_samples, cfg.seed)
    texts_train, labels_train = train["text"], train["label"]
    texts_test, labels_test = test["text"], test["label"]

    baseline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20_000)),
        ("classifier", LogisticRegression(max_iter=600, random_state=cfg.seed)),
    ])
    start = time.perf_counter()
    baseline.fit(texts_train, labels_train)
    baseline_seconds = time.perf_counter() - start
    baseline_predictions = baseline.predict(texts_test).tolist()
    results = {
        "dataset": DATASET_ID,
        "dataset_config": "sentiment",
        "dataset_license": "CC BY 3.0 per dataset card; attribution required",
        "dataset_url": "https://huggingface.co/datasets/cardiffnlp/tweet_eval",
        "labels": LABELS,
        "seed": cfg.seed,
        "stratified": cfg.stratified,
        "train_samples": len(train),
        "validation_samples": len(valid),
        "test_samples": len(test),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "baseline": {
            **metric_report(labels_test, baseline_predictions),
            "train_seconds": round(baseline_seconds, 2),
            "latency": benchmark(lambda t: baseline.predict(t), texts_test[: min(50, len(test))]),
        },
    }

    if not cfg.skip_transformer:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )

        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = "cuda" if cfg.device == "cuda" or (
            cfg.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
        torch.manual_seed(cfg.seed)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=3, id2label=dict(enumerate(LABELS)),
            label2id={v: i for i, v in enumerate(LABELS)}
        )
        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)
        train_tokens = train.map(tokenize, batched=True)
        valid_tokens = valid.map(tokenize, batched=True)
        test_tokens = test.map(tokenize, batched=True)
        training_args = TrainingArguments(
            output_dir=str(output / "checkpoints"),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            learning_rate=3e-5,
            weight_decay=0.01,
            save_strategy="no",
            report_to="none",
            logging_steps=20,
            seed=cfg.seed,
            dataloader_num_workers=0,
            use_cpu=device == "cpu",
        )
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_tokens, eval_dataset=valid_tokens,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )
        start = time.perf_counter()
        trainer.train()
        train_seconds = time.perf_counter() - start
        prediction = trainer.predict(test_tokens)
        predictions = np.argmax(prediction.predictions, axis=1).tolist()
        model.eval()
        model.to(device)

        def transformer_predict(items):
            encoded = tokenizer(items, return_tensors="pt", truncation=True,
                                padding=True, max_length=cfg.max_length)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                return model(**encoded).logits.argmax(dim=-1).cpu().tolist()

        results["transformer"] = {
            **metric_report(labels_test, predictions),
            "model": MODEL_ID,
            "device": device,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "max_length": cfg.max_length,
            "train_seconds": round(train_seconds, 2),
            "latency": benchmark(transformer_predict, texts_test[: min(50, len(test))]),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated())
                if device == "cuda" else None,
        }
        if cfg.save_model:
            model_path = output / "approved_model"
            model_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_path, safe_serialization=True)
            tokenizer.save_pretrained(model_path)
            results["transformer"]["export_path"] = str(model_path)
            results["transformer"]["promotion_status"] = "unapproved-staging-only"
        del trainer, model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Linux ru_maxrss is KiB; on macOS it is bytes.
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        results["process_peak_rss_bytes"] = int(
            rss * 1024 if platform.system() == "Linux" else rss
        )
    except ImportError:
        results["process_peak_rss_bytes"] = None
    target = output / "comparison.json"
    target.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline_macro_f1": results["baseline"]["macro_f1"],
        "transformer_macro_f1": results.get("transformer", {}).get("macro_f1"),
        "results": str(target),
    }, indent=2))


if __name__ == "__main__":
    main()
