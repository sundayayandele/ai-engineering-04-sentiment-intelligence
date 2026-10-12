"""Load a locally approved Hugging Face sequence-classification artifact, CPU only."""
from __future__ import annotations

from pathlib import Path


class TransformerService:
    def __init__(self, model_dir: str | Path, max_length: int = 128):
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Model directory missing: {self.model_dir}")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self.torch = torch
        # Trust only locally curated artifacts. No model hub or arbitrary remote code.
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False,
            use_safetensors=True
        )
        self.model.to("cpu").eval()
        self.max_length = max_length
        self.labels = {int(k): str(v) for k, v in self.model.config.id2label.items()}

    def predict(self, text: str) -> dict:
        encoded = self.tokenizer(
            [text], truncation=True, padding=True, max_length=self.max_length,
            return_tensors="pt"
        )
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model(**encoded).logits[0], dim=-1)
        scores = {self.labels[i]: float(probabilities[i]) for i in range(len(probabilities))}
        label = max(scores, key=scores.get)
        return {"sentiment": label, "confidence": scores[label],
                "probabilities": scores, "model_type": "approved-local-transformer"}
