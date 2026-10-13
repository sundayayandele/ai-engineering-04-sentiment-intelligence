"""Generate a deterministic, reviewable staging model manifest (no automatic promotion)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED = ("config.json", "model.safetensors", "tokenizer_config.json")


def manifest(model_dir: Path, comparison: Path) -> dict:
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    files = {}
    for name in REQUIRED:
        path = model_dir / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f"Missing, symlinked or empty model artifact: {name}")
    for path in sorted(model_dir.iterdir()):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Unexpected non-regular artifact: {path.name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files[path.name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    metrics = json.loads(comparison.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "stage": "staging-unapproved",
        "production_approved": False,
        "seed": metrics.get("seed"),
        "dataset": metrics.get("dataset"),
        "dataset_config": metrics.get("dataset_config"),
        "transformer_macro_f1": metrics.get("transformer", {}).get("macro_f1"),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = manifest(args.model_dir, args.comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(f"Staging manifest written: {args.output}")


if __name__ == "__main__":
    main()
