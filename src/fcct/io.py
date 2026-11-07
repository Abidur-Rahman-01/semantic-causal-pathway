"""Strict JSON artifacts, hashes, and run provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl(path: Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    temp.replace(path)


def environment():
    packages = {}
    for name in ("torch", "transformers", "numpy", "Pillow", "accelerate", "huggingface-hub",
                 "tokenizers", "safetensors", "requests", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
