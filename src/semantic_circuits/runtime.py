"""Project-local cache and device helpers used by every model entry point."""
from __future__ import annotations

import os
import re
from pathlib import Path


def configure_local_runtime(project_root: Path) -> Path:
    """Keep downloaded model weights and hub metadata inside the project venv."""
    cache = project_root.resolve() / ".venv" / "hf_home"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    return cache


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_cached_revision(model_id: str, revision: str | None = None) -> str | None:
    """Return the resolved Hub commit recorded in the local cache, when available."""
    if revision and re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return revision
    if Path(model_id).exists():
        return revision
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_cache = cache_root / ("models--" + model_id.replace("/", "--"))
    ref_file = repo_cache / "refs" / (revision or "main")
    try:
        return ref_file.read_text(encoding="utf-8").strip() or revision
    except OSError:
        return revision
