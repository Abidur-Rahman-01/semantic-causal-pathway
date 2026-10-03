"""Prefetch every configured checkpoint into the project-local Hugging Face cache."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".venv" / "hf_home"
os.environ["HF_HOME"] = str(CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE / "hub")
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--components", nargs="+", choices=("qwen", "grounding", "sam2", "clip"),
                        default=("qwen", "grounding", "sam2"))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    config = json.loads(args.config.read_text(encoding="utf-8"))
    repo_ids = {"qwen": config["qwen_model"], "grounding": config["grounding_model"],
                "sam2": config["sam2_model"], "clip": config["clip_model"]}
    revisions = {"qwen": config.get("qwen_revision"), "grounding": config.get("grounding_revision"),
                 "sam2": config.get("sam2_revision"), "clip": config.get("clip_revision")}
    for component in args.components:
        repo_id = repo_ids[component]
        print(f"Downloading {component}: {repo_id} into {CACHE}", flush=True)
        snapshot_download(repo_id=repo_id, revision=revisions[component], cache_dir=str(CACHE / "hub"))
    print("All selected checkpoints are cached under .venv/hf_home", flush=True)


if __name__ == "__main__":
    main()
