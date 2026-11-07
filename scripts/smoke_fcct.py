"""Real trained-checkpoint engineering check; never a paper evaluation."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["HF_HOME"] = str(ROOT / ".venv/hf_home")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image

from fcct.core import capture, patch_component
from fcct.data import QUESTION
from fcct.experiment import code_hashes, deterministic
from fcct.io import environment, sha256, write_json
from fcct.llava import Llava15, gaussian_corruption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fcct_llava15.json")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fcct_engineering_smoke.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_hashes = code_hashes(ROOT)
    deterministic(config["seed"])
    started = time.perf_counter()
    backend = Llava15(config)
    image = Image.open(args.image).convert("RGB")
    question = QUESTION.format(object=args.object)
    batch, spans = backend.batch(image, question, args.object)
    length = batch["input_ids"].shape[1]
    baseline = backend.probability(batch)
    cached = backend.trace_batch(batch)
    cached_p = backend.probability(cached)
    with capture(backend.layers, {"last": [length - 1]}) as (activations, _):
        clean = backend.probability(cached)
    noise = gaussian_corruption(image, config["noise_sigma"], config["seed"])
    corrupt, _ = backend.batch(noise, question)
    corrupt = backend.trace_batch(corrupt)
    corrupted = backend.probability(corrupt)
    with patch_component(backend.layers[-1], "hidden", [length - 1], activations[31, "hidden"]):
        restored = backend.probability(corrupt)
    after = backend.probability(corrupt)
    answer = backend.generate(image, question, 8)
    checks = {"cached_vision_matches_direct": abs(baseline - cached_p) < 1e-6,
              "final_hidden_restores_clean": abs(clean - restored) < 1e-6,
              "patch_hooks_removed": after == corrupted,
              "visual_tokens": len(spans["image_positions"]) == 576,
              "textual_object_span": bool(spans["object_text"]),
              "generation_nonempty": bool(answer)}
    result = {"status": "engineering_check_only", "paper_result": False, "checks": checks,
              "all_checks_pass": all(checks.values()), "model_id": config["model_id"],
              "config": config, "code_sha256": source_hashes,
              "revision": config["revision"], "image_sha256": sha256(args.image),
              "question": question, "yes_token_id": backend.yes_token_id,
              "prompt_tokens": length, "p_clean": clean, "p_corrupted": corrupted,
              "p_final_hidden_restored": restored, "illustrative_baseline_answer": answer,
              "seconds_including_load": time.perf_counter() - started,
              "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
              "environment": environment()}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
