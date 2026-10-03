"""Run a reviewed JSONL manifest through the semantic pathway experiment."""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".venv" / "hf_home")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(ROOT / ".venv" / "hf_home" / "hub")
sys.path.insert(0, str(ROOT / "src"))


def rows_from_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def resolve_path(value, manifest: Path, data_root: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    from_manifest = manifest.parent / candidate
    return from_manifest if from_manifest.exists() else data_root / candidate


def vqa_soft_score(prediction: str, answers: list[str] | None) -> float | None:
    """Official VQA consensus scoring rule, applied to the available 10 human answers."""
    if not answers:
        return None
    import re
    number_words = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
    def normalize(value):
        value = value.lower().strip()
        value = re.sub(r"(?<=\d),(?=\d)", "", value)
        value = re.sub(r"[^\w\s']", " ", value)
        words = [number_words.get(word, word) for word in value.split() if word not in {"a", "an", "the"}]
        return " ".join(words)
    pred = normalize(prediction)
    refs = [normalize(answer) for answer in answers]
    return min(sum(answer == pred for answer in refs) / 3.0, 1.0)


def run_manifest(config: dict, limit: int | None = None, resume: bool = True):
    from semantic_circuits.qwen_backend import Qwen25VL, QwenConfig
    from semantic_circuits.runner import run_pathway_instance
    import numpy as np
    import torch

    seed = int(config.get("seed", 17))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    manifest = Path(config["manifest"])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    data_root = Path(config.get("data_root", "data"))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    output_dir = Path(config.get("output_dir", "outputs"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "semantic_pathway_results.jsonl"
    completed = set()
    if resume and output_path.exists():
        completed = {str(row.get("sample_id")) for row in rows_from_jsonl(output_path)}

    if limit is None and config.get("max_samples") is not None:
        limit = int(config["max_samples"])
    model = Qwen25VL(QwenConfig(
        model_id=config["qwen_model"], revision=config.get("qwen_revision"),
        device=config.get("device", "auto"), dtype=config.get("dtype", "auto"),
        max_new_tokens=int(config.get("max_new_tokens", 48)),
        max_image_pixels=config.get("max_image_pixels", 1_003_520)))
    gpu = None
    if model.device.type == "cuda":
        gpu = {"name": torch.cuda.get_device_name(model.device),
               "total_memory_bytes": int(torch.cuda.get_device_properties(model.device).total_memory)}
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    runtime_metadata = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "torch_cuda_build": torch.version.cuda,
        "transformers": transformers_version, "device": str(model.device),
        "gpu": gpu, "parameter_dtype": str(next(model.model.parameters()).dtype),
        "seed": seed, "qwen_resolved_revision": model.resolved_revision,
    }
    count, failures = 0, []
    with output_path.open("a", encoding="utf-8") as sink:
        for row in rows_from_jsonl(manifest):
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise ValueError("Every manifest row needs a stable sample_id.")
            if sample_id in completed:
                continue
            if row.get("split") == "test" and config.get("allow_test_split") is not True:
                raise ValueError("Locked test split blocked. Set allow_test_split=true only for final evaluation.")
            if row.get("evidence_mask_reviewed") is not True:
                raise ValueError(f"{sample_id}: evidence_mask_reviewed must be true after visual review.")
            image_path = resolve_path(row["image"], manifest, data_root)
            mask_path = resolve_path(row["evidence_mask"], manifest, data_root)
            image = Image.open(image_path).convert("RGB")
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            variants = []
            for variant in row.get("variants", []):
                variant_path = resolve_path(variant["image"], manifest, data_root)
                loaded = {**variant, "image": Image.open(variant_path).convert("RGB")}
                if variant.get("evidence_mask"):
                    variant_mask_path = resolve_path(variant["evidence_mask"], manifest, data_root)
                    loaded["evidence_mask"] = np.asarray(Image.open(variant_mask_path).convert("L")) > 0
                variants.append(loaded)
            control_masks = []
            for control in row.get("control_masks", []):
                if control.get("reviewed") is not True or control.get("accepted") is not True:
                    continue
                control_path = resolve_path(control["image"], manifest, data_root)
                control_masks.append(__import__("numpy").asarray(Image.open(control_path).convert("L")) > 0)
            min_controls = int(config.get("min_reviewed_controls", 1))
            if config.get("require_reviewed_controls", False) and len(control_masks) < min_controls:
                raise ValueError(f"{sample_id}: {len(control_masks)} reviewed spatial controls; protocol requires {min_controls}.")
            min_variants = int(config.get("min_accepted_variants", 4))
            accepted_count = sum(all(variant.get(key) is True for key in
                ("answer_preserved", "critical_evidence_preserved", "relations_preserved", "human_audited"))
                for variant in variants)
            if accepted_count < min_variants:
                raise ValueError(f"{sample_id}: {accepted_count} approved variants; protocol requires {min_variants}.")
            answers = row.get("candidate_answers") or row.get("answers")
            if not answers or len(set(answers)) < 2:
                raise ValueError(f"{sample_id}: provide >=2 fixed candidate_answers (including plausible alternatives).")
            started = time.perf_counter()
            try:
                result = run_pathway_instance(model, image, row["question"], mask, list(dict.fromkeys(answers)), variants,
                    control_masks=control_masks,
                    top_k=int(config.get("top_k", 20)),
                    recovery_threshold=float(config.get("recovery_threshold", .8)),
                    intervention_method=config.get("intervention_method", "blur"))
                result.update({"sample_id": sample_id, "split": row.get("split"),
                    "question_type": row.get("question_type"), "gold_answers": row.get("answers"),
                    "vqa_consensus_score": vqa_soft_score(result["baseline_answer"], row.get("answers")),
                    "variant_vqa_scores": {item["transform"]: vqa_soft_score(item["answer"], row.get("answers"))
                                           for item in result["variant_pathways"]},
                    "semantic_variant_failure": any(not item["answer_matches_baseline"]
                                                     for item in result["variant_pathways"] if item["transform"] != "original"),
                    "wall_seconds": time.perf_counter() - started,
                    "runtime": runtime_metadata,
                    "checkpoints": {
                        "qwen": {"model_id": config["qwen_model"], "revision": model.resolved_revision},
                        "grounding_dino": {"model_id": row.get("evidence_proposal_models", {}).get("grounding_dino"),
                            "revision": row.get("evidence_proposal_revisions", {}).get("grounding_dino")},
                        "sam2": {"model_id": row.get("evidence_proposal_models", {}).get("sam2"),
                            "revision": row.get("evidence_proposal_revisions", {}).get("sam2")},
                        "clip": {"model_id": row.get("evidence_proposal_models", {}).get("clip"),
                            "revision": row.get("evidence_proposal_revisions", {}).get("clip")},
                    },
                    "protocol": {"intervention_method": config.get("intervention_method", "blur"),
                        "top_k": int(config.get("top_k", 20)),
                        "recovery_threshold": float(config.get("recovery_threshold", .8)),
                        "min_accepted_variants": int(config.get("min_accepted_variants", 4)),
                        "min_reviewed_controls": int(config.get("min_reviewed_controls", 1)),
                        "max_image_pixels": config.get("max_image_pixels", 1_003_520),
                        "candidate_answers": list(dict.fromkeys(answers)),
                        "question_type": row.get("question_type"), "dataset": row.get("dataset"),
                        "prompt_format": "single user message with image and question; deterministic decoding"},
                    "data_provenance": {
                        "source_image": row.get("image"), "evidence_mask": row.get("evidence_mask"),
                        "critical_concepts": row.get("critical_concepts"),
                        "evidence_proposals": row.get("evidence_proposals"),
                        "evidence_proposal_models": row.get("evidence_proposal_models"),
                        "evidence_proposal_revisions": row.get("evidence_proposal_revisions"),
                        "evidence_proposal_thresholds": row.get("evidence_proposal_thresholds"),
                        "evidence_mask_review": row.get("evidence_mask_review"),
                        "reviewed_variants": [{"transform": v.get("transform"), "image": v.get("image"),
                            "evidence_mask": v.get("evidence_mask"), "review_flags": {
                                key: v.get(key) for key in ("answer_preserved", "critical_evidence_preserved",
                                                            "relations_preserved", "human_audited")}}
                            for v in row.get("variants", []) if all(v.get(k) is True for k in (
                                "answer_preserved", "critical_evidence_preserved", "relations_preserved", "human_audited"))],
                        "reviewed_control_masks": [{"control_index": c.get("control_index"), "image": c.get("image"),
                            "review": c.get("review")} for c in row.get("control_masks", [])
                            if c.get("reviewed") is True and c.get("accepted") is True]}})
                sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                sink.flush()
                count += 1
                print(f"[{count}] {sample_id}: SCC={result['scc']}")
            except Exception as exc:
                failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAILED {sample_id}: {type(exc).__name__}: {exc}")
                if config.get("fail_fast", True):
                    raise
            if limit is not None and count >= limit:
                break
    failure_path = output_dir / "semantic_pathway_failures.json"
    failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Wrote results to {output_path}; new completed rows: {count}; failures: {len(failures)}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiment.json"))
    parser.add_argument("--limit", type=int, default=None, help="Limit newly processed rows for a smoke run")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_manifest(config, limit=args.limit, resume=not args.no_resume)


if __name__ == "__main__":
    main()
