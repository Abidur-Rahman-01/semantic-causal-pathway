"""Run the configured reviewed pilot, then write its summary report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_experiment import read_rows, summarize
from semantic_circuits.transforms import transformation_family


def _resolve(value: str, manifest: Path, data_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    beside_manifest = manifest.parent / path
    return beside_manifest if beside_manifest.exists() else data_root / path


def check_ready(config: dict, limit: int | None, resume: bool = True) -> tuple[Path, Path, list[dict]]:
    manifest = Path(config["manifest"])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    data_root = Path(config.get("data_root", "data"))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Reviewed manifest not found: {manifest}\n"
            "Complete evidence and variant reviews first; see the README's Pilot workflow."
        )
    all_rows = list(read_rows(manifest))
    target_images = config.get("target_independent_images")
    if target_images is not None:
        image_groups = {f"{row.get('dataset', '')}:{row.get('image_id', row.get('sample_id'))}"
                        for row in all_rows}
        if len(image_groups) != int(target_images):
            raise ValueError(f"Confirmatory manifest has {len(image_groups)} distinct image groups; "
                             f"expected {target_images}. Sample images before annotation.")
    if config.get("heldout_failure_eval"):
        split_groups = {}
        for row in all_rows:
            group = f"{row.get('dataset', '')}:{row.get('image_id', row.get('sample_id'))}"
            analysis_split = row.get("analysis_split")
            if analysis_split not in {"train", "validation", "test"}:
                raise ValueError(f"{row.get('sample_id')}: run sample_pathway_images.py to assign analysis_split")
            if group in split_groups and split_groups[group] != analysis_split:
                raise ValueError(f"Image-group leakage: {group} occurs in multiple analysis splits")
            split_groups[group] = analysis_split
        if set(split_groups.values()) != {"train", "validation", "test"}:
            raise ValueError("Held-out prediction requires image groups in train, validation, and test splits")
    output_dir = Path(config.get("output_dir", "outputs"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_path = output_dir / "semantic_pathway_results.jsonl"
    completed = set()
    if resume and output_path.exists():
        completed = {str(row.get("sample_id")) for row in read_rows(output_path)}
    rows = [row for row in all_rows if str(row.get("sample_id")) not in completed]
    if limit is None:
        limit = int(config.get("max_samples") or len(rows))
    rows = rows[:limit]
    if not rows and not all_rows:
        raise ValueError(f"No manifest rows selected from {manifest}")
    min_variants = int(config.get("min_accepted_variants", 4))
    required = ("answer_preserved", "critical_evidence_preserved", "relations_preserved", "human_audited")
    errors = []
    for row in rows:
        sid = row.get("sample_id", "<missing sample_id>")
        if row.get("split") == "test" and config.get("allow_test_split") is not True:
            errors.append(f"{sid}: locked test split requires allow_test_split=true")
        if (row.get("evidence_mask_reviewed") is not True
                or not str((row.get("evidence_mask_review") or {}).get("reviewer", "")).strip()
                or not str((row.get("evidence_mask_review") or {}).get("review_date", "")).strip()):
            errors.append(f"{sid}: evidence_mask_reviewed is not true")
        for field in ("image", "evidence_mask"):
            if not row.get(field):
                errors.append(f"{sid}: missing {field}")
            elif not _resolve(row[field], manifest, data_root).is_file():
                errors.append(f"{sid}: {field} file not found ({row[field]})")
        accepted = [v for v in row.get("variants", []) if all(v.get(k) is True for k in required)
                    and str(v.get("reviewer", "")).strip() and str(v.get("review_date", "")).strip()]
        if len(accepted) < min_variants:
            errors.append(f"{sid}: {len(accepted)} approved variants; need {min_variants}")
        if config.get("heldout_failure_eval"):
            roles = {v.get("analysis_role") for v in accepted}
            probe_rows = [v for v in accepted if v.get("analysis_role") == "probe"]
            heldout_rows = [v for v in accepted if v.get("analysis_role") == "heldout"]
            if roles - {"probe", "heldout"} or len(probe_rows) != 2 or len(heldout_rows) != 2:
                errors.append(f"{sid}: assign exactly two approved variants each to analysis_role=probe and heldout")
            family = lambda v: transformation_family(v.get("transform", ""))
            probe_families = {family(v) for v in accepted if v.get("analysis_role") == "probe"}
            heldout_families = {family(v) for v in accepted if v.get("analysis_role") == "heldout"}
            if probe_families & heldout_families:
                errors.append(f"{sid}: probe and heldout variants must use disjoint transform families")
        for variant in accepted:
            if not variant.get("image") or not _resolve(variant["image"], manifest, data_root).is_file():
                errors.append(f"{sid}: missing image for approved variant {variant.get('transform', '<unnamed>')}")
            variant_mask = variant.get("evidence_mask")
            if variant_mask and not _resolve(variant_mask, manifest, data_root).is_file():
                errors.append(f"{sid}: missing evidence mask for approved variant {variant.get('transform', '<unnamed>')}")
        answers = row.get("candidate_answers") or row.get("answers") or []
        if len(set(answers)) < 2:
            errors.append(f"{sid}: provide at least two fixed candidate_answers")
        if row.get("candidate_answers_protocol") != "annotate_before_model_inference_without_consulting_answers":
            errors.append(f"{sid}: record candidate_answers_protocol after freezing the outcome bins before model inference")
        answer_review = row.get("candidate_answers_review") or {}
        if not str(answer_review.get("reviewer", "")).strip() or not str(answer_review.get("review_date", "")).strip():
            errors.append(f"{sid}: candidate answer bins need reviewer and review_date metadata")
        reviewed_controls = [c for c in row.get("control_masks", []) if c.get("reviewed") is True and c.get("accepted") is True
                             and str((c.get("review") or {}).get("reviewer", "")).strip()
                             and str((c.get("review") or {}).get("review_date", "")).strip()]
        if config.get("require_reviewed_controls", False):
            min_controls = int(config.get("min_reviewed_controls", 1))
            if len(reviewed_controls) < min_controls:
                errors.append(f"{sid}: {len(reviewed_controls)} reviewed spatial controls; need {min_controls}")
            for control in reviewed_controls:
                if not control.get("image") or not _resolve(control["image"], manifest, data_root).is_file():
                    errors.append(f"{sid}: missing reviewed control mask {control.get('control_index', '<unnamed>')}")
        try:
            image_path = _resolve(row["image"], manifest, data_root)
            mask_path = _resolve(row["evidence_mask"], manifest, data_root)
            with Image.open(image_path) as im, Image.open(mask_path) as mask:
                image_size = im.size
                if mask.size != image_size:
                    errors.append(f"{sid}: evidence mask size {mask.size} does not match image {image_size}")
            for variant in accepted:
                with Image.open(_resolve(variant["image"], manifest, data_root)) as im:
                    if im.size != image_size:
                        errors.append(f"{sid}: variant {variant['transform']} size {im.size} does not match image {image_size}")
                if variant.get("evidence_mask"):
                    with Image.open(_resolve(variant["evidence_mask"], manifest, data_root)) as mask:
                        if mask.size != image_size:
                            errors.append(f"{sid}: evidence mask for {variant['transform']} has size {mask.size}, expected {image_size}")
            for control in reviewed_controls:
                with Image.open(_resolve(control["image"], manifest, data_root)) as im:
                    if im.size != image_size:
                        errors.append(f"{sid}: control mask {control['control_index']} size {im.size} does not match image {image_size}")
        except Exception as exc:
            errors.append(f"{sid}: could not read image/mask/variant: {type(exc).__name__}: {exc}")
    if errors:
        raise ValueError("Pilot preflight failed:\n- " + "\n- ".join(errors))
    return manifest, output_dir, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--limit", type=int, default=None, help="Maximum new samples; defaults to config max_samples")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest, output_dir, rows = check_ready(config, args.limit, resume=False)
    if not rows:
        raise ValueError("No manifest rows selected.")
    print(f"Preflight passed for {len(rows)} sample(s) in {manifest}", flush=True)

    from run_experiment import run_manifest
    model_specs = config.get("models") or [{"name": "primary", "architecture": config.get("architecture", "qwen2_5_vl"),
                                             "model_id": config["qwen_model"], "revision": config.get("qwen_revision")}]
    for spec in model_specs:
        model_name = spec["name"]
        model_config = {**config, "qwen_model": spec["model_id"],
                        "qwen_revision": spec.get("revision"), "architecture": spec["architecture"],
                        "output_dir": str(output_dir / "models" / model_name)}
        model_output = Path(model_config["output_dir"])
        result_path = run_manifest(model_config, limit=args.limit, resume=not args.no_resume)
        if not result_path.is_file():
            raise FileNotFoundError(f"Experiment produced no result file: {result_path}")
        report = summarize(list(read_rows(result_path)))
        report.update({"study_stage": config.get("study_stage"), "study_claim": config.get("study_claim"),
                       "core_novelty": config.get("core_novelty"), "primary_metric": config.get("primary_metric"),
                       "secondary_metrics": config.get("secondary_metrics", []), "model": spec})
        summary_path = model_output / "semantic_pathway_summary.json"
        summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Completed {model_name}: {result_path}", flush=True)

    if config.get("heldout_failure_eval"):
        from analyze_model_comparison import analyze
        comparison = analyze(output_dir, failure_vqa_score_threshold=float(config.get("failure_vqa_score_threshold", 0.5)))
        print(f"Held-out prediction report: {comparison}")


if __name__ == "__main__":
    main()
