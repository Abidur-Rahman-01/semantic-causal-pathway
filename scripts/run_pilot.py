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
        if row.get("evidence_mask_reviewed") is not True:
            errors.append(f"{sid}: evidence_mask_reviewed is not true")
        for field in ("image", "evidence_mask"):
            if not row.get(field):
                errors.append(f"{sid}: missing {field}")
            elif not _resolve(row[field], manifest, data_root).is_file():
                errors.append(f"{sid}: {field} file not found ({row[field]})")
        accepted = [v for v in row.get("variants", []) if all(v.get(k) is True for k in required)]
        if len(accepted) < min_variants:
            errors.append(f"{sid}: {len(accepted)} approved variants; need {min_variants}")
        for variant in accepted:
            if not variant.get("image") or not _resolve(variant["image"], manifest, data_root).is_file():
                errors.append(f"{sid}: missing image for approved variant {variant.get('transform', '<unnamed>')}")
            variant_mask = variant.get("evidence_mask")
            if variant_mask and not _resolve(variant_mask, manifest, data_root).is_file():
                errors.append(f"{sid}: missing evidence mask for approved variant {variant.get('transform', '<unnamed>')}")
        answers = row.get("candidate_answers") or row.get("answers") or []
        if len(set(answers)) < 2:
            errors.append(f"{sid}: provide at least two fixed candidate_answers")
        reviewed_controls = [c for c in row.get("control_masks", []) if c.get("reviewed") is True and c.get("accepted") is True]
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
    manifest, output_dir, rows = check_ready(config, args.limit, resume=not args.no_resume)
    result_path = output_dir / "semantic_pathway_results.jsonl"
    if not rows:
        if not result_path.is_file():
            raise ValueError("No unfinished samples and no result file were found.")
        print("All samples are already complete; refreshing the summary only.")
        report = summarize(list(read_rows(result_path)))
        summary_path = output_dir / "semantic_pathway_summary.json"
        summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Summary: {summary_path}")
        return
    print(f"Preflight passed for {len(rows)} sample(s) in {manifest}", flush=True)

    from run_experiment import run_manifest
    run_manifest(config, limit=args.limit, resume=not args.no_resume)

    if not result_path.exists():
        raise FileNotFoundError(f"Experiment produced no result file: {result_path}")
    report = summarize(list(read_rows(result_path)))
    summary_path = output_dir / "semantic_pathway_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nPilot complete. Results: {result_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
