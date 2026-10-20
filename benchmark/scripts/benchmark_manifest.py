"""Validate and deterministically split Image Evidence Stability benchmark manifests.

Commands are intentionally separate from model inference. Run validation and freeze
the resulting manifest/config hashes before invoking a model runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VALID_SPLITS = {"train", "validation", "test"}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: each JSONL record must be an object")
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def resolve_asset(value: str, manifest: Path, data_root: Path | None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [manifest.parent / path]
    if data_root is not None:
        candidates.append(data_root / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def asset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _err(errors: list[str], row: dict, message: str) -> None:
    errors.append(f"line {row.get('_line_no', '?')} ({row.get('sample_id', '?')}): {message}")


def _valid_iso_date(value) -> bool:
    try:
        date.fromisoformat(str(value))
        return True
    except (TypeError, ValueError):
        return False


def infer_transform_family(name: str | None) -> str:
    value = str(name or "").lower()
    if value.startswith("jpeg_quality_"):
        return "jpeg_compression"
    if value.startswith("translate_"):
        return "translation"
    if value.startswith(("brightness_", "contrast_")):
        return "photometric"
    if value.startswith("background_"):
        return "background_edit"
    if value.startswith("remove_noncritical_"):
        return "object_removal"
    return "unclassified"


def validate_rows(rows: list[dict], manifest: Path, data_root: Path | None,
                  require_hashes: bool = False, require_schema_v1: bool = False,
                  require_assets: bool = False) -> tuple[list[str], dict]:
    errors: list[str] = []
    sample_ids, variant_ids = set(), set()
    image_to_split: dict[str, str] = {}
    image_hash_to_split: dict[str, str] = {}
    family_roles: dict[str, set[str]] = defaultdict(set)
    split_counts, dataset_counts = Counter(), Counter()
    accepted_counts = Counter()
    for row in rows:
        modern = row.get("schema_version") == "1.0"
        if require_schema_v1 and not modern:
            _err(errors, row, "schema_version must be '1.0'")
        required = (("schema_version", "sample_id", "image_id", "dataset", "dataset_split", "image",
                     "question", "gold_answers", "candidate_answers", "candidate_answers_protocol",
                     "candidate_answers_reviewer", "candidate_answers_review_date", "analysis_split",
                     "evidence", "variants") if modern else
                    ("sample_id", "image_id", "dataset", "question", "analysis_split", "variants"))
        for key in required:
            if key not in row:
                _err(errors, row, f"missing required field {key!r}")
        sid = str(row.get("sample_id", ""))
        if not sid or sid in sample_ids:
            _err(errors, row, "sample_id is empty or duplicated")
        sample_ids.add(sid)
        split = row.get("analysis_split")
        if split not in VALID_SPLITS:
            _err(errors, row, f"analysis_split must be one of {sorted(VALID_SPLITS)}")
        split_counts[str(split)] += 1
        dataset_counts[(str(row.get("dataset", "unknown")), str(split))] += 1
        image_id = str(row.get("image_id", ""))
        if image_id in image_to_split and image_to_split[image_id] != split:
            _err(errors, row, f"image_id {image_id!r} appears in multiple analysis splits")
        image_to_split[image_id] = str(split)
        answers = row.get("candidate_answers")
        if not isinstance(answers, list) or len(answers) < 2 or len(set(answers)) != len(answers):
            _err(errors, row, "candidate_answers must contain at least two unique frozen answer strings")
        elif any(not isinstance(answer, str) or not answer.strip() for answer in answers):
            _err(errors, row, "candidate_answers must contain non-empty strings")
        if modern:
            if not isinstance(row.get("dataset_split"), str) or not row["dataset_split"].strip():
                _err(errors, row, "dataset_split must be a non-empty string")
            if not isinstance(row.get("dataset"), str) or not row["dataset"].strip():
                _err(errors, row, "dataset must be a non-empty string")
            img_val = row.get("image") or row.get("image_path")
            if not isinstance(img_val, str) or not img_val.strip():
                _err(errors, row, "image must be a non-empty path")
            if not isinstance(row.get("gold_answers"), list):
                _err(errors, row, "gold_answers must be an array")
            elif any(not isinstance(answer, str) for answer in row.get("gold_answers", [])):
                _err(errors, row, "gold_answers entries must be strings")
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                _err(errors, row, "question must be a non-empty string")
            if not row.get("candidate_answers_protocol") or not row.get("candidate_answers_reviewer") or not row.get("candidate_answers_review_date"):
                _err(errors, row, "candidate-answer bins need protocol, reviewer, and review date metadata")
            elif not _valid_iso_date(row.get("candidate_answers_review_date")):
                _err(errors, row, "candidate_answers_review_date must be ISO YYYY-MM-DD")
            evidence = row.get("evidence") or {}
            if evidence.get("accepted") is not True or not evidence.get("mask_path"):
                _err(errors, row, "evidence mask must be explicitly accepted and have mask_path")
            if not evidence.get("reviewer_ids"):
                _err(errors, row, "evidence review must record reviewer_ids")
            if not _valid_iso_date(evidence.get("review_date")):
                _err(errors, row, "evidence review_date must be ISO YYYY-MM-DD")
            if not isinstance(evidence.get("reviewer_ids"), list) or any(not isinstance(x, str) or not x.strip() for x in evidence.get("reviewer_ids", [])):
                _err(errors, row, "evidence reviewer_ids must be a non-empty string array")
        image_ref = row.get("image") or row.get("image_path")
        if modern:
            image_ref = row.get("image") or row.get("image_path")
        if image_ref:
            image_path = resolve_asset(str(image_ref), manifest, data_root)
            if not image_path.is_file():
                if require_assets:
                    _err(errors, row, f"image file not found: {image_path}")
            else:
                digest = asset_hash(image_path)
                declared = row.get("image_sha256")
                if require_hashes and not declared:
                    _err(errors, row, "image_sha256 is required")
                if declared and declared.lower() != digest:
                    _err(errors, row, "image_sha256 does not match image bytes")
                prior_split = image_hash_to_split.get(digest)
                if prior_split and prior_split != split:
                    _err(errors, row, f"duplicate image content hash crosses {prior_split}/{split} splits")
                image_hash_to_split[digest] = str(split)
                try:
                    with Image.open(image_path) as im:
                        image_size = im.size
                except Exception as exc:
                    _err(errors, row, f"cannot read image: {exc}")
                    image_size = None
                evidence = row.get("evidence") if modern else None
                mask_ref = evidence.get("mask_path") if evidence else row.get("evidence_mask")
                if mask_ref:
                    mask_path = resolve_asset(str(mask_ref), manifest, data_root)
                    if not mask_path.is_file():
                        if require_assets:
                            _err(errors, row, f"evidence mask file not found: {mask_path}")
                    elif image_size:
                        try:
                            with Image.open(mask_path) as mask:
                                if mask.size != image_size:
                                    _err(errors, row, f"evidence mask size {mask.size} != image size {image_size}")
                        except Exception as exc:
                            _err(errors, row, f"cannot read evidence mask: {exc}")
                for control in row.get("control_masks", []):
                    if modern and (not isinstance(control, dict) or control.get("accepted") is not True or
                                   not control.get("reviewer_id") or not _valid_iso_date(control.get("review_date"))):
                        _err(errors, row, "schema-v1 control masks need explicit acceptance, reviewer_id, and review_date")
                    control_ref = control if isinstance(control, str) else (control.get("mask_path") or control.get("image") or control.get("path"))
                    if not control_ref:
                        _err(errors, row, "control mask has no path")
                        continue
                    control_path = resolve_asset(str(control_ref), manifest, data_root)
                    if not control_path.is_file():
                        if require_assets:
                            _err(errors, row, f"control mask file not found: {control_path}")
                        continue
                    try:
                        with Image.open(control_path) as control_image:
                            if control_image.size != image_size:
                                _err(errors, row, f"control mask size {control_image.size} != image size {image_size}")
                    except Exception as exc:
                        _err(errors, row, f"cannot read control mask: {exc}")
        variants = row.get("variants", [])
        if not isinstance(variants, list):
            _err(errors, row, "variants must be an array")
            continue
        probe_families, heldout_families = set(), set()
        approved_roles = Counter()
        for variant in variants:
            if not isinstance(variant, dict):
                _err(errors, row, "variant must be an object")
                continue
            vid = str(variant.get("variant_id", variant.get("transform", "")))
            if modern and not vid:
                _err(errors, row, "variant_id is required")
            if vid:
                key = vid if modern else f"{sid}:{vid}"
                if key in variant_ids:
                    _err(errors, row, f"duplicate variant ID {vid!r}")
                variant_ids.add(key)
            role = variant.get("role", variant.get("analysis_role"))
            family = variant.get("transform_family", variant.get("family")) or infer_transform_family(variant.get("transform_name", variant.get("transform")))
            approved = variant.get("accepted") is True if modern else all(variant.get(k) is True for k in
                ("answer_preserved", "critical_evidence_preserved", "relations_preserved", "human_audited"))
            if approved:
                accepted_counts[(str(split), str(role))] += 1
                approved_roles[str(role)] += 1
            if role in {"probe", "heldout"}:
                family_roles[str(family)].add(role)
                (probe_families if role == "probe" else heldout_families).add(family)
            if modern:
                for field in ("answer_preserved", "evidence_preserved", "relations_preserved", "artifact_introduced", "human_audited", "accepted"):
                    if not isinstance(variant.get(field), bool):
                        _err(errors, row, f"variant {vid!r} field {field!r} must be boolean")
                if role not in {"probe", "heldout"}:
                    _err(errors, row, f"variant {vid!r} role must be probe or heldout")
                if not family or not variant.get("transform_name"):
                    _err(errors, row, f"variant {vid!r} needs transform_family and transform_name")
                if not variant.get("reviewer_id") or not variant.get("review_date"):
                    _err(errors, row, f"variant {vid!r} lacks reviewer_id or review_date")
                elif not _valid_iso_date(variant.get("review_date")):
                    _err(errors, row, f"variant {vid!r} review_date must be ISO YYYY-MM-DD")
                if approved and (variant.get("artifact_introduced") is True or not all(
                        variant.get(k) is True for k in ("answer_preserved", "evidence_preserved", "relations_preserved", "human_audited"))):
                    _err(errors, row, f"variant {vid!r} is approved despite a failed review gate")
                if approved and not variant.get("reviewer_id"):
                    _err(errors, row, f"approved variant {vid!r} needs reviewer_id")
                if not approved and not variant.get("rejection_reason"):
                    _err(errors, row, f"rejected variant {vid!r} needs rejection_reason")
            path_ref = variant.get("image_path", variant.get("image"))
            if modern and (not isinstance(path_ref, str) or not path_ref.strip()):
                _err(errors, row, f"variant {vid!r} image_path must be a non-empty path")
            if path_ref:
                variant_path = resolve_asset(str(path_ref), manifest, data_root)
                if not variant_path.is_file():
                    if require_assets:
                        _err(errors, row, f"variant image not found: {variant_path}")
                elif image_size:
                    try:
                        variant_digest = asset_hash(variant_path)
                        declared_variant_digest = variant.get("image_sha256")
                        if require_hashes and not declared_variant_digest:
                            _err(errors, row, f"variant {vid!r} is missing image_sha256")
                        if declared_variant_digest and declared_variant_digest.lower() != variant_digest:
                            _err(errors, row, f"variant {vid!r} image_sha256 does not match image bytes")
                        with Image.open(variant_path) as im:
                            variant_size = im.size
                        mask_ref = variant.get("mask_path", variant.get("evidence_mask"))
                        if mask_ref:
                            mask_path = resolve_asset(str(mask_ref), manifest, data_root)
                            if not mask_path.is_file():
                                _err(errors, row, f"variant mask file not found: {mask_path}")
                            else:
                                with Image.open(mask_path) as mask:
                                    if mask.size != variant_size:
                                        _err(errors, row, f"variant mask size {mask.size} != variant image size {variant_size}")
                    except Exception as exc:
                        _err(errors, row, f"cannot inspect variant dimensions: {exc}")
        if probe_families & heldout_families:
            _err(errors, row, f"probe and heldout transformation families overlap: {sorted(probe_families & heldout_families)}")
        if modern:
            if approved_roles["probe"] < 2:
                _err(errors, row, "at least two approved probe variants are required")
            if approved_roles["heldout"] < 2:
                _err(errors, row, "at least two approved heldout variants are required")
            controls = row.get("control_masks", [])
            if len(controls) < 3:
                _err(errors, row, "at least three reviewed spatial controls are required for the white-box benchmark")
    for family, roles in family_roles.items():
        if len(roles) > 1:
            errors.append(f"transformation family {family!r} is assigned to both probe and heldout roles")
    summary = {
        "rows": len(rows), "unique_image_ids": len(image_to_split), "unique_image_hashes": len(image_hash_to_split),
        "split_counts": dict(split_counts),
        "dataset_split_counts": {f"{d}:{s}": n for (d, s), n in sorted(dataset_counts.items())},
        "approved_variant_counts_by_split_role": {f"{s}:{r}": n for (s, r), n in sorted(accepted_counts.items())},
        "errors": len(errors)
    }
    return errors, summary


def assign_group_splits(rows: list[dict], seed: int, ratios: dict[str, float]) -> list[dict]:
    if set(ratios) != VALID_SPLITS or any(v <= 0 for v in ratios.values()) or abs(sum(ratios.values()) - 1) > 1e-9:
        raise ValueError("split ratios must define positive train/validation/test values summing to 1")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if "image_id" not in row:
            raise ValueError(f"row {row.get('sample_id')} has no image_id")
        grouped[str(row["image_id"])].append(row)
    # Deterministic ordering and largest-remainder allocation independently per dataset.
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for image_id, group in grouped.items():
        datasets = {str(row.get("dataset", "unknown")) for row in group}
        if len(datasets) != 1:
            raise ValueError(f"image_id {image_id} appears under multiple datasets")
        by_dataset[next(iter(datasets))].append(image_id)
    out = []
    for dataset in sorted(by_dataset):
        ids = sorted(by_dataset[dataset], key=lambda x: hashlib.sha256(f"{seed}:{dataset}:{x}".encode()).hexdigest())
        n = len(ids)
        exact = {split: n * ratio for split, ratio in ratios.items()}
        counts = {split: int(exact[split]) for split in ratios}
        for split in sorted(ratios, key=lambda k: (-(exact[k] - counts[k]), k))[: n - sum(counts.values())]:
            counts[split] += 1
        cursor = 0
        assignment = {}
        for split in ("train", "validation", "test"):
            for image_id in ids[cursor:cursor + counts[split]]:
                assignment[image_id] = split
            cursor += counts[split]
        for image_id in ids:
            for row in grouped[image_id]:
                copied = {k: v for k, v in row.items() if k != "_line_no"}
                copied["analysis_split"] = assignment[image_id]
                out.append(copied)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_manifest_row(row: dict) -> dict:
    """Adapt schema v1 records to the existing inference runner's field names."""
    if row.get("schema_version") != "1.0":
        return row
    result = {k: v for k, v in row.items() if k != "_line_no"}
    evidence = result.get("evidence") or {}
    result["evidence_mask"] = evidence.get("mask_path")
    result["answers"] = result.get("gold_answers", result.get("answers"))
    result["split"] = result.get("dataset_split", result.get("split"))
    variants = []
    for item in result.get("variants", []):
        variant = dict(item)
        variant.update({
            "transform": item.get("transform_name"),
            "family": item.get("transform_family", item.get("family")) or infer_transform_family(item.get("transform_name", item.get("transform"))),
            "analysis_role": item.get("role"),
            "image": item.get("image_path"),
            "evidence_mask": item.get("mask_path"),
            "critical_evidence_preserved": item.get("evidence_preserved"),
            "reviewer": item.get("reviewer_id"),
        })
        variants.append(variant)
    result["variants"] = variants
    controls = result.get("control_masks", [])
    result["control_masks"] = [({"image": item} if isinstance(item, str) else
                                 {**item, "image": item.get("image", item.get("mask_path"))})
                                for item in controls]
    return result


def seal_hashes(rows: list[dict], manifest: Path, data_root: Path | None) -> list[dict]:
    """Add byte hashes for original and variant images before model inference."""
    sealed = []
    for row in rows:
        result = {k: v for k, v in row.items() if k != "_line_no"}
        original = resolve_asset(str(result["image"]), manifest, data_root)
        if not original.is_file():
            raise FileNotFoundError(original)
        result["image_sha256"] = asset_hash(original)
        for variant in result.get("variants", []):
            ref = variant.get("image_path", variant.get("image"))
            if ref:
                path = resolve_asset(str(ref), manifest, data_root)
                if not path.is_file():
                    raise FileNotFoundError(path)
                variant["image_sha256"] = asset_hash(path)
        sealed.append(result)
    return sealed


def rebase_asset_paths(rows: list[dict], source_manifest: Path, output_manifest: Path,
                       data_root: Path | None) -> list[dict]:
    """Keep relative assets resolvable when a derived manifest is written elsewhere."""
    def rewrite(value):
        if not value or Path(value).is_absolute():
            return value
        resolved = resolve_asset(str(value), source_manifest, data_root)
        return os.path.relpath(resolved, output_manifest.parent)
    updated = []
    for row in rows:
        item = {k: v for k, v in row.items() if k != "_line_no"}
        if item.get("image"):
            item["image"] = rewrite(item["image"])
        if item.get("evidence_mask"):
            item["evidence_mask"] = rewrite(item["evidence_mask"])
        if isinstance(item.get("evidence"), dict) and item["evidence"].get("mask_path"):
            item["evidence"]["mask_path"] = rewrite(item["evidence"]["mask_path"])
        for variant in item.get("variants", []):
            for key in ("image", "image_path", "evidence_mask", "mask_path"):
                if variant.get(key):
                    variant[key] = rewrite(variant[key])
        controls = []
        for control in item.get("control_masks", []):
            if isinstance(control, str):
                controls.append(rewrite(control))
            elif isinstance(control, dict):
                control = dict(control)
                for key in ("image", "mask_path", "path"):
                    if control.get(key):
                        control[key] = rewrite(control[key])
                controls.append(control)
            else:
                controls.append(control)
        if "control_masks" in item:
            item["control_masks"] = controls
        updated.append(item)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate IDs, split isolation, reviews, assets, and mask dimensions")
    validate.add_argument("manifest")
    validate.add_argument("--data-root")
    validate.add_argument("--require-schema-v1", action="store_true")
    validate.add_argument("--require-hashes", action="store_true")
    validate.add_argument("--require-assets", action="store_true")
    validate.add_argument("--summary-out")
    split = sub.add_parser("split", help="Assign deterministic image-grouped splits before inference")
    split.add_argument("manifest")
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=17)
    split.add_argument("--train", type=float, default=.6)
    split.add_argument("--validation", type=float, default=.2)
    split.add_argument("--test", type=float, default=.2)
    split.add_argument("--summary-out")
    split.add_argument("--data-root")
    seal = sub.add_parser("seal", help="Add image byte hashes to a frozen manifest before inference")
    seal.add_argument("manifest")
    seal.add_argument("--data-root")
    seal.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    rows = read_jsonl(manifest)
    if args.command == "validate":
        data_root = Path(args.data_root).resolve() if args.data_root else None
        errors, summary = validate_rows(rows, manifest, data_root, args.require_hashes, args.require_schema_v1, args.require_assets)
        print(json.dumps(summary, indent=2))
        if args.summary_out:
            Path(args.summary_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            raise SystemExit(1)
    elif args.command == "split":
        ratios = {"train": args.train, "validation": args.validation, "test": args.test}
        assigned = assign_group_splits(rows, args.seed, ratios)
        out = Path(args.output).resolve()
        data_root = Path(args.data_root).resolve() if args.data_root else None
        assigned = rebase_asset_paths(assigned, manifest, out, data_root)
        write_jsonl(out, assigned)
        errors, summary = validate_rows(read_jsonl(out), out, None)
        if args.summary_out:
            Path(args.summary_out).write_text(json.dumps({"seed": args.seed, "ratios": ratios, **summary}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(out), "seed": args.seed, **summary}, indent=2))
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            raise SystemExit(1)
    else:
        data_root = Path(args.data_root).resolve() if args.data_root else None
        sealed = seal_hashes(rows, manifest, data_root)
        out = Path(args.output).resolve()
        sealed = rebase_asset_paths(sealed, manifest, out, data_root)
        write_jsonl(out, sealed)
        errors, summary = validate_rows(read_jsonl(out), out, data_root, require_hashes=True)
        print(json.dumps({"output": str(out), **summary}, indent=2))
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
