"""Generate area/shape-matched spatial control masks and a review sheet."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semantic_circuits.controls import sample_translated_control_masks


def resolve(value: str, manifest: Path, data_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    beside_manifest = manifest.parent / path
    return beside_manifest if beside_manifest.exists() else data_root / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/controls"))
    parser.add_argument("--manifest-out", type=Path, default=Path("data/manifest_with_controls.jsonl"))
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    manifest, data_root, output = args.manifest.resolve(), args.data_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    review_rows = []
    with manifest.open(encoding="utf-8-sig") as src, args.manifest_out.open("w", encoding="utf-8") as dst:
        for row_index, line in enumerate(src):
            if not line.strip():
                continue
            if args.limit is not None and row_index >= args.limit:
                break
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if row.get("evidence_mask_reviewed") is not True or not row.get("evidence_mask"):
                raise ValueError(f"{sample_id}: review the evidence mask before generating spatial controls")
            mask_path = resolve(row["evidence_mask"], manifest, data_root)
            target = np.asarray(Image.open(mask_path).convert("L")) > 0
            image_path = resolve(row["image"], manifest, data_root)
            with Image.open(image_path) as image:
                if image.size != (target.shape[1], target.shape[0]):
                    raise ValueError(f"{sample_id}: evidence mask dimensions do not match source image")
            controls = sample_translated_control_masks(target, n=args.count,
                seed=args.seed + int(row_index))
            records = []
            for index, control in enumerate(controls):
                filename = f"{sample_id}__control_{index:02d}.png"
                target_path = output / filename
                Image.fromarray(control.astype("uint8") * 255, mode="L").save(target_path)
                try:
                    portable_path = target_path.resolve().relative_to(data_root)
                except ValueError:
                    portable_path = target_path.resolve()
                records.append({"control_index": index, "image": str(portable_path),
                                "reviewed": False, "accepted": False})
                review_rows.append({"sample_id": sample_id, "control_index": index,
                    "mask": str(target_path.resolve()), "accepted": "", "reviewer": "", "review_date": ""})
            row["control_masks"] = records
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    review_path = output / "control_review.csv"
    with review_path.open("w", newline="", encoding="utf-8-sig") as stream:
        columns = list(review_rows[0]) if review_rows else ["sample_id", "control_index", "mask", "accepted", "reviewer", "review_date"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"Wrote unreviewed control masks to {args.manifest_out} and {review_path}")
    print("Reject any control that overlaps other question-critical evidence or an important object.")


if __name__ == "__main__":
    main()
