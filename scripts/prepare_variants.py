"""Create deterministic nuisance candidates and a human review sheet from JSONL."""
import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semantic_circuits.transforms import generate_semantic_variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Input manifest JSONL")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="data/variants")
    parser.add_argument("--manifest-out", default="data/manifest_with_variants.jsonl")
    parser.add_argument("--photometric-only", action="store_true",
                        help="Generate only photometric candidates, without masks")
    args = parser.parse_args()
    manifest, data_root, output = map(Path, (args.manifest, args.data_root, args.output))
    manifest = manifest.resolve()
    data_root = data_root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(args.manifest_out)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    review_rows = []
    with manifest.open(encoding="utf-8-sig") as src, out_manifest.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = manifest.parent / image_path if (manifest.parent / image_path).exists() else data_root / image_path
            image = Image.open(image_path).convert("RGB")

            def resolve(value):
                candidate = Path(value)
                if candidate.is_absolute():
                    return candidate
                from_manifest = manifest.parent / candidate
                return from_manifest if from_manifest.exists() else data_root / candidate

            evidence_mask = None
            if row.get("evidence_mask"):
                evidence_mask = Image.open(resolve(row["evidence_mask"])).convert("L")
            noncritical_masks = []
            for item in row.get("noncritical_masks", []):
                noncritical_masks.append({**item, "mask": Image.open(resolve(item["mask"])).convert("L")})
            row["variants"] = []
            candidates = generate_semantic_variants(
                image, row["question"], evidence_mask=evidence_mask,
                noncritical_masks=noncritical_masks,
                include_photometric=True,
                include_geometric=not args.photometric_only,
                include_background=not args.photometric_only)
            for variant in candidates:
                target = output / f"{row['sample_id']}__{variant['transform']}.png"
                variant["image"].save(target, format="PNG")
                try:
                    portable_path = target.resolve().relative_to(data_root)
                except ValueError:
                    portable_path = target.resolve()
                record = {"transform": variant["transform"], "image": str(portable_path),
                          "answer_preserved": False, "critical_evidence_preserved": False,
                          "relations_preserved": False, "human_audited": False,
                          "analysis_role": None}
                variant_evidence_mask_path = None
                if variant.get("evidence_mask") is not None:
                    mask_target = output / "masks" / f"{row['sample_id']}__{variant['transform']}.png"
                    mask_target.parent.mkdir(parents=True, exist_ok=True)
                    variant["evidence_mask"].save(mask_target)
                    try:
                        variant_evidence_mask_path = str(mask_target.resolve().relative_to(data_root))
                    except ValueError:
                        variant_evidence_mask_path = str(mask_target.resolve())
                    record["evidence_mask"] = variant_evidence_mask_path
                row["variants"].append(record)
                review_rows.append({"sample_id": row["sample_id"], "question": row["question"],
                    "variant": variant["transform"], "image": str(target.resolve()),
                    "answer_preserved": "", "critical_evidence_preserved": "",
                    "relations_preserved": "", "human_audited": "", "reviewer": "", "review_date": "",
                    "analysis_role": "",
                    "family": variant["transform"].split("_")[0],
                    "variant_evidence_mask": variant_evidence_mask_path or "(same as original)"})
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    review_path = output / "variant_review.csv"
    with review_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(review_rows[0]) if review_rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"Wrote candidate manifest: {out_manifest}")
    print(f"Wrote review sheet: {review_path}")
    print("All review booleans are intentionally false in the manifest. Approve only after viewing each image and updating the manifest.")


if __name__ == "__main__":
    main()
