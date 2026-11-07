"""Export recorded variant decisions without inventing reviewers, dates, or rejection reasons."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semantic_circuits.transforms import validity_gate, transformation_family


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for line in args.manifest.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        from benchmark_manifest import normalize_manifest_row
        row = normalize_manifest_row(json.loads(line))
        for variant in row.get("variants", []):
            accepted, failures = validity_gate(variant)
            records.append({"sample_id": row["sample_id"], "variant_id": f"{row['sample_id']}__{variant['transform']}",
                            "transformation_family": transformation_family(variant["transform"]),
                            "accepted": accepted and variant.get("analysis_role") in {"probe", "heldout"},
                            "failed_gate_fields": failures, **{"review": variant}})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in [("variant_reviews.jsonl", records),
                         ("rejected_variants.jsonl", [r for r in records if not r["accepted"]])]:
        with (args.output_dir / name).open("x", encoding="utf-8") as stream:
            for record in subset:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Exported {len(records)} recorded decisions")


if __name__ == "__main__":
    main()
