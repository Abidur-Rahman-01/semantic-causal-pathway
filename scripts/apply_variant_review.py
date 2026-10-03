"""Apply a completed variant_review.csv to a prepared JSONL manifest."""
import argparse
import csv
import json
from pathlib import Path

REQUIRED = ("answer_preserved", "critical_evidence_preserved", "relations_preserved", "human_audited")


def truth(value):
    value = str(value).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"Review fields must be true/false; got {value!r}")
    return value == "true"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("review_csv")
    parser.add_argument("--output", default="data/approved_manifest.jsonl")
    args = parser.parse_args()
    manifest, review, output = map(Path, (args.manifest, args.review_csv, args.output))
    decisions = {}
    with review.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (str(row["sample_id"]), str(row["variant"]))
            decisions[key] = {field: truth(row.get(field, "")) for field in REQUIRED}
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with manifest.open(encoding="utf-8-sig") as src, output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            for variant in row.get("variants", []):
                key = (str(row["sample_id"]), str(variant["transform"]))
                seen.add(key)
                variant.update(decisions.get(key, {field: False for field in REQUIRED}))
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    extra = set(decisions) - seen
    if extra:
        raise ValueError(f"Review sheet contains unmatched sample/variant rows, e.g. {sorted(extra)[0]}")
    print(f"Wrote {output}. Approved variants per sample:")
    with output.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            accepted = [v["transform"] for v in row.get("variants", []) if all(v.get(k) is True for k in REQUIRED)]
            print(f"  {row['sample_id']}: {len(accepted)}")


if __name__ == "__main__":
    main()
