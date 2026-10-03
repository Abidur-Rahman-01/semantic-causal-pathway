"""Apply explicit human evidence-mask decisions to proposal manifests."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("review_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decisions = {}
    with args.review_csv.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            sample_id = str(row.get("sample_id", ""))
            accepted = str(row.get("mask_accepted", "")).strip().lower()
            if not sample_id or accepted not in {"true", "false"}:
                raise ValueError("Each evidence review row needs a sample_id and mask_accepted=true/false")
            if accepted == "true" and (not row.get("reviewer", "").strip() or not row.get("review_date", "").strip()):
                raise ValueError(f"{sample_id}: accepted mask needs reviewer and review_date")
            decisions[sample_id] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with args.manifest.open(encoding="utf-8-sig") as source, args.output.open("w", encoding="utf-8") as destination:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if sample_id not in decisions:
                row["evidence_mask_reviewed"] = False
            else:
                decision = decisions[sample_id]
                seen.add(sample_id)
                accepted = decision["mask_accepted"].strip().lower() == "true"
                if accepted and not row.get("evidence_mask"):
                    raise ValueError(f"{sample_id}: cannot accept a missing evidence mask")
                row["evidence_mask_reviewed"] = accepted
                row["evidence_mask_review"] = {"reviewer": decision.get("reviewer"), "review_date": decision.get("review_date")}
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    extra = set(decisions) - seen
    if extra:
        raise ValueError(f"Review contains unknown sample_id: {sorted(extra)[0]}")
    print(f"Wrote reviewed evidence manifest: {args.output}")


if __name__ == "__main__":
    main()
