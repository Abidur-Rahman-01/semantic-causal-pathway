"""Freeze question-specific answer bins from a reviewed CSV without consulting gold labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROTOCOL = "annotate_before_model_inference_without_consulting_answers"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("review_csv", type=Path, help="Columns: sample_id,candidate_answers,reviewer,review_date")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decisions = {}
    with args.review_csv.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            sid = str(row.get("sample_id", "")).strip()
            bins = [item.strip() for item in str(row.get("candidate_answers", "")).split("|") if item.strip()]
            reviewer, review_date = str(row.get("reviewer", "")).strip(), str(row.get("review_date", "")).strip()
            if not sid or len(bins) < 2 or len(set(bins)) != len(bins):
                raise ValueError("Each row needs sample_id and at least two unique pipe-separated answer bins")
            if not reviewer or not review_date:
                raise ValueError(f"{sid}: preregistered candidate bins need reviewer and review_date")
            if sid in decisions:
                raise ValueError(f"Duplicate candidate-set review for {sid}")
            decisions[sid] = {"candidate_answers": bins, "candidate_answers_protocol": PROTOCOL,
                              "candidate_answers_review": {"reviewer": reviewer, "review_date": review_date}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with args.manifest.open(encoding="utf-8-sig") as source, args.output.open("w", encoding="utf-8") as destination:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id", ""))
            if sid not in decisions:
                raise ValueError(f"No preregistered candidate answer set for {sid}")
            seen.add(sid)
            row.update(decisions[sid])
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    extra = set(decisions) - seen
    if extra:
        raise ValueError(f"Review CSV contains unknown sample_id, e.g. {sorted(extra)[0]}")
    print(f"Wrote frozen answer-bin manifest: {args.output}")


if __name__ == "__main__":
    main()
