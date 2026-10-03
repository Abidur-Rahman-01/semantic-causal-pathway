"""Apply explicit human decisions to generated control masks."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("review_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decisions = {}
    with args.review_csv.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (str(row.get("sample_id", "")), str(row.get("control_index", "")))
            accepted = str(row.get("accepted", "")).strip().lower()
            if not key[0] or accepted not in {"true", "false"}:
                raise ValueError("Each control review needs sample_id, control_index, and accepted=true/false")
            if accepted == "true" and (not row.get("reviewer", "").strip() or not row.get("review_date", "").strip()):
                raise ValueError(f"{key}: accepted control needs reviewer and review_date")
            decisions[key] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with args.manifest.open(encoding="utf-8-sig") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            for control in row.get("control_masks", []):
                key = (str(row["sample_id"]), str(control["control_index"]))
                seen.add(key)
                decision = decisions.get(key)
                if decision is None:
                    control.update({"reviewed": False, "accepted": False})
                    continue
                accepted = decision["accepted"].strip().lower() == "true"
                control.update({"reviewed": accepted, "accepted": accepted,
                    "review": {"reviewer": decision.get("reviewer"), "review_date": decision.get("review_date")}})
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    extra = set(decisions) - seen
    if extra:
        raise ValueError(f"Review contains unmatched sample/control rows, e.g. {sorted(extra)[0]}")
    print(f"Wrote reviewed control manifest: {args.output}")


if __name__ == "__main__":
    main()
