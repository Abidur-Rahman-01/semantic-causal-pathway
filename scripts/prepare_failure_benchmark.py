"""Prepare observed failure-prediction data without synthesizing labels or splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare(rows: list[dict]) -> list[dict]:
    groups, seen, prepared = {}, set(), []
    for row in rows:
        split = row.get("analysis_split")
        if split not in {"train", "validation", "test"}:
            raise ValueError("Every result needs analysis_split=train/validation/test")
        if row.get("image_id", row.get("sample_id")) is None:
            raise ValueError("Every result needs an image_id or sample_id")
        group = f"{row.get('dataset', '')}:{row.get('image_id', row.get('sample_id'))}"
        if group in groups and groups[group] != split:
            raise ValueError(f"Image-group leakage: {group}")
        groups[group] = split
        key = (row.get("model_id"), row.get("sample_id"))
        if key in seen:
            raise ValueError(f"Duplicate prediction: {key}")
        seen.add(key)
        if row.get("heldout_failure") not in (0, 1, False, True):
            continue
        prepared.append({**row, "source_split": row.get("split"), "split": split,
                         "image_id": group, "future_failure": int(row["heldout_failure"])})
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() == args.predictions.resolve():
        parser.error("Output must differ from source predictions")
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    prepared = prepare(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in prepared:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(prepared)} observed labels; excluded {len(rows)-len(prepared)} undefined outcomes")


if __name__ == "__main__":
    main()
