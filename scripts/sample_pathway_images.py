"""Create a reproducible image-grouped manifest subset for pathway experiments.

All questions for a selected image stay together. This samples images, not rows,
so questions and semantic variants from one image cannot be split by sampling.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def image_key(row: dict) -> tuple[str, str]:
    image_id = row.get("image_id")
    if image_id is None:
        image_id = row.get("image")
    return str(row.get("dataset", "")), str(image_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Reviewed or raw JSONL manifest")
    parser.add_argument("--images", type=int, required=True, help="Number of distinct images")
    parser.add_argument("--questions-per-image", type=int, default=1,
                        help="Questions retained per image; default 1 keeps the analysis unit independent")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.images < 1 or args.questions_per_image < 1:
        parser.error("--images and --questions-per-image must be positive")

    rows = read_rows(args.manifest)
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = image_key(row)
        if not key[1] or key[1] == "None":
            raise ValueError(f"Missing image_id and image path for sample {row.get('sample_id')}")
        groups.setdefault(key, []).append(row)
    if args.images > len(groups):
        raise ValueError(f"Requested {args.images} images but manifest contains {len(groups)}")

    keys = sorted(groups)
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    chosen = set(keys[:args.images])
    train_end = int(args.images * 0.6)
    validation_end = train_end + int(args.images * 0.2)
    analysis_split = {key: ("train" if index < train_end else
                            "validation" if index < validation_end else "test")
                      for index, key in enumerate(keys[:args.images])}
    selected = []
    for key in keys[:args.images]:
        image_rows = sorted(groups[key], key=lambda row: str(row.get("sample_id", "")))
        rng.shuffle(image_rows)
        selected.extend({**row, "analysis_split": analysis_split[key]}
                         for row in image_rows[:args.questions_per_image])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "source_manifest": str(args.manifest), "sampled_manifest": str(args.output),
        "sampling_unit": "distinct image, grouped by dataset and image_id",
        "seed": args.seed, "requested_images": args.images,
        "questions_per_image": args.questions_per_image,
        "selected_images": len(chosen), "selected_questions": len(selected),
        "analysis_images_by_split": {name: sum(value == name for value in analysis_split.values())
                                     for name in ("train", "validation", "test")},
        "questions_by_dataset": dict(Counter(str(r.get("dataset", "unspecified")) for r in selected)),
        "questions_by_type": dict(Counter(str(r.get("question_type", "unspecified")) for r in selected)),
        "image_keys": [list(key) for key in sorted(chosen)],
    }
    report_path = args.output.with_suffix(args.output.suffix + ".sampling.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "image_keys"}, indent=2))


if __name__ == "__main__":
    main()
