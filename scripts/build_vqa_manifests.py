"""Combine converted VQAv2/GQA splits and reserve an image-disjoint test holdout."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing converted split: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--test-fraction", type=float, default=0.1,
                        help="Fraction of labeled validation images reserved for final test (default: 0.1).")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")

    train = read_jsonl(args.data_dir / "vqav2_train.jsonl") + read_jsonl(args.data_dir / "gqa_train.jsonl")
    official_val = read_jsonl(args.data_dir / "vqav2_val.jsonl") + read_jsonl(args.data_dir / "gqa_val.jsonl")

    # Split by dataset and image so questions for the same image cannot appear in both sets.
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in official_val:
        grouped.setdefault((row["dataset"], str(row["image_id"])), []).append(row)
    keys = list(grouped)
    random.Random(args.seed).shuffle(keys)
    test_count = max(1, round(len(keys) * args.test_fraction))
    test_keys = set(keys[:test_count])
    validation, test = [], []
    for key, rows in grouped.items():
        (test if key in test_keys else validation).extend(rows)

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    write_jsonl(args.data_dir / "train.jsonl", train)
    write_jsonl(args.data_dir / "validation.jsonl", validation)
    write_jsonl(args.data_dir / "locked_test.jsonl", test)

    def counts(rows: list[dict]) -> dict[str, int]:
        return dict(Counter(row["dataset"] for row in rows))

    print(json.dumps({"train": counts(train), "validation": counts(validation),
                      "locked_test": counts(test), "seed": args.seed,
                      "test_fraction_of_official_validation_images": args.test_fraction}, indent=2))


if __name__ == "__main__":
    main()
