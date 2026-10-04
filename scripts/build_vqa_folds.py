"""Create deterministic, image-grouped cross-validation manifests from train.jsonl."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/folds"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        parser.error(f"No rows found in {args.input}")
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], str(row["image_id"])), []).append(row)
    if len(grouped) < args.folds:
        parser.error(f"Need at least {args.folds} distinct dataset/image groups; found {len(grouped)}")

    # Assign large image groups first to the lightest fold, separately by dataset.
    # This balances question counts while keeping every image's questions together.
    by_dataset: dict[str, list[tuple[str, str]]] = {}
    for key in grouped:
        by_dataset.setdefault(key[0], []).append(key)
    assignment: dict[tuple[str, str], int] = {}
    rng = random.Random(args.seed)
    for dataset, keys in sorted(by_dataset.items()):
        rng.shuffle(keys)
        keys.sort(key=lambda key: len(grouped[key]), reverse=True)
        loads = [0] * args.folds
        for key in keys:
            fold = min(range(args.folds), key=lambda i: (loads[i], i))
            assignment[key] = fold
            loads[fold] += len(grouped[key])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(args.folds):
        validation = [row for key, group in grouped.items() if assignment[key] == fold for row in group]
        training = [row for key, group in grouped.items() if assignment[key] != fold for row in group]
        rng.shuffle(training)
        rng.shuffle(validation)
        fold_dir = args.output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for name, split in (("train", training), ("validation", validation)):
            with (fold_dir / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
                for row in split:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"fold": fold, "train_rows": len(training), "validation_rows": len(validation),
                          "validation_by_dataset": dict(Counter(row["dataset"] for row in validation))}))


if __name__ == "__main__":
    main()
