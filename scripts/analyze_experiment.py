"""Small descriptive summaries; inferential analysis lives in report_benchmark.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def summarize(rows: list[dict]) -> dict:
    groups = {f"{r.get('dataset', '')}:{r.get('image_id', r.get('sample_id'))}" for r in rows}
    metrics = {}
    for field in ("scc", "cps_weighted", "cps_hard", "ceca_distribution_mean", "vqa_consensus_score"):
        values = [float(r[field]) for r in rows if r.get(field) is not None]
        metrics[field] = {"defined_n": len(values), "undefined_n": len(rows) - len(values),
                          "mean": sum(values) / len(values) if values else None}
    eligible = [r for r in rows if r.get("heldout_failure") in (0, 1, False, True)]
    return {"n": len(rows), "unique_images": len(groups), "metrics": metrics,
            "heldout_labeled_n": len(eligible),
            "heldout_failures": sum(int(r["heldout_failure"]) for r in eligible),
            "claim_boundary": "Descriptive only; undefined metrics excluded without imputation. Use pathway.py report for grouped inference."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(list(read_rows(args.predictions))), indent=2))


if __name__ == "__main__":
    main()
