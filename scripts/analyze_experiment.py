"""Aggregate per-sample results and compare preregistered failure signals."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semantic_circuits.evaluation import binary_metrics, grouped_bootstrap_metric


def read_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def summarize(rows: list[dict]) -> dict:
    output = {"n": len(rows), "by_question_type": {}}
    strata = defaultdict(list)
    for row in rows:
        strata[row.get("question_type") or "unspecified"].append(row)
    for name, subset in sorted(strata.items()):
        output["by_question_type"][name] = {
            "n": len(subset),
            "mean_scc": _mean(subset, "scc"),
            "mean_cps_weighted": _mean(subset, "cps_weighted"),
            "mean_ceca": _mean(subset, "ceca_distribution_mean"),
            "mean_vqa_consensus": _mean(subset, "vqa_consensus_score"),
            "variant_flip_rate": _mean(subset, "semantic_variant_failure"),
        }
    labeled = [row for row in rows if row.get("semantic_variant_failure") is not None]
    prediction_metrics = {}
    if labeled:
        y = [int(row["semantic_variant_failure"]) for row in labeled]
        signals = {
            "low_scc": [1 - float(row["scc"]) if row.get("scc") is not None else 0.5 for row in labeled],
            "low_cps": [1 - float(row["cps_weighted"]) if row.get("cps_weighted") is not None else 0.5 for row in labeled],
            "low_answer_confidence": [1 - max((row.get("baseline_candidate_distribution") or {"x": 0.5}).values()) for row in labeled],
            "external_necessity_js": [float((row.get("external_intervention") or {}).get("necessity_js") or 0) for row in labeled],
            "external_control_adjusted_necessity": [float((row.get("external_intervention") or {}).get("necessity_js_control_adjusted") or 0) for row in labeled],
        }
        groups = [str(row.get("image_id", row.get("sample_id"))) for row in labeled]
        for name, scores in signals.items():
            prediction_metrics[name] = {**binary_metrics(y, scores),
                "grouped_auroc_ci95": grouped_bootstrap_metric(y, scores, groups, "auroc")}
    output["variant_failure_prediction"] = prediction_metrics
    output["interpretation"] = "Descriptive pilot analysis; use grouped confidence intervals and preserve locked split protocol."
    return output


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = list(read_rows(args.results))
    report = summarize(rows)
    destination = args.output or args.results.with_name("semantic_pathway_summary.json")
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Analyzed {len(rows)} records; wrote {destination}")


if __name__ == "__main__":
    main()
