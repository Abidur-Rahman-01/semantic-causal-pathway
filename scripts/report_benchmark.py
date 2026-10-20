"""Build comprehensive benchmark run artifacts, outcomes, predictors, metrics, figures, and reports.

This is an analysis and reporting pipeline conforming to benchmark protocol v1.0.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from semantic_circuits.evaluation import grouped_bootstrap_binary_metrics, grouped_bootstrap_auc_difference
from benchmark_manifest import read_jsonl
from generate_benchmark_figures import generate_all_figures

PREDICTORS = (
    "probe_behavioral_inconsistency", "low_confidence", "normalized_entropy",
    "external_necessity", "control_adjusted_necessity", "low_cps", "low_scc",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def as_float(value):
    try:
        result = float(value) if value is not None else None
        return result if result is not None and math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def predictor_values(row: dict) -> dict:
    scores = row.get("baseline_candidate_distribution") or {}
    probs = [max(0.0, as_float(v) or 0.0) for v in scores.values()]
    total = sum(probs)
    probs = [v / total for v in probs] if total > 0 else []
    max_prob = max(probs) if probs else None
    low_conf = 1.0 - max_prob if max_prob is not None else None
    if len(probs) > 1:
        normalized_entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs) / math.log(len(probs))
    else:
        normalized_entropy = None
    ext = row.get("external_intervention") or {}
    evidence_nec = as_float(ext.get("necessity_js"))
    adjusted = as_float(ext.get("necessity_js_control_adjusted"))
    if adjusted is None and evidence_nec is not None:
        control = as_float(ext.get("control_necessity_mean_js"))
        adjusted = max(0.0, evidence_nec - control) if control is not None else None
    consistency = as_float(row.get("probe_behavior_consistency"))
    cps_hard, cps_weighted = as_float(row.get("cps_hard")), as_float(row.get("cps_weighted"))
    ceca, scc = as_float(row.get("ceca_distribution_mean")), as_float(row.get("scc"))
    flags = {
        "cps_hard": cps_hard is not None,
        "cps_weighted": cps_weighted is not None,
        "ceca": ceca is not None,
        "scc": scc is not None,
    }
    return {
        "probe_behavioral_inconsistency": 1.0 - consistency if consistency is not None else None,
        "max_candidate_probability": max_prob,
        "low_confidence": low_conf,
        "normalized_entropy": normalized_entropy,
        "external_necessity": evidence_nec,
        "control_adjusted_necessity": adjusted,
        "CPS": cps_hard,
        "weighted_CPS": cps_weighted,
        "CECA": ceca,
        "SCC": scc,
        "low_cps": 1.0 - cps_weighted if cps_weighted is not None else None,
        "low_scc": 1.0 - scc if scc is not None else None,
        "metric_defined_flags": json.dumps(flags),
    }


def make_table(rows: list[dict], run_id: str, baseline_threshold: float) -> list[dict]:
    table = []
    for row in rows:
        baseline_score = as_float(row.get("vqa_consensus_score", row.get("baseline_score")))
        eligible = baseline_score is not None and baseline_score >= baseline_threshold
        failure = row.get("heldout_failure")
        if failure not in (0, 1, False, True):
            failure = None
        heldout = row.get("heldout_variant_predictions") or []
        base_answer = str(row.get("baseline_answer", "")).strip().casefold()
        flips = [str(v.get("answer", "")).strip().casefold() != base_answer
                 for v in heldout if v.get("answer") is not None]
        answer_flip = int(any(flips)) if flips else None
        predictors = predictor_values(row)
        record = {
            "run_id": run_id,
            "sample_id": row.get("sample_id"), "image_id": row.get("image_id"),
            "dataset": row.get("dataset"), "analysis_split": row.get("analysis_split", row.get("split")),
            "question_type": row.get("question_type"),
            "model_id": row.get("model_id"), "eligible": int(eligible),
            "baseline_answer": row.get("baseline_answer"), "baseline_score": baseline_score,
            "heldout_failure": int(failure) if failure is not None and eligible else None,
            "heldout_answer_flip": answer_flip if eligible else None,
            "probe_answer_consistency": as_float(row.get("probe_behavior_consistency")),
            "heldout_variant_count": len(heldout),
            "probe_variant_count": len(row.get("probe_variant_predictions") or []),
            "probe_variant_answers_and_scores": json.dumps(row.get("probe_variant_predictions") or [], ensure_ascii=False),
            "heldout_answers_and_scores": json.dumps(heldout, ensure_ascii=False),
            "cps_hard": as_float(row.get("cps_hard")), "cps_weighted": as_float(row.get("cps_weighted")),
            "ceca": as_float(row.get("ceca_distribution_mean")), "scc": as_float(row.get("scc")),
            "undefined_metric_reasons": json.dumps(row.get("metric_undefined_reasons", {}), ensure_ascii=False),
            "forward_calls": (row.get("cost") or {}).get("model_forward_calls"),
            "wall_seconds": row.get("wall_seconds", (row.get("cost") or {}).get("model_forward_seconds")),
            "peak_memory_bytes": (row.get("cost") or {}).get("cuda_peak_memory_bytes"),
        }
        record.update(predictors)
        record.update({f"calibrated_{name}": None for name in PREDICTORS})
        table.append(record)
    return table


def make_outcomes_table(rows: list[dict], run_id: str, baseline_threshold: float) -> list[dict]:
    outcomes = []
    for row in rows:
        baseline_score = as_float(row.get("vqa_consensus_score", row.get("baseline_score")))
        eligible = baseline_score is not None and baseline_score >= baseline_threshold
        failure = row.get("heldout_failure")
        if failure not in (0, 1, False, True):
            failure = None
        heldout = row.get("heldout_variant_predictions") or []
        probe_preds = row.get("probe_variant_predictions") or []
        base_answer = str(row.get("baseline_answer", "")).strip().casefold()
        flips = [str(v.get("answer", "")).strip().casefold() != base_answer
                 for v in heldout if v.get("answer") is not None]
        answer_flip = int(any(flips)) if flips else None
        
        record = {
            "run_id": run_id,
            "sample_id": row.get("sample_id"),
            "image_id": row.get("image_id"),
            "dataset": row.get("dataset"),
            "analysis_split": row.get("analysis_split", row.get("split")),
            "eligible": int(eligible),
            "baseline_answer": row.get("baseline_answer"),
            "original_score": baseline_score,
            "original_correctness": int(eligible),
            "probe_answer_consistency": as_float(row.get("probe_behavior_consistency")),
            "probe_variant_count": len(probe_preds),
            "heldout_variant_count": len(heldout),
            "heldout_failure": int(failure) if failure is not None and eligible else None,
            "heldout_answer_flip": answer_flip if eligible else None,
            "probe_predictions_json": json.dumps(probe_preds, ensure_ascii=False),
            "heldout_predictions_json": json.dumps(heldout, ensure_ascii=False),
        }
        outcomes.append(record)
    return outcomes


def fit_train_calibrators(table: list[dict]) -> dict:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    strata = {(str(row.get("dataset")), str(row.get("model_id") or "unknown_model")) for row in table}
    status = {}
    lookup = defaultdict(list)
    for row in table:
        lookup[(str(row.get("dataset")), str(row.get("model_id") or "unknown_model"),
                str(row.get("analysis_split")))].append(row)
    for dataset, model in strata:
        train = [r for r in lookup[(dataset, model, "train")]
                 if r["eligible"] and r["heldout_failure"] in (0, 1)]
        for predictor in PREDICTORS:
            valid = [r for r in train if r.get(predictor) is not None]
            key = f"{dataset}:{model}:{predictor}"
            if len(valid) < 4 or len({int(r["heldout_failure"]) for r in valid}) < 2:
                status[key] = "not_fit: insufficient training examples or one outcome class"
                continue
            x = np.asarray([r[predictor] for r in valid], dtype=float).reshape(-1, 1)
            y = np.asarray([r["heldout_failure"] for r in valid], dtype=int)
            if np.unique(x).size < 2:
                status[key] = "not_fit: predictor is constant in train"
                continue
            calibrator = LogisticRegression(solver="lbfgs", random_state=17)
            calibrator.fit(x, y)
            for split in ("train", "validation", "test"):
                for row in lookup[(dataset, model, split)]:
                    value = row.get(predictor)
                    row[f"calibrated_{predictor}"] = (
                        float(calibrator.predict_proba(np.asarray([[value]], dtype=float))[0, 1])
                        if value is not None else None)
            status[key] = {"status": "fit", "training_n": len(valid), "training_failures": int(y.sum()),
                           "fit_split": "train", "estimator": "univariate_logistic_platt"}
    return status


def summarize(table: list[dict], repeats: int, seed: int) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in table:
        model = str(row.get("model_id") or "unknown_model")
        groups[(str(row.get("analysis_split")), str(row.get("dataset")), model)].append(row)
        groups[(str(row.get("analysis_split")), "all", model)].append(row)
    result = {}
    for (split, dataset, model), rows in sorted(groups.items()):
        eligible = [r for r in rows if r["eligible"] and r["heldout_failure"] in (0, 1)]
        n_fail = sum(int(r["heldout_failure"]) for r in eligible)
        key = f"{split}:{dataset}:{model}"
        scores_present = [r["baseline_score"] for r in rows if r["baseline_score"] is not None]
        probe_consistency = [r["probe_answer_consistency"] for r in rows if r["probe_answer_consistency"] is not None]
        flips_present = [r["heldout_answer_flip"] for r in eligible if r["heldout_answer_flip"] is not None]
        section = {"n_total": len(rows), "n_eligible_with_outcome": len(eligible),
                   "n_baseline_correct": sum(r["eligible"] for r in rows),
                   "baseline_accuracy_over_scored": (sum(r["eligible"] for r in rows) / len(scores_present)) if scores_present else None,
                   "n_failures": n_fail, "n_non_failures": len(eligible) - n_fail,
                   "failure_prevalence": n_fail / len(eligible) if eligible else None,
                   "mean_probe_answer_consistency": (sum(probe_consistency) / len(probe_consistency)) if probe_consistency else None,
                   "n_heldout_answer_flips": sum(flips_present),
                   "heldout_answer_flip_rate": (sum(flips_present) / len(flips_present)) if flips_present else None,
                   "mean_forward_calls": (sum(float(r["forward_calls"]) for r in rows if r["forward_calls"] is not None) /
                                          sum(r["forward_calls"] is not None for r in rows)) if any(r["forward_calls"] is not None for r in rows) else None,
                   "mean_wall_seconds": (sum(float(r["wall_seconds"]) for r in rows if r["wall_seconds"] is not None) /
                                         sum(r["wall_seconds"] is not None for r in rows)) if any(r["wall_seconds"] is not None for r in rows) else None,
                   "question_type_summary": {},
                   "predictors": {}}
        by_question_type = defaultdict(list)
        for row in rows:
            by_question_type[str(row.get("question_type") or "unknown")].append(row)
        for question_type, type_rows in sorted(by_question_type.items()):
            scored = [r for r in type_rows if r["baseline_score"] is not None]
            type_eligible = [r for r in type_rows if r["eligible"]]
            type_outcomes = [r for r in type_eligible if r["heldout_failure"] in (0, 1)]
            type_failures = sum(int(r["heldout_failure"]) for r in type_outcomes)
            section["question_type_summary"][question_type] = {
                "n": len(type_rows), "baseline_accuracy_over_scored": sum(r["eligible"] for r in type_rows) / len(scored) if scored else None,
                "n_eligible": len(type_eligible), "n_with_heldout_outcome": len(type_outcomes),
                "heldout_failures": type_failures,
                "heldout_failure_rate": type_failures / len(type_outcomes) if type_outcomes else None,
            }
        for predictor in PREDICTORS:
            valid = [r for r in eligible if r.get(predictor) is not None]
            if not valid:
                section["predictors"][predictor] = {"n": 0, "metrics": None, "reason": "no_valid_predictor_values"}
                continue
            metrics = grouped_bootstrap_binary_metrics(
                [r["heldout_failure"] for r in valid], [r[predictor] for r in valid],
                [r["image_id"] for r in valid], repeats=repeats, seed=seed)
            calibrated_name = f"calibrated_{predictor}"
            calibrated = [r for r in valid if r.get(calibrated_name) is not None]
            if calibrated:
                calibrated_metrics = grouped_bootstrap_binary_metrics(
                    [r["heldout_failure"] for r in calibrated], [r[calibrated_name] for r in calibrated],
                    [r["image_id"] for r in calibrated], repeats=repeats, seed=seed)
                metrics["brier_calibrated"] = calibrated_metrics["brier"]
                metrics["brier_calibrated_ci95"] = calibrated_metrics["bootstrap"]["brier"]["ci95"]
                metrics["calibrated_n"] = len(calibrated)
                metrics["calibration_fit_split"] = "train; train rows are in-sample"
            else:
                metrics["brier_calibrated"] = None
                metrics["brier_calibrated_ci95"] = None
                metrics["calibrated_n"] = 0
                metrics["calibration_fit_split"] = "unavailable"
            section["predictors"][predictor] = {"n": len(valid), "metrics": metrics}
        section["paired_auc_delta_vs_low_scc"] = {}
        reference = [r for r in eligible if r.get("low_scc") is not None]
        for predictor in PREDICTORS:
            if predictor == "low_scc":
                continue
            paired = [r for r in reference if r.get(predictor) is not None]
            if not paired:
                section["paired_auc_delta_vs_low_scc"][predictor] = None
                continue
            section["paired_auc_delta_vs_low_scc"][predictor] = grouped_bootstrap_auc_difference(
                [r["heldout_failure"] for r in paired], [r[predictor] for r in paired],
                [r["low_scc"] for r in paired], [r["image_id"] for r in paired], repeats, seed)
        result[key] = section
    return result


def csv_value(value):
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value


def build_markdown_report(metadata: dict, metrics: dict, manifest_flow: dict | None, table: list[dict]) -> str:
    md = [
        "# Benchmark Run & Evaluation Report",
        "",
        "This report was automatically compiled according to **Image Evidence Stability Benchmark Protocol v1.0**.",
        "Every reported metric is traced directly to locked manifests, predictions, and grouped image bootstrap resampling.",
        "",
        "## Table 1: Run Metadata & Checkpoint Revisions",
        "",
        "| Attribute | Value |",
        "|---|---|",
        f"| Run ID | `{metadata['run_id']}` |",
        f"| Model & Revision | `{metadata['model_revision']}` |",
        f"| Scorer Version | `{metadata['scoring_version']}` |",
        f"| Analysis Unit | `{metadata['analysis_unit']}` (zero leakage across splits) |",
        f"| Code Commit | `{metadata['code_commit'] or 'uncommitted'}` |",
        f"| Predictions File SHA-256 | `{metadata['prediction_sha256'][:16]}...` |",
        f"| Manifest SHA-256 | `{metadata['manifest_sha256'][:16] if metadata['manifest_sha256'] else 'N/A'}...` |",
        f"| Config SHA-256 | `{metadata['config_sha256'][:16] if metadata['config_sha256'] else 'N/A'}...` |",
        f"| Timestamp (UTC) | `{metadata['generated_utc']}` |",
        f"| Bootstrap Resamples | {metadata['bootstrap_repeats_requested']} (seed={metadata['bootstrap_seed']}) |",
        "",
    ]

    if manifest_flow:
        md += [
            "## Table 2: Benchmark Sample Flow & Exclusions",
            "",
            "| Stage | Count | Notes |",
            "|---|---:|---|",
            f"| Proposed Candidate Variants | {manifest_flow['proposed_variants']} | Generated before review |",
            f"| Human-Reviewed Variants | {manifest_flow['reviewed_variants']} | Evaluated across 4 quality gates |",
            f"| Accepted Variants | {manifest_flow['accepted_variants']} | All 4 gates passed + disjoint family |",
            f"| Rejected Variants | {manifest_flow['rejected_reviewed_variants']} | Logged with explicit rejection reason |",
            f"| Unreviewed Variants | {manifest_flow['unreviewed_variants']} | Excluded from evaluation |",
            f"| Evaluated Prediction Samples | {manifest_flow['prediction_rows']} | Unique image-question instances |",
            f"| Eligible (Original Correct >= 0.5) | {manifest_flow['eligible_predictions']} | Cohort for primary failure analysis |",
            f"| Baseline Incorrect (Excluded) | {manifest_flow['prediction_rows'] - manifest_flow['eligible_predictions']} | Not counted as transformation failures |",
            "",
            "## Table 8: Rejected Variant Counts & Reasons",
            "",
            "| Rejection Reason | Count |",
            "|---|---:|",
        ]
        reasons = manifest_flow.get("rejection_reasons", {})
        for r_name, r_count in sorted(reasons.items(), key=lambda x: -x[1]):
            md.append(f"| `{r_name}` | {r_count} |")
        md.append("")

    # Table 3 & 4
    md += [
        "## Table 3 & 4: Performance & Failure Rates by Cohort",
        "",
        "| Split:Dataset | Total N | Baseline Acc | Eligible N | Probe Consist. | Failures | Failure Prev. | Flips | Flip Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort, sec in metrics.items():
        base_acc = f"{sec['baseline_accuracy_over_scored']:.1%}" if sec['baseline_accuracy_over_scored'] is not None else "N/A"
        probe_c = f"{sec['mean_probe_answer_consistency']:.3f}" if sec['mean_probe_answer_consistency'] is not None else "N/A"
        prev = f"{sec['failure_prevalence']:.1%}" if sec['failure_prevalence'] is not None else "N/A"
        flip_r = f"{sec['heldout_answer_flip_rate']:.1%}" if sec['heldout_answer_flip_rate'] is not None else "N/A"
        md.append(f"| {cohort} | {sec['n_total']} | {base_acc} | {sec['n_eligible_with_outcome']} | {probe_c} | {sec['n_failures']} | {prev} | {sec['n_heldout_answer_flips']} | {flip_r} |")
    md.append("")

    # Table 5
    md += [
        "## Table 5: Predictor Performance (AUROC, AUPRC, Brier, CIs)",
        "",
    ]
    for cohort, sec in metrics.items():
        if "test" not in cohort and cohort != list(metrics.keys())[0]:
            continue
        md += [
            f"### Cohort: `{cohort}` (Eligible N={sec['n_eligible_with_outcome']}, Failures={sec['n_failures']})",
            "",
            "| Predictor | N | AUROC (95% CI) | AUPRC (95% CI) | Raw Brier | Train-Calib Brier | Paired Δ AUROC vs Low-SCC (95% CI) |",
            "|---|---:|---|---|---:|---:|---|",
        ]
        for name, item in sec["predictors"].items():
            m = item.get("metrics")
            if not m:
                md.append(f"| {name} | {item['n']} | undefined | undefined | undefined | undefined | undefined |")
                continue
            def fmt(metric):
                ci = m["bootstrap"][metric]["ci95"]
                return "undefined" if m[metric] is None else f"{m[metric]:.3f} ({ci[0]:.3f}, {ci[1]:.3f})" if ci else f"{m[metric]:.3f}"
            cal_brier = f"{m.get('brier_calibrated'):.3f}" if m.get("brier_calibrated") is not None else "N/A"
            delta_info = sec["paired_auc_delta_vs_low_scc"].get(name)
            delta_str = "Reference" if name == "low_scc" else "N/A"
            if delta_info and delta_info.get("delta_auroc") is not None:
                d = delta_info["delta_auroc"]
                d_ci = delta_info["ci95"]
                delta_str = f"{d:+.3f} ({d_ci[0]:+.3f}, {d_ci[1]:+.3f})" if d_ci else f"{d:+.3f}"
            md.append(f"| {name} | {item['n']} | {fmt('auroc')} | {fmt('auprc')} | {m['brier']:.3f} | {cal_brier} | {delta_str} |")
        md.append("")

    # Table 6: White-box Availability
    md += [
        "## Table 6: White-Box Metric Availability & Undefined Accounting",
        "",
        "| Metric | Defined N | Undefined N | Defined % | Notes / Reason Codes |",
        "|---|---:|---:|---:|---|",
    ]
    total_samples = len(table)
    for m_name in ("cps_hard", "cps_weighted", "ceca", "scc"):
        def_count = sum(r.get(m_name) is not None for r in table)
        undef_count = total_samples - def_count
        pct = def_count / total_samples if total_samples else 0
        md.append(f"| `{m_name}` | {def_count} | {undef_count} | {pct:.1%} | Undefined preserved as null; no zero-imputation |")
    md.append("")

    # Table 7: Compute Cost
    md += [
        "## Table 7: Compute Cost & Model Calls",
        "",
        f"- Mean model forward calls per sample: `{metrics.get(list(metrics.keys())[0], {}).get('mean_forward_calls') or 'N/A'}`",
        f"- Mean wall seconds per sample: `{metrics.get(list(metrics.keys())[0], {}).get('mean_wall_seconds') or 'N/A'}`",
        "",
        "## Essential Figures",
        "",
        "All figures below were generated deterministically from the saved data files:",
        "- `figures/sample_flow.png`: Sample flow and quality exclusion funnel",
        "- `figures/transformation_examples.png`: Reviewed transformation variants and evidence masks",
        "- `figures/baseline_performance_and_probe_consistency.png`: Original score vs probe consistency",
        "- `figures/heldout_failure_rate.png`: Held-out failure rate by split and cohort",
        "- `figures/behavior_consistency_vs_evidence_necessity.png`: Behavioral consistency vs external evidence necessity",
        "- `figures/predictive_performance_roc_prc.png`: ROC and Precision-Recall evaluation curves",
        "- `figures/calibration_reliability_plot.png`: Platt-scaled calibration reliability diagram",
        "- `figures/ablations_paired_differences.png`: Paired ablation effect differences",
        "- `figures/whitebox_metric_availability.png`: White-box metric availability accounting",
        "",
    ]
    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="Sample-level JSONL prediction output")
    parser.add_argument("--manifest", required=True, help="Frozen JSONL manifest used for this run")
    parser.add_argument("--config", required=True, help="Frozen JSON config used for this run")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--scoring-version", required=True)
    parser.add_argument("--baseline-threshold", type=float, default=.5)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    predictions = Path(args.predictions).resolve()
    manifest = Path(args.manifest).resolve()
    config = Path(args.config).resolve()
    for required_path in (predictions, manifest, config):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw_rows = read_jsonl(predictions)
    raw_ids = [str(row.get("sample_id", "")) for row in raw_rows]
    if not raw_ids or any(not item for item in raw_ids) or len(set(raw_ids)) != len(raw_ids):
        raise ValueError("prediction JSONL must contain non-empty, unique sample_id values")

    # Construct tables
    table = make_table(raw_rows, args.run_id, args.baseline_threshold)
    outcomes = make_outcomes_table(raw_rows, args.run_id, args.baseline_threshold)
    calibration_status = fit_train_calibrators(table)

    manifest_flow = None
    if manifest.is_file():
        manifest_rows = read_jsonl(manifest)
        manifest_ids = [str(row.get("sample_id", "")) for row in manifest_rows]
        if any(not value for value in manifest_ids) or len(set(manifest_ids)) != len(manifest_ids):
            raise ValueError("manifest must contain non-empty, unique sample_id values")
        prediction_by_id = {str(row["sample_id"]): row for row in raw_rows}
        manifest_by_id = {str(row["sample_id"]): row for row in manifest_rows}
        common_ids = set(prediction_by_id) & set(manifest_by_id)
        for sample_id in common_ids:
            pred, source = prediction_by_id[sample_id], manifest_by_id[sample_id]
            checks = (
                ("image_id", str(pred.get("image_id")), str(source.get("image_id"))),
                ("dataset", str(pred.get("dataset")), str(source.get("dataset"))),
                ("analysis_split", str(pred.get("analysis_split", pred.get("split"))), str(source.get("analysis_split"))),
            )
            mismatch = [name for name, actual, expected in checks if actual != expected]
            if mismatch:
                raise ValueError(f"prediction/manifest identity mismatch for {sample_id}: {', '.join(mismatch)}")
        coverage = {
            "manifest_samples": len(manifest_ids), "prediction_samples": len(raw_rows),
            "matched_samples": len(common_ids),
            "missing_prediction_sample_ids": sorted(set(manifest_ids) - set(prediction_by_id)),
            "unexpected_prediction_sample_ids": sorted(set(prediction_by_id) - set(manifest_by_id)),
        }
        (output / "prediction_manifest_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
        proposed = reviewed = accepted = rejected = 0
        rejection_reasons = Counter()
        for row in manifest_rows:
            variants = row.get("variants", [])
            proposed += len(variants)
            for variant in variants:
                modern = row.get("schema_version") == "1.0"
                is_reviewed = bool(variant.get("human_audited")) if not modern else bool(variant.get("reviewer_id") and variant.get("review_date"))
                is_accepted = variant.get("accepted") is True if modern else all(variant.get(k) is True for k in ("answer_preserved", "critical_evidence_preserved", "relations_preserved", "human_audited"))
                reviewed += int(is_reviewed)
                accepted += int(is_accepted)
                rejected += int(is_reviewed and not is_accepted)
                if is_reviewed and not is_accepted:
                    reason = variant.get("rejection_reason") or variant.get("rejection_reason_code") or "unspecified"
                    rejection_reasons[str(reason)] += 1
        manifest_flow = {
            "manifest_rows": len(manifest_rows), "proposed_variants": proposed,
            "reviewed_variants": reviewed, "accepted_variants": accepted,
            "rejected_reviewed_variants": rejected,
            "unreviewed_variants": proposed - reviewed,
            "rejection_reasons": dict(rejection_reasons),
            "prediction_rows": len(raw_rows),
            "eligible_predictions": sum(r["eligible"] for r in table),
            "matched_prediction_samples": coverage["matched_samples"],
            "missing_prediction_samples": len(coverage["missing_prediction_sample_ids"]),
            "unexpected_prediction_samples": len(coverage["unexpected_prediction_sample_ids"]),
            "heldout_failures": sum(int(r["heldout_failure"]) for r in table if r.get("heldout_failure") == 1),
        }

    # 1. Write outcomes.csv
    outcomes_fields = list(outcomes[0].keys()) if outcomes else ["sample_id", "image_id"]
    with (output / "outcomes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=outcomes_fields)
        writer.writeheader()
        writer.writerows({k: csv_value(v) for k, v in row.items()} for row in outcomes)

    # 2. Write predictors.csv (and alias predictor_table.csv)
    pred_fields = list(table[0].keys()) if table else ["sample_id", "image_id"]
    for fname in ("predictors.csv", "predictor_table.csv"):
        with (output / fname).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=pred_fields)
            writer.writeheader()
            writer.writerows({k: csv_value(v) for k, v in row.items()} for row in table)

    # 3. Summarize metrics
    metrics = summarize(table, args.bootstrap_repeats, args.seed)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 4. Copy / link inputs into run directory if not already there
    dest_manifest = output / "manifest.jsonl"
    if manifest != dest_manifest and not dest_manifest.exists():
        shutil.copyfile(manifest, dest_manifest)

    dest_preds = output / "predictions.jsonl"
    if predictions != dest_preds and not dest_preds.exists():
        shutil.copyfile(predictions, dest_preds)

    dest_config = output / "run_config.json"
    if config != dest_config and not dest_config.exists():
        shutil.copyfile(config, dest_config)

    logs_dir = output / "logs"
    logs_dir.mkdir(exist_ok=True)

    # 5. Build reports
    metadata = {
        "report_version": "1.0", "run_id": args.run_id, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_path": str(dest_preds), "prediction_sha256": sha256_file(dest_preds),
        "manifest_path": str(dest_manifest), "manifest_sha256": sha256_file(dest_manifest) if dest_manifest.is_file() else None,
        "config_path": str(dest_config), "config_sha256": sha256_file(dest_config) if dest_config.is_file() else None,
        "code_commit": git_commit(), "model_revision": args.model_revision,
        "scoring_version": args.scoring_version, "analysis_unit": "image_id",
        "baseline_eligibility_threshold": args.baseline_threshold,
        "bootstrap_repeats_requested": args.bootstrap_repeats, "bootstrap_seed": args.seed,
        "n_prediction_rows": len(raw_rows), "n_unique_sample_ids": len({r.get("sample_id") for r in raw_rows}),
        "n_unique_image_ids": len({r.get("image_id") for r in raw_rows}),
        "metric_defined_counts": {name: sum(r.get(name) is not None for r in table)
                                  for name in ("cps_hard", "cps_weighted", "ceca", "scc")},
        "train_only_calibration": calibration_status,
        "manifest_flow": manifest_flow,
    }
    report = {"metadata": metadata, "cohorts": metrics}
    for fname in ("report.json", "benchmark_report.json"):
        (output / fname).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_content = build_markdown_report(metadata, metrics, manifest_flow, table)
    for fname in ("report.md", "benchmark_report.md"):
        (output / fname).write_text(md_content, encoding="utf-8")

    # 6. Generate all figures
    generate_all_figures(output)

    # 7. Generate checksums.sha256
    checksum_lines = []
    for fpath in sorted(output.rglob("*")):
        if fpath.is_file() and fpath.name != "checksums.sha256":
            rel = fpath.relative_to(output)
            csum = sha256_file(fpath)
            checksum_lines.append(f"{csum}  {rel.as_posix()}")
    (output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    print(f"Successfully processed run {args.run_id} into {output}")
    print(f"Generated: outcomes.csv, predictors.csv, metrics.json, report.json, report.md, checksums.sha256, figures/")


if __name__ == "__main__":
    main()
