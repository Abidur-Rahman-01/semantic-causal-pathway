"""Compare SCC and behavioral baselines on held-out variants and image groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def values(rows: list[dict]):
    usable = []
    for row in rows:
        label = row.get("heldout_failure")
        scc = row.get("scc")
        consistency = row.get("probe_behavior_consistency")
        distribution = row.get("baseline_candidate_distribution") or {}
        if label not in (0, 1, False, True) or scc is None or consistency is None or not distribution:
            continue
        proposals = (row.get("data_provenance") or {}).get("evidence_proposals") or []
        clip_scores = [float(x["region_clip_cosine"]) for x in proposals if x.get("region_clip_cosine") is not None]
        grounding_scores = [float(x["grounding_score"]) for x in proposals if x.get("grounding_score") is not None]
        external = row.get("external_intervention") or {}
        probability_values = np.asarray(list(distribution.values()), dtype=float)
        entropy = float(-np.sum(probability_values * np.log(np.clip(probability_values, 1e-12, 1.0))))
        normalized_entropy = entropy / max(np.log(len(probability_values)), 1e-12)
        usable.append({
            "y": int(label), "group": f"{row.get('dataset', '')}:{row.get('image_id', row.get('sample_id'))}",
            "low_scc": 1 - float(scc), "probe_inconsistency": 1 - float(consistency),
            "low_confidence": 1 - max(float(p) for p in distribution.values()),
            "normalized_entropy": normalized_entropy,
            "scc": float(scc), "model_id": row.get("model_id"),
            # Higher risk scores consistently mean weaker evidence/grounding signals.
            "external_necessity": -float(external["necessity_js"]) if external.get("necessity_js") is not None else None,
            "external_control_adjusted": -float(external["necessity_js_control_adjusted"]) if external.get("necessity_js_control_adjusted") is not None else None,
            "clip_relevance": -max(clip_scores) if clip_scores else None,
            "grounding_confidence": -max(grounding_scores) if grounding_scores else None,
        })
    return usable


def grouped_auc_ci(records, signal, repeats=2000, seed=17, metric="auroc"):
    records = [r for r in records if r.get(signal) is not None]
    y = np.asarray([r["y"] for r in records], dtype=int)
    s = np.asarray([r[signal] for r in records], dtype=float)
    groups = np.asarray([r["group"] for r in records])
    scorer = roc_auc_score if metric == "auroc" else average_precision_score
    if len(np.unique(y)) < 2:
        return {metric: None, "ci95": None, "valid_replicates": 0}
    estimate = float(scorer(y, s))
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        if len(np.unique(y[idx])) == 2:
            boot.append(float(scorer(y[idx], s[idx])))
    return {metric: estimate,
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))] if boot else None,
            "valid_replicates": len(boot)}


def grouped_auc_difference(records, left, right, repeats=2000, seed=17):
    y = np.asarray([r["y"] for r in records], dtype=int)
    a = np.asarray([r[left] for r in records], dtype=float)
    b = np.asarray([r[right] for r in records], dtype=float)
    groups = np.asarray([r["group"] for r in records])
    if len(np.unique(y)) < 2:
        return {"delta_auroc": None, "ci95": None, "interpretation": "not estimable: test set needs both outcome classes"}
    observed = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    unique, rng, deltas = np.unique(groups), np.random.default_rng(seed), []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        if len(np.unique(y[idx])) == 2:
            deltas.append(float(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])))
    ci = [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))] if deltas else None
    return {"delta_auroc": observed, "ci95": ci,
            "interpretation": "SCC exceeds comparator if the full interval is above zero" if ci else "not estimable"}


def grouped_brier_ci(labels, scores, groups, repeats=2000, seed=17):
    y, p, g = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float), np.asarray(groups)
    observed = float(brier_score_loss(y, p)) if len(y) else None
    if observed is None:
        return {"brier": None, "ci95": None, "valid_replicates": 0}
    unique, rng, boot = np.unique(g), np.random.default_rng(seed), []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(g == group) for group in sampled])
        boot.append(float(brier_score_loss(y[idx], p[idx])))
    return {"brier": observed, "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "valid_replicates": len(boot)}


def _risk_coverage(labels, probabilities):
    """Selective failure risk after retaining examples with lowest train-fit failure risk."""
    if not len(labels):
        return {"points": [], "aurc": None}
    order = np.argsort(np.asarray(probabilities, dtype=float))
    labels = np.asarray(labels, dtype=int)[order]
    points = []
    for retained in range(1, len(labels) + 1):
        points.append({"coverage": retained / len(labels), "risk": float(labels[:retained].mean())})
    aurc = float(np.mean([point["risk"] for point in points]))
    return {"points": points, "aurc": aurc, "n": len(labels),
            "note": "Failure probabilities calibrated on analysis-train only; lower-risk cases retained first."}


def _single_signal_test_probs(train_records, test_records, signal):
    training = [r for r in train_records if r.get(signal) is not None]
    testing = [r for r in test_records if r.get(signal) is not None]
    if not testing or len({r["y"] for r in training}) < 2:
        return None, None
    estimator = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=17))
    estimator.fit(np.asarray([[r[signal]] for r in training], dtype=float), np.asarray([r["y"] for r in training]))
    probs = estimator.predict_proba(np.asarray([[r[signal]] for r in testing], dtype=float))[:, 1]
    return [r["y"] for r in testing], probs


def _invariance_summary(rows, low_cps_threshold=0.5):
    eligible = [r for r in rows if r.get("vqa_consensus_score") is not None
                and r.get("vqa_consensus_score", 0) >= 0.5
                and r.get("probe_behavior_consistency") is not None
                and r.get("cps_weighted") is not None]
    invariant = [r for r in eligible if r["probe_behavior_consistency"] == 1.0]
    return {
        "n_baseline_correct_with_defined_pathway": len(eligible),
        "n_behaviorally_invariant_correct": len(invariant),
        "fraction_invariant_correct_with_low_weighted_cps": (
            sum(r["cps_weighted"] < low_cps_threshold for r in invariant) / len(invariant) if invariant else None),
        "low_cps_threshold": low_cps_threshold,
        "low_cps_threshold_is_descriptive_sensitivity_cut_not_a_validated_boundary": True,
    }


def _cost_summary(rows):
    fields = {"wall_seconds": [], "model_forward_calls": [], "model_forward_seconds": [], "cuda_peak_memory_bytes": []}
    for row in rows:
        cost = row.get("cost") or {}
        for key in fields:
            value = row.get(key, cost.get(key))
            if value is not None:
                fields[key].append(float(value))
    return {key: {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
                  "max": float(np.max(values))} if values else {"n": 0}
            for key, values in fields.items()}


def analyze(output_root: Path, bootstrap_repeats: int = 2000, failure_score_threshold: float = 0.5, failure_vqa_score_threshold: float | None = None) -> Path:
    if failure_vqa_score_threshold is not None:
        failure_score_threshold = failure_vqa_score_threshold
    model_dirs = sorted(path for path in (output_root / "models").iterdir() if path.is_dir())
    report = {"outcome": "baseline-correct answer fails on at least one human-approved heldout variant",
              "evaluation_split": "analysis_split=test",
              "label_rule": f"baseline VQA consensus >= {failure_score_threshold} and any heldout variant below it",
              "predictor_protocol": "fixed logistic regression fit on analysis_split=train; no test tuning",
              "baseline_features": ["probe behavioral inconsistency", "candidate-set confidence",
                                    "normalized entropy", "external necessity (low effect means higher risk)",
                                    "CLIP relevance (low means higher risk)", "grounding confidence (low means higher risk)"],
              "models": {}}
    plot_rows = []
    for model_dir in model_dirs:
        result_path = model_dir / "semantic_pathway_results.jsonl"
        if not result_path.is_file():
            continue
        all_rows = read_rows(result_path)
        test_rows = [row for row in all_rows if row.get("analysis_split") == "test"]
        records = values(test_rows)
        train_records = values([row for row in all_rows if row.get("analysis_split") == "train"])
        metrics = {}
        for signal in ("low_scc", "probe_inconsistency", "low_confidence", "normalized_entropy", "external_necessity",
                       "external_control_adjusted", "clip_relevance", "grounding_confidence"):
            eligible = [r for r in records if r.get(signal) is not None]
            auc = grouped_auc_ci(eligible, signal, bootstrap_repeats)
            auc["auprc_grouped_ci95"] = grouped_auc_ci(eligible, signal, bootstrap_repeats, metric="auprc")
            labels = np.asarray([r["y"] for r in eligible], dtype=int)
            scores = np.asarray([r[signal] for r in eligible], dtype=float)
            auc.update({"n": len(eligible), "auprc": None, "brier": None})
            if len(np.unique(labels)) == 2:
                auc["auprc"] = float(average_precision_score(labels, scores))
                train_signal = [r for r in train_records if r.get(signal) is not None]
                if len({r["y"] for r in train_signal}) == 2:
                    calibration = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=17))
                    calibration.fit(np.asarray([[r[signal]] for r in train_signal], dtype=float), np.asarray([r["y"] for r in train_signal]))
                    calibrated = calibration.predict_proba(np.asarray([[value] for value in scores]))[:, 1]
                    auc["brier"] = float(brier_score_loss(labels, calibrated))
                    auc["brier_grouped_ci95"] = grouped_brier_ci(labels, calibrated,
                        [r["group"] for r in eligible], bootstrap_repeats)
                    auc["brier_protocol"] = "one-feature logistic calibration fit on analysis-train only"
            metrics[signal] = auc
        metrics["delta_scc_vs_probe_behavior"] = grouped_auc_difference(
            records, "low_scc", "probe_inconsistency", bootstrap_repeats)
        metrics["delta_scc_vs_confidence"] = grouped_auc_difference(
            records, "low_scc", "low_confidence", bootstrap_repeats)
        learned = {"baseline_only": None, "baseline_plus_scc": None,
                   "delta_scc_added_to_baseline": None}
        if len({row["y"] for row in train_records}) == 2 and len({row["y"] for row in records}) == 2:
            baseline_features = ["probe_inconsistency", "low_confidence", "normalized_entropy", "external_necessity",
                                 "external_control_adjusted", "clip_relevance", "grounding_confidence"]
            plus_scc_features = [*baseline_features, "low_scc"]
            def fitted_test_scores(features):
                x_train = np.asarray([[np.nan if row.get(col) is None else row[col] for col in features] for row in train_records], dtype=float)
                y_train = np.asarray([row["y"] for row in train_records])
                x_test = np.asarray([[np.nan if row.get(col) is None else row[col] for col in features] for row in records], dtype=float)
                estimator = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                          LogisticRegression(class_weight="balanced", max_iter=2000,
                                                             random_state=17))
                estimator.fit(x_train, y_train)
                return estimator.predict_proba(x_test)[:, 1]
            for key, features in (("baseline_only", baseline_features),
                                  ("baseline_plus_scc", plus_scc_features)):
                scores = fitted_test_scores(features)
                for row, score in zip(records, scores):
                    row[key] = float(score)
                learned[key] = grouped_auc_ci(records, key, bootstrap_repeats)
                learned[key]["auprc_grouped_ci95"] = grouped_auc_ci(records, key, bootstrap_repeats, metric="auprc")
                learned[key]["auprc"] = float(average_precision_score(
                    [r["y"] for r in records], scores))
                learned[key]["brier_grouped_ci95"] = grouped_brier_ci(
                    [r["y"] for r in records], scores, [r["group"] for r in records], bootstrap_repeats)
                learned[key]["brier"] = float(brier_score_loss([r["y"] for r in records], scores))
            learned["delta_scc_added_to_baseline"] = grouped_auc_difference(
                records, "baseline_plus_scc", "baseline_only", bootstrap_repeats)
        metrics["train_fitted_prediction"] = learned
        report["models"][model_dir.name] = {
            "n_test_rows": len(test_rows), "n_labeled": len(records),
            "n_baseline_incorrect_or_unscored": len(test_rows) - len(records),
            "n_train_labeled": len(train_records),
            "n_test_images": len({r["group"] for r in records}),
            "outcome_positive": sum(r["y"] for r in records), "metrics": metrics,
            "risk_coverage": {},
            "behavioral_vs_mechanistic_invariance": _invariance_summary(test_rows),
            "by_question_type": {
                str(kind): _invariance_summary([r for r in test_rows if r.get("question_type") == kind])
                for kind in sorted({r.get("question_type") for r in test_rows if r.get("question_type")})
            },
            "cost": _cost_summary(test_rows),
        }
        correlation_rows = [r for r in records if r.get("clip_relevance") is not None
                            and r.get("external_necessity") is not None]
        if len(correlation_rows) >= 3:
            from scipy.stats import spearmanr
            clip_risk = [r["clip_relevance"] for r in correlation_rows]
            external_risk = [r["external_necessity"] for r in correlation_rows]
            report["models"][model_dir.name]["clip_external_effect_correlation"] = {
                "n": len(correlation_rows),
                "spearman_rho": float(spearmanr(clip_risk, external_risk).statistic),
                "interpretation": "Correlation of lower CLIP relevance risk with lower external evidence effect risk; both signs are oriented as risk.",
            }
        else:
            report["models"][model_dir.name]["clip_external_effect_correlation"] = {
                "n": len(correlation_rows), "spearman_rho": None, "interpretation": "not estimable"}
        for signal in ("low_scc", "probe_inconsistency", "low_confidence", "normalized_entropy",
                       "external_necessity", "external_control_adjusted", "clip_relevance", "grounding_confidence"):
            coverage_y, coverage_p = _single_signal_test_probs(train_records, records, signal)
            if coverage_y is not None:
                report["models"][model_dir.name]["risk_coverage"][signal] = _risk_coverage(coverage_y, coverage_p)
        plot_rows.append((model_dir.name, records, metrics))

    report_path = output_root / "heldout_prediction_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    labels = ["SCC risk", "Probe inconsistency", "Low confidence"]
    signals = ["low_scc", "probe_inconsistency", "low_confidence"]
    names, positions, heights, errors, missing_positions = [], [], [], [], []
    for model_index, (name, _, metrics) in enumerate(plot_rows):
        names.append(name)
        for signal_index, signal in enumerate(signals):
            result = metrics[signal]
            positions.append(model_index + (signal_index - 1) * .22)
            heights.append(result["auroc"] if result["auroc"] is not None else 0)
            if result["auroc"] is None:
                missing_positions.append(positions[-1])
            errors.append(((result["auroc"] - result["ci95"][0], result["ci95"][1] - result["auroc"])
                           if result["auroc"] is not None and result["ci95"] else (0, 0)))
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 2.3), 5))
    if positions:
        err = np.asarray(errors, dtype=float).T
        ax.bar(positions, heights, width=.19, yerr=err, capsize=3,
               color=["#3569a8", "#7c9d58", "#d28a42"] * len(names))
        ax.axhline(.5, color="#777777", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(names)), names, rotation=15, ha="right")
        ax.set_ylim(0, 1)
        for position in missing_positions:
            ax.text(position, .025, "N/E", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_ylabel("AUROC on image held-out outcome")
    ax.set_title("Predicting failure on variants excluded from pathway measurement")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=color) for color in ("#3569a8", "#7c9d58", "#d28a42")], labels)
    fig.tight_layout()
    fig.savefig(output_root / "heldout_prediction_auroc.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(7, len(plot_rows) * 2.3), 5))
    deltas = [(name, metrics["train_fitted_prediction"]["delta_scc_added_to_baseline"])
              for name, _, metrics in plot_rows]
    valid_deltas = [(name, result) for name, result in deltas if result and result.get("delta_auroc") is not None]
    if valid_deltas:
        x = np.arange(len(valid_deltas))
        center = [result["delta_auroc"] for _, result in valid_deltas]
        low = [center[i] - result["ci95"][0] for i, (_, result) in enumerate(valid_deltas)]
        high = [result["ci95"][1] - center[i] for i, (_, result) in enumerate(valid_deltas)]
        ax.bar(x, center, yerr=np.asarray([low, high]), capsize=4, color="#3569a8")
        ax.set_xticks(x, [name for name, _ in valid_deltas], rotation=15, ha="right")
    else:
        ax.text(.5, .5, "Not estimable: insufficient labeled train/test outcomes",
                ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_ylabel("Test AUROC gain after adding SCC")
    ax.set_title("Incremental held-out prediction value of SCC")
    fig.tight_layout()
    fig.savefig(output_root / "incremental_scc_auroc.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(7, len(plot_rows) * 2.3), 5))
    plotted = False
    for index, (name, records, _) in enumerate(plot_rows):
        failures = [r["scc"] for r in records if r["y"] == 1]
        successes = [r["scc"] for r in records if r["y"] == 0]
        if failures and successes:
            boxes = ax.boxplot([successes, failures], positions=[index * 3 + 1, index * 3 + 2],
                               widths=.7, patch_artist=True)
            for box, color in zip(boxes["boxes"], ("#8fb4d8", "#e7a08b")):
                box.set_facecolor(color)
            plotted = True
    if plotted:
        tick_positions = [index * 3 + 1.5 for index in range(len(plot_rows))]
        ax.set_xticks(tick_positions, [row[0] for row in plot_rows], rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.legend(handles=[Patch(facecolor="#8fb4d8", label="No heldout failure"),
                           Patch(facecolor="#e7a08b", label="Heldout failure")])
    ax.set_ylabel("SCC from probe variants")
    ax.set_title("Probe pathway consistency vs failure on heldout variants")
    fig.tight_layout()
    fig.savefig(output_root / "scc_heldout_failures.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {report_path} and comparison plots under {output_root}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--failure-vqa-score-threshold", type=float, default=0.5)
    args = parser.parse_args()
    analyze(args.output_root, args.bootstrap_repeats, args.failure_vqa_score_threshold)


if __name__ == "__main__":
    main()
