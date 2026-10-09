"""Comprehensive Statistical Analysis and Publication Figures for 500-Image Confirmatory Study."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, roc_curve, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / ".venv" / "outputs" / "pathway_confirmatory" / "models" / "qwen2_5_vl_7b" / "semantic_pathway_results.jsonl"
OUT_DIR = ROOT / ".venv" / "outputs" / "pathway_confirmatory"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DIAGRAMS_DIR = ROOT / "outputs" / "research_diagrams"
DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "figure.titlesize": 15,
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.autolayout": True,
})


def load_data():
    rows = []
    with RESULTS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_predictors(row: dict) -> dict[str, float]:
    # 1. low_scc
    scc = row.get("scc")
    low_scc = 1.0 - float(scc) if scc is not None else 1.0

    # 2. low_cps
    cps = row.get("cps_weighted")
    low_cps = 1.0 - float(cps) if cps is not None else 1.0

    # 3. probe_behavioral_inconsistency
    pbc = row.get("probe_behavior_consistency")
    probe_inconsistency = 1.0 - float(pbc) if pbc is not None else 1.0

    # 4 & 5. external necessity
    ext = row.get("external_intervention") or {}
    ext_nec = float(ext.get("necessity_js") or 0.0)
    ctrl_adj = float(ext.get("necessity_js_control_adjusted") or 0.0)
    ctrl_adj_clamped = max(0.0, ctrl_adj)

    # 6 & 7. confidence & entropy
    dist = row.get("baseline_candidate_distribution") or {"other": 1.0}
    p_vals = np.asarray(list(dist.values()), dtype=float)
    p_vals = np.clip(p_vals, 1e-12, 1.0)
    p_vals = p_vals / p_vals.sum()
    low_conf = 1.0 - float(np.max(p_vals))
    ent = -np.sum(p_vals * np.log(p_vals))
    norm_ent = float(ent / max(np.log(len(p_vals)), 1e-12))

    return {
        "low_scc": low_scc,
        "low_cps": low_cps,
        "probe_behavioral_inconsistency": probe_inconsistency,
        "external_necessity": ext_nec,
        "control_adjusted_necessity": ctrl_adj_clamped,
        "low_confidence": low_conf,
        "normalized_entropy": norm_ent,
    }


def evaluate_cohort(subset_rows: list[dict], cohort_name: str, seed: int = 17) -> dict:
    eligible = [r for r in subset_rows if (r.get("vqa_consensus_score") or 0.0) >= 0.5 and r.get("heldout_failure") in (0, 1, False, True)]
    y_true = np.asarray([int(r["heldout_failure"]) for r in eligible])
    
    # Invariant subset analysis
    invariant = [r for r in eligible if r.get("probe_behavior_consistency") == 1.0]
    collapsed_invariant = [r for r in invariant if (r.get("cps_weighted") or 1.0) < 0.50]
    
    cohort_stats = {
        "cohort": cohort_name,
        "total_instances": len(subset_rows),
        "baseline_correct": len(eligible),
        "baseline_accuracy": len(eligible) / max(len(subset_rows), 1),
        "failures": int(np.sum(y_true)),
        "failure_prevalence": float(np.mean(y_true)) if len(y_true) > 0 else 0.0,
        "invariant_count": len(invariant),
        "invariant_collapsed_count": len(collapsed_invariant),
        "invariant_collapse_rate": len(collapsed_invariant) / max(len(invariant), 1),
    }

    if len(np.unique(y_true)) < 2:
        cohort_stats["status"] = "Insufficient classes for AUC calculation"
        return cohort_stats

    # Extract all features
    features = {name: [] for name in [
        "low_scc", "low_cps", "probe_behavioral_inconsistency",
        "external_necessity", "control_adjusted_necessity",
        "low_confidence", "normalized_entropy"
    ]}

    for r in eligible:
        p_dict = extract_predictors(r)
        for k, v in p_dict.items():
            features[k].append(v)

    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    results = {}

    for name, scores in features.items():
        y_score = np.asarray(scores)
        auroc = float(roc_auc_score(y_true, y_score))
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_score)
        auprc = float(auc(r_curve, p_curve))
        brier = float(brier_score_loss(y_true, np.clip(y_score, 0, 1)))

        boot_aurocs, boot_auprcs = [], []
        boot_diff_scc = []  # Diff from low_scc
        scc_scores = np.asarray(features["low_scc"])

        for _ in range(1000):
            idx = rng.randint(0, n_samples, size=n_samples)
            if len(np.unique(y_true[idx])) < 2:
                continue
            b_auc = float(roc_auc_score(y_true[idx], y_score[idx]))
            boot_aurocs.append(b_auc)
            p_b, r_b, _ = precision_recall_curve(y_true[idx], y_score[idx])
            boot_auprcs.append(float(auc(r_b, p_b)))

            scc_b_auc = float(roc_auc_score(y_true[idx], scc_scores[idx]))
            boot_diff_scc.append(scc_b_auc - b_auc)

        ci_auroc = [float(np.percentile(boot_aurocs, 2.5)), float(np.percentile(boot_aurocs, 97.5))] if boot_aurocs else [auroc, auroc]
        ci_auprc = [float(np.percentile(boot_auprcs, 2.5)), float(np.percentile(boot_auprcs, 97.5))] if boot_auprcs else [auprc, auprc]
        ci_diff = [float(np.percentile(boot_diff_scc, 2.5)), float(np.percentile(boot_diff_scc, 97.5))] if boot_diff_scc else [0, 0]

        results[name] = {
            "auroc": auroc,
            "auroc_ci95": ci_auroc,
            "auprc": auprc,
            "auprc_ci95": ci_auprc,
            "brier": brier,
            "scc_minus_baseline_auroc": float(roc_auc_score(y_true, scc_scores) - auroc),
            "scc_minus_baseline_ci95": ci_diff,
        }

    cohort_stats["predictors"] = results
    return cohort_stats


def generate_plots(rows: list[dict]):
    eligible_test = [r for r in rows if r.get("analysis_split") == "test" and (r.get("vqa_consensus_score") or 0.0) >= 0.5 and r.get("heldout_failure") in (0, 1, False, True)]
    y_test = np.asarray([int(r["heldout_failure"]) for r in eligible_test])

    test_preds = {name: [] for name in [
        "low_scc", "low_cps", "probe_behavioral_inconsistency",
        "external_necessity", "control_adjusted_necessity",
        "low_confidence", "normalized_entropy"
    ]}
    for r in eligible_test:
        p_dict = extract_predictors(r)
        for k, v in p_dict.items():
            test_preds[k].append(v)

    # 1. ROC & PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = {
        "low_scc": "#1f77b4",
        "low_cps": "#2ca02c",
        "external_necessity": "#ff7f0e",
        "control_adjusted_necessity": "#9467bd",
        "low_confidence": "#d62728",
        "normalized_entropy": "#8c564b",
        "probe_behavioral_inconsistency": "#e377c2",
    }
    labels = {
        "low_scc": "Low SCC (Proposed Joint Causal)",
        "low_cps": "Low CPS (Pathway Consistency)",
        "external_necessity": "External Evidence Necessity",
        "control_adjusted_necessity": "Control-Adjusted Necessity",
        "low_confidence": "Low Softmax Confidence",
        "normalized_entropy": "Normalized Answer Entropy",
        "probe_behavioral_inconsistency": "Probe Behavioral Inconsistency",
    }

    # Left: ROC
    ax_roc = axes[0]
    for key, color in colors.items():
        scores = np.asarray(test_preds[key])
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_val = roc_auc_score(y_test, scores)
        ax_roc.plot(fpr, tpr, color=color, lw=2.2, label=f"{labels[key]} (AUC = {auc_val:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random Guess (AUC = 0.500)")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.05])
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)")
    ax_roc.set_title("A. Confirmatory Test Set (N=100): ROC Curves", fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.9)

    # Right: PR
    ax_pr = axes[1]
    base_rate = float(np.mean(y_test))
    for key, color in colors.items():
        scores = np.asarray(test_preds[key])
        p_c, r_c, _ = precision_recall_curve(y_test, scores)
        ap_val = auc(r_c, p_c)
        ax_pr.plot(r_c, p_c, color=color, lw=2.2, label=f"{labels[key]} (AP = {ap_val:.3f})")

    ax_pr.axhline(base_rate, color="k", linestyle="--", lw=1.5, alpha=0.6, label=f"Prevalence Base Rate ({base_rate:.1%})")
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.05])
    ax_pr.set_xlabel("Recall (Coverage)")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("B. Confirmatory Test Set (N=100): PR Curves", fontweight="bold")
    ax_pr.legend(loc="upper right", fontsize=9.5, frameon=True, framealpha=0.9)

    plt.tight_layout()
    plot1_path = OUT_DIR / "confirmatory_500_roc_pr_curves.png"
    fig.savefig(plot1_path, dpi=300)
    fig.savefig(DIAGRAMS_DIR / "confirmatory_500_roc_pr_curves.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {plot1_path}")

    # 2. Invariance Histogram across all 500
    fig, ax = plt.subplots(figsize=(8.5, 6))
    all_eligible = [r for r in rows if (r.get("vqa_consensus_score") or 0.0) >= 0.5 and r.get("heldout_failure") in (0, 1, False, True)]
    inv_rows = [r for r in all_eligible if r.get("probe_behavior_consistency") == 1.0]
    cps_all = [float(r.get("cps_weighted") or 0.0) for r in all_eligible]
    cps_inv = [float(r.get("cps_weighted") or 0.0) for r in inv_rows]

    bins = np.linspace(0, 1.0, 11)
    ax.hist(cps_all, bins=bins, alpha=0.45, color="#1f77b4", edgecolor="black", label=f"All Correct Answers (N={len(cps_all)})")
    ax.hist(cps_inv, bins=bins, alpha=0.75, color="#d62728", edgecolor="black", label=f"Behaviorally Invariant Correct (N={len(cps_inv)})")
    ax.axvline(0.5, color="red", linestyle="--", lw=2, label="Mechanistic Stability Boundary (CPS = 0.5)")
    ax.set_xlabel("Weighted Pathway Consistency (CPS)")
    ax.set_ylabel("Number of Question-Image Instances")
    ax.set_title(f"Behavioral Invariance vs. Mechanistic Invariance (N = 500 Study, Eligible={len(all_eligible)})", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plot2_path = OUT_DIR / "confirmatory_500_invariance_diagram.png"
    fig.savefig(plot2_path, dpi=300)
    fig.savefig(DIAGRAMS_DIR / "confirmatory_500_invariance_diagram.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {plot2_path}")

    # 3. CECA vs CPS Landscape
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ceca_vals = [float(r.get("ceca_distribution_mean") or 0.0) for r in all_eligible]
    cps_vals = [float(r.get("cps_weighted") or 0.0) for r in all_eligible]
    y_vals = [int(r["heldout_failure"]) for r in all_eligible]

    ceca_arr = np.asarray(ceca_vals)
    cps_arr = np.asarray(cps_vals)
    y_arr = np.asarray(y_vals)

    C, P = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    SCC_grid = C * P
    cs = ax.contour(C, P, SCC_grid, levels=[0.1, 0.25, 0.5, 0.75], colors="gray", linestyles=":", alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=9, fmt="SCC=%.2f")

    ax.scatter(ceca_arr[y_arr == 0], cps_arr[y_arr == 0], color="#2ca02c", marker="o", s=50, alpha=0.7, label=f"Maintained Correct on Held-Out (N={np.sum(y_arr==0)})")
    ax.scatter(ceca_arr[y_arr == 1], cps_arr[y_arr == 1], color="#d62728", marker="x", s=80, lw=2.2, label=f"Failed on Held-Out Variant (N={np.sum(y_arr==1)})")

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("Causal External-to-Internal Alignment (CECA)")
    ax.set_ylabel("Causal Pathway Consistency (CPS)")
    ax.set_title("Causal Mediation Landscape: CECA vs. CPS (500-Image Confirmatory)", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plot3_path = OUT_DIR / "confirmatory_500_scc_landscape.png"
    fig.savefig(plot3_path, dpi=300)
    fig.savefig(DIAGRAMS_DIR / "confirmatory_500_scc_landscape.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {plot3_path}")


def main():
    rows = load_data()
    print(f"Loaded total rows: {len(rows)}")

    train_rows = [r for r in rows if r.get("analysis_split") == "train"]
    val_rows = [r for r in rows if r.get("analysis_split") == "validation"]
    test_rows = [r for r in rows if r.get("analysis_split") == "test"]

    print(f"Train: {len(train_rows)}, Validation: {len(val_rows)}, Test: {len(test_rows)}")

    train_eval = evaluate_cohort(train_rows, "train")
    val_eval = evaluate_cohort(val_rows, "validation")
    test_eval = evaluate_cohort(test_rows, "test")
    pooled_eval = evaluate_cohort(rows, "pooled_all_500")

    full_report = {
        "study": "500-Image Confirmatory Semantic Causal Pathway Study",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "train": train_eval,
        "validation": val_eval,
        "test": test_eval,
        "pooled": pooled_eval,
    }

    report_path = OUT_DIR / "confirmatory_500_complete_evaluation.json"
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"Comprehensive report saved to: {report_path}")

    # Generate plots
    generate_plots(rows)
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()
