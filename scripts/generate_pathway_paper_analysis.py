"""Generate publication-ready 300 DPI figures and statistical analysis for the 200-Image Semantic Causal Pathway Study."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS_DIR = ROOT / "outputs" / "research_diagrams"
DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling (IEEE / NeurIPS / ACM standard)
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


def load_dataset(dataset_path: Path) -> list[dict]:
    with dataset_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_features_and_labels(all_rows: list[dict]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[str], list[dict]]:
    eligible = [r for r in all_rows if r.get("heldout_failure") in (0, 1, False, True)]
    y = np.asarray([int(r["heldout_failure"]) for r in eligible])
    groups = np.asarray([f"{r.get('dataset','')}:{r.get('image_id', r.get('sample_id'))}" for r in eligible])
    splits = [str(r.get("analysis_split", "unspecified")) for r in eligible]

    low_scc_list, low_cps_list, low_conf_list, norm_ent_list = [], [], [], []
    ext_nec_list, ctrl_adj_list = [], []

    for r in eligible:
        scc_val = r.get("scc")
        low_scc_list.append(1.0 - float(scc_val) if scc_val is not None else 1.0)
        cps_val = r.get("cps_weighted")
        low_cps_list.append(1.0 - float(cps_val) if cps_val is not None else 1.0)

        dist = r.get("baseline_candidate_distribution") or {"other": 1.0}
        p_vals = np.asarray(list(dist.values()), dtype=float)
        p_vals = np.clip(p_vals, 1e-12, 1.0)
        p_vals = p_vals / p_vals.sum()
        low_conf_list.append(1.0 - float(np.max(p_vals)))

        ent = -np.sum(p_vals * np.log(p_vals))
        norm_ent = ent / max(np.log(len(p_vals)), 1e-12)
        norm_ent_list.append(float(norm_ent))

        ext = r.get("external_intervention") or {}
        ext_nec_list.append(-float(ext.get("necessity_js") or 0.0))
        ctrl_adj_list.append(-float(ext.get("necessity_js_control_adjusted") or 0.0))

    signals = {
        "low_scc": np.asarray(low_scc_list),
        "low_cps": np.asarray(low_cps_list),
        "low_confidence": np.asarray(low_conf_list),
        "normalized_entropy": np.asarray(norm_ent_list),
        "external_necessity": np.asarray(ext_nec_list),
        "control_adjusted_necessity": np.asarray(ctrl_adj_list),
    }
    return signals, y, groups, splits, eligible


def grouped_bootstrap_ci(y: np.ndarray, s: np.ndarray, groups: np.ndarray, metric: str = "auroc",
                         n_boot: int = 2000, seed: int = 17) -> tuple[float, float, float]:
    scorer = roc_auc_score if metric == "auroc" else average_precision_score
    obs = float(scorer(y, s))
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    boot_scores = []
    for _ in range(n_boot):
        sample_g = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sample_g])
        if len(np.unique(y[idx])) == 2:
            boot_scores.append(float(scorer(y[idx], s[idx])))
    if boot_scores:
        ci_low = float(np.quantile(boot_scores, 0.025))
        ci_high = float(np.quantile(boot_scores, 0.975))
    else:
        ci_low, ci_high = obs, obs
    return obs, ci_low, ci_high


def generate_roc_curves_figure(signals_test: dict[str, np.ndarray], y_test: np.ndarray, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "low_scc": "#1f77b4",
        "low_cps": "#2ca02c",
        "external_necessity": "#ff7f0e",
        "control_adjusted_necessity": "#9467bd",
        "low_confidence": "#d62728",
        "normalized_entropy": "#8c564b",
    }
    labels = {
        "low_scc": "Low SCC (Proposed Joint Causal)",
        "low_cps": "Low CPS (Pathway Consistency)",
        "external_necessity": "External Evidence Necessity",
        "control_adjusted_necessity": "Control-Adjusted Necessity",
        "low_confidence": "Low Softmax Confidence",
        "normalized_entropy": "Normalized Answer Entropy",
    }

    # Left: ROC Curve
    ax_roc = axes[0]
    for key, color in colors.items():
        if key not in signals_test:
            continue
        scores = signals_test[key]
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        ax_roc.plot(fpr, tpr, color=color, lw=2.2, label=f"{labels[key]} (AUC = {auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random Guess (AUC = 0.500)")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.05])
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)")
    ax_roc.set_title("A. ROC Curves: Held-Out Variant Failure Prediction", fontweight="bold")
    ax_roc.legend(loc="lower right", frameon=True, framealpha=0.9)

    # Right: Precision-Recall Curve
    ax_pr = axes[1]
    base_rate = float(np.mean(y_test))
    for key, color in colors.items():
        if key not in signals_test:
            continue
        scores = signals_test[key]
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax_pr.plot(recall, precision, color=color, lw=2.2, label=f"{labels[key]} (AP = {ap:.3f})")

    ax_pr.axhline(base_rate, color="k", linestyle="--", lw=1.5, alpha=0.6, label=f"Prevalence Base Rate ({base_rate:.1%})")
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.05])
    ax_pr.set_xlabel("Recall (Coverage)")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("B. Precision-Recall Curves (PRC)", fontweight="bold")
    ax_pr.legend(loc="upper right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_invariance_diagram(rows: list[dict], output_path: Path):
    """Plot behavioral consistency vs mechanistic consistency (CPS)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    correct_rows = [r for r in rows if r.get("vqa_consensus_score", 0) >= 0.5]
    invariant_rows = [r for r in correct_rows if r.get("probe_behavior_consistency") == 1.0]

    cps_invariant = [float(r.get("cps_weighted") or 0.0) for r in invariant_rows]
    cps_all_correct = [float(r.get("cps_weighted") or 0.0) for r in correct_rows]

    bins = np.linspace(0, 1.0, 11)
    ax.hist(cps_all_correct, bins=bins, alpha=0.45, color="#1f77b4", edgecolor="black", label=f"All Correct Answers (N={len(cps_all_correct)})")
    ax.hist(cps_invariant, bins=bins, alpha=0.75, color="#d62728", edgecolor="black", label=f"Behaviorally Invariant Correct (N={len(cps_invariant)})")

    ax.axvline(0.5, color="red", linestyle="--", lw=2, label="Mechanistic Stability Boundary (CPS = 0.5)")
    ax.set_xlabel("Weighted Pathway Consistency (CPS)")
    ax.set_ylabel("Number of Question-Image Instances")
    ax.set_title("Behavioral Invariance vs. Mechanistic Invariance (N = 200)", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_scc_landscape(rows: list[dict], output_path: Path):
    """Plot 2D Landscape: CECA vs CPS with Heldout Failure markers."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    eligible = [r for r in rows if r.get("heldout_failure") in (0, 1, False, True)]
    ceca_vals = [float(r.get("ceca_distribution_mean") or 0.0) for r in eligible]
    cps_vals = [float(r.get("cps_weighted") or 0.0) for r in eligible]
    y_vals = [int(r["heldout_failure"]) for r in eligible]

    ceca_arr = np.asarray(ceca_vals)
    cps_arr = np.asarray(cps_vals)
    y_arr = np.asarray(y_vals)

    # Iso-SCC contour background
    C, P = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    SCC_grid = C * P
    cs = ax.contour(C, P, SCC_grid, levels=[0.1, 0.25, 0.5, 0.75], colors="gray", linestyles=":", alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=9, fmt="SCC=%.2f")

    # Plot non-failures and failures
    ax.scatter(ceca_arr[y_arr == 0], cps_arr[y_arr == 0], color="#2ca02c", marker="o", s=60, alpha=0.75, label="Maintained Correct on Held-Out (y = 0)")
    ax.scatter(ceca_arr[y_arr == 1], cps_arr[y_arr == 1], color="#d62728", marker="x", s=90, lw=2.5, label="Failed on Held-Out Variant (y = 1)")

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("Causal External-to-Internal Alignment (CECA)")
    ax.set_ylabel("Causal Pathway Consistency (CPS)")
    ax.set_title("Causal Mediation Landscape: CECA vs. CPS (N = 200)", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    results_path = ROOT / ".venv" / "outputs" / "pathway_pilot" / "models" / "qwen2_5_vl_7b" / "semantic_pathway_results.jsonl"
    all_rows = load_dataset(results_path)
    print(f"Loaded {len(all_rows)} samples from {results_path}")

    signals, y, groups, splits, eligible_rows = extract_features_and_labels(all_rows)
    print(f"Eligible baseline-correct instances: {len(y)} / {len(all_rows)}")

    test_idx = np.asarray([i for i, s in enumerate(splits) if s == "test"])
    if len(test_idx) == 0 or len(np.unique(y[test_idx])) < 2:
        test_idx = np.arange(len(y))

    y_test = y[test_idx]
    g_test = groups[test_idx]
    signals_test = {k: v[test_idx] for k, v in signals.items()}

    print(f"Evaluating {len(y_test)} instances in evaluation split (Failures: {np.sum(y_test)})...")

    # 1. Comparisons with bootstrap CIs
    comparisons = {}
    train_idx = np.asarray([i for i, s in enumerate(splits) if s == "train"])
    for sig in ("low_scc", "low_cps", "low_confidence", "normalized_entropy", "external_necessity", "control_adjusted_necessity"):
        auc_obs, auc_lo, auc_hi = grouped_bootstrap_ci(y_test, signals_test[sig], g_test, metric="auroc")
        ap_obs, ap_lo, ap_hi = grouped_bootstrap_ci(y_test, signals_test[sig], g_test, metric="auprc")
        s_norm = (signals_test[sig] - np.min(signals_test[sig])) / max(np.max(signals_test[sig]) - np.min(signals_test[sig]), 1e-12)
        brier = float(brier_score_loss(y_test, s_norm))
        comparisons[sig] = {
            "auroc": auc_obs, "auroc_ci95": [auc_lo, auc_hi],
            "auprc": ap_obs, "auprc_ci95": [ap_lo, ap_hi],
            "brier": brier,
        }

    # 2. Invariance Summary
    correct_rows = [r for r in all_rows if r.get("vqa_consensus_score", 0) >= 0.5]
    invariant_rows = [r for r in correct_rows if r.get("probe_behavior_consistency") == 1.0]
    collapsed_count = sum(1 for r in invariant_rows if (r.get("cps_weighted") or 0.0) < 0.5)

    invariance_summary = {
        "baseline_correct_total": len(correct_rows),
        "behaviorally_invariant_correct": len(invariant_rows),
        "mechanistically_collapsed_count": collapsed_count,
        "collapse_rate_under_invariance": collapsed_count / len(invariant_rows) if invariant_rows else 0.0,
    }

    # 3. Save Summary JSON
    summary = {
        "comparisons": comparisons,
        "invariance": invariance_summary,
        "n_total": len(all_rows),
        "n_labeled": len(y),
        "n_test_labeled": len(y_test),
        "failure_prevalence": float(np.mean(y)),
    }
    out_json = ROOT / "outputs" / "pathway_200_comparative_evaluation.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation metrics to {out_json}")

    # 4. Generate 300 DPI Publication Figures
    generate_roc_curves_figure(signals_test, y_test, DIAGRAMS_DIR / "pathway_roc_curves.png")
    generate_invariance_diagram(all_rows, DIAGRAMS_DIR / "behavioral_vs_mechanistic_invariance.png")
    generate_scc_landscape(all_rows, DIAGRAMS_DIR / "scc_scatter_landscape.png")


if __name__ == "__main__":
    main()
