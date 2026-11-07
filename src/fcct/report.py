"""Measured POPE scores and honest comparison to published reference numbers."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .io import read_jsonl, sha256, write_json


def pope_answer(text: str) -> str:
    """Exact classification rule in AoiDragon/POPE evaluate.py (original POPE).

    Upstream classifies any first sentence without No/not/no as yes. Keep that
    unusual rule for comparison and record empty/ambiguous generations too.
    """
    words = text.split(".", 1)[0].replace(",", "").split(" ")
    return "no" if any(token in words for token in ("No", "not", "no")) else "yes"


def binary_metrics(rows, method):
    if not rows:
        raise ValueError("Cannot score an empty evaluation")
    tp = tn = fp = fn = 0
    for row in rows:
        if row["label"] not in ("yes", "no"):
            raise ValueError("Invalid reference label")
        prediction = pope_answer(row["answers"][method])
        truth = row["label"]
        tp += prediction == truth == "yes"
        tn += prediction == truth == "no"
        fp += prediction == "yes" and truth == "no"
        fn += prediction == "no" and truth == "yes"
    return {"n": len(rows), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": 100 * (tp + tn) / len(rows),
            "precision": 100 * tp / (tp + fp) if tp + fp else 0.0,
            "recall": 100 * tp / (tp + fn) if tp + fn else 0.0,
            "f1": 200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "empty_generations": sum(not r["answers"][method].strip() for r in rows)}


def plot_profile(profile_path: Path, destination: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), sharex=True)
    for axis, component in zip(axes, ("attention", "mlp", "hidden")):
        for group, values in profile["curves"][component].items():
            axis.plot(range(32), [float("nan") if v is None else v for v in values], label=group)
        axis.set(title=component, xlabel="Decoder layer (zero based)", ylabel="Mean recovery rate")
        axis.axhline(0, color="black", linewidth=0.5)
    axes[-1].legend(fontsize=8)
    fig.suptitle("Independent FCCT reconstruction" + (" — partial engineering run" if not profile["complete"] else ""))
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def report(results: Path, manifest: Path, profile_path: Path, repo: Path, output: Path, tolerance=1.0):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be nonnegative and finite")
    samples, rows = read_jsonl(manifest), read_jsonl(results)
    expected = {r["sample_id"]: r for r in samples}
    observed = {r["sample_id"]: r for r in rows}
    if len(observed) != len(rows) or not observed.keys() <= expected.keys():
        raise ValueError("Duplicate or unknown prediction sample IDs")
    for sample_id, row in observed.items():
        if any(row[key] != expected[sample_id][key] for key in ("image_id", "question", "label", "split", "image_sha256")):
            raise ValueError("Predictions do not match the supplied reference manifest")
    meta = json.loads(results.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if meta["manifest_sha256"] != sha256(manifest) or meta["extra"]["profile_sha256"] != sha256(profile_path):
        raise ValueError("Evaluation provenance does not match supplied manifest/profile")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    reference_path = repo / "configs/fcct_paper_targets.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if meta["config"]["model_id"] != "llava-hf/llava-1.5-7b-hf":
        raise ValueError("These published targets are only for LLaVA-1.5-7B")
    complete = observed.keys() == expected.keys() and profile["complete"]
    metrics, comparisons = {}, []
    for split in ("random", "popular", "adversarial"):
        group = [r for r in rows if r["split"] == split]
        metrics[split] = {}
        if sum(r["split"] == split for r in samples) != 3000:
            raise ValueError("Expected the full original POPE manifest: 3000 rows per split")
        for method in ("regular", "iri"):
            if not group:
                metrics[split][method] = None
                continue
            measured = binary_metrics(group, method)
            metrics[split][method] = measured
            for metric in ("accuracy", "f1"):
                target = reference["pope"][split][method][metric]
                delta = measured[metric] - target
                comparisons.append({"split": split, "method": method, "metric": metric,
                                    "measured": measured[metric], "paper": target, "delta_pp": delta,
                                    "within_tolerance": abs(delta) <= tolerance if complete else None})
    summary = {"status": "complete_independent_reconstruction" if complete else "partial_engineering_run",
               "exact_reproduction": False, "paper": reference["source"], "tolerance_pp": tolerance,
               "all_pope_metrics_within_tolerance": all(c["within_tolerance"] for c in comparisons) if complete else None,
               "processed_questions": len(rows), "expected_questions": len(samples),
               "metrics": metrics, "comparison": comparisons,
               "not_evaluated": ["MME", "CHAIR", "MMHal-Bench", "MHumanEval", "other paper models", "Table 6 ablations", "Table 7 latency", "Figure 4 attention"],
               "unpublished_settings": meta["config"]["known_reconstruction_choices"],
               "hashes": {"predictions": sha256(results), "manifest": sha256(manifest),
                          "profile": sha256(profile_path), "paper_targets": sha256(reference_path)}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", summary)
    lines = ["# FCCT/IRI independent reconstruction", "", f"Status: **{summary['status']}**.",
             "Exact reproduction is not established. Published values below are reference targets.", "",
             f"Processed {len(rows)} of {len(samples)} POPE questions. Tolerance: {tolerance:g} percentage point(s).", "",
             "| Split | Method | Metric | Measured | Paper | Difference (pp) |",
             "|---|---|---|---:|---:|---:|"]
    for item in comparisons:
        lines.append(f"| {item['split']} | {item['method']} | {item['metric']} | {item['measured']:.2f} | {item['paper']:.2f} | {item['delta_pp']:+.2f} |")
    lines += ["", "Other benchmark/model/ablation results have not been reproduced by this report.", "",
              "## Unpublished settings reconstructed locally", ""]
    lines += [f"- {choice}" for choice in summary["unpublished_settings"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_profile(profile_path, output / "fcct_curves.png")
    return {k: summary[k] for k in ("status", "exact_reproduction", "all_pope_metrics_within_tolerance")}
