"""R-FCCT report: paired statistics, controls, and comparison with the paper."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fcct.io import read_jsonl, sha256, write_json
from fcct.report import pope_answer

from .pipeline import expand, predictions
from .stats import cluster_bootstrap_delta, confusion, holm, mcnemar_exact

SPLITS = ("random", "popular", "adversarial")


def _method_predictions(rows, methods, frozen):
    preds = {m: predictions(rows, m) for m in methods}
    if frozen is not None:
        preds["threshold_control"] = predictions(rows, None, threshold=frozen["threshold_control"]["tau"])
        for name, m in frozen["methods"].items():
            if name in methods:
                preds[f"threshold_rate[{name}]"] = predictions(rows, None, threshold=m["threshold_rate"])
    return preds


def build_report(rc, base, repo: Path, paths):
    rows = read_jsonl(paths["test"])
    frozen = json.loads(paths["frozen"].read_text(encoding="utf-8")) if paths["frozen"].is_file() else None
    targets = json.loads((repo / "configs/fcct_paper_targets.json").read_text(encoding="utf-8"))
    methods = rc["evaluation"]["methods"]
    greedy = expand(rows, read_jsonl(paths["evaluate"]))["None"]
    draws, seed = rc["report"]["bootstrap"], rc["seed"]
    result = {"protocol": {"prompt_suffix": rc["pope_prompt_suffix"], "decoding": "greedy",
                           "max_new_tokens": rc["max_new_tokens"]},
              "frozen": None if frozen is None else {k: {kk: v[kk] for kk in ("variant", "attention_scale", "mlp_scale", "is_identity")}
                                                      for k, v in frozen["methods"].items()},
              "greedy": {}, "comparisons": [], "sampling": None,
              "hashes": {"test": sha256(paths["test"]), "evaluate": sha256(paths["evaluate"])}}
    all_preds = {}
    for split in SPLITS:
        split_rows = [r for r in greedy if r["split"] == split]
        labels = np.array([r["label"] == "yes" for r in split_rows], int)
        preds = _method_predictions(split_rows, methods, frozen)
        all_preds[split] = (split_rows, labels, preds)
        result["greedy"][split] = {m: confusion(labels, p) for m, p in preds.items()}
        regular_margin_yes = np.array([r["answers"]["regular"]["margin"] > 0 for r in split_rows])
        regular_text_yes = np.array([pope_answer(r["answers"]["regular"]["text"]) == "yes" for r in split_rows])
        result["greedy"][split]["_parser_margin_agreement"] = float((regular_margin_yes == regular_text_yes).mean())

    def compare(a, b, family):
        for split in SPLITS:
            split_rows, labels, preds = all_preds[split]
            if a not in preds or b not in preds:
                continue
            correct_a, correct_b = preds[a] == labels, preds[b] == labels
            test = mcnemar_exact(correct_a, correct_b)
            ci = cluster_bootstrap_delta([r["image_id"] for r in split_rows], labels, preds[a], preds[b],
                                         draws=draws, seed=seed)
            ma, mb = confusion(labels, preds[a]), confusion(labels, preds[b])
            result["comparisons"].append({
                "family": family, "split": split, "baseline": a, "method": b,
                "delta_pp": {k: mb[k] - ma[k] for k in ("accuracy", "f1", "precision", "recall", "yes_rate")},
                "ci95_pp": ci, "corrected": test["b_only_correct"], "harmed": test["a_only_correct"],
                "p_value": test["p_value"]})

    for a, b in rc["report"]["primary"]:
        compare(a, b, "primary")
    for m in methods:
        if m != "regular":
            compare("regular", m, "secondary")
    for family in ("primary", "secondary"):
        idx = [i for i, c in enumerate(result["comparisons"]) if c["family"] == family]
        for i, p in zip(idx, holm([result["comparisons"][i]["p_value"] for i in idx])):
            result["comparisons"][i]["p_holm"] = p

    if paths["sample"].is_file():
        sampled = expand(rows, read_jsonl(paths["sample"]))
        summary = {}
        for split in SPLITS:
            summary[split] = {}
            for m in rc["sampling"]["methods"]:
                values = []
                for seed_rows in sampled.values():
                    split_rows = [r for r in seed_rows if r["split"] == split]
                    labels = np.array([r["label"] == "yes" for r in split_rows], int)
                    values.append(confusion(labels, predictions(split_rows, m)))
                summary[split][m] = {k: {"mean": float(np.mean([v[k] for v in values])),
                                         "sd": float(np.std([v[k] for v in values], ddof=1)) if len(values) > 1 else 0.0}
                                     for k in ("accuracy", "precision", "recall", "f1", "yes_rate")}
        result["sampling"] = {"seeds": sorted(sampled), "settings": {k: rc["sampling"][k] for k in ("temperature", "top_p", "top_k")},
                              "metrics": summary}
        result["hashes"]["sample"] = sha256(paths["sample"])
    result["paper"] = targets["pope"]
    write_json(paths["report_dir"] / "results.json", result)
    (paths["report_dir"] / "report.md").write_text(render(result, rc), encoding="utf-8")
    return {"report": str(paths["report_dir"] / "report.md")}


def render(result, rc):
    lines = ["# R-FCCT evaluation report", "",
             "All choices (noise level, layer plans, injection strengths, thresholds) were fixed on tracing or "
             "calibration images. POPE test images were used only for this final evaluation.", "",
             f"Prompt suffix: `{result['protocol']['prompt_suffix']}`; max new tokens: {result['protocol']['max_new_tokens']}.", ""]
    if result["frozen"]:
        lines += ["## Frozen strengths (selected on calibration data)", "",
                  "| Method | Variant | lambda_a | lambda_m | identity? |", "|---|---|---:|---:|---|"]
        for name, f in result["frozen"].items():
            lines.append(f"| {name} | {f['variant']} | {f['attention_scale']} | {f['mlp_scale']} | {f['is_identity']} |")
        lines.append("")
    lines += ["## Greedy decoding (deterministic)", ""]
    for split in SPLITS:
        lines += [f"### {split}", "", "| Method | Acc | Prec | Rec | F1 | Yes-rate | FP | FN |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for m, s in result["greedy"][split].items():
            if m.startswith("_"):
                continue
            lines.append(f"| {m} | {s['accuracy']:.2f} | {s['precision']:.2f} | {s['recall']:.2f} | {s['f1']:.2f} | "
                         f"{s['yes_rate']:.2f} | {s['fp']} | {s['fn']} |")
        paper = result["paper"][split]
        lines += [f"| *paper Regular (VCD sampling baseline)* | {paper['regular']['accuracy']:.2f} | | | {paper['regular']['f1']:.2f} | | | |",
                  f"| *paper IRI* | {paper['iri']['accuracy']:.2f} | | | {paper['iri']['f1']:.2f} | | | |",
                  "", f"Parser/first-token-margin agreement (regular): {result['greedy'][split]['_parser_margin_agreement']:.4f}", ""]
    lines += ["## Paired comparisons", "",
              "Delta = method minus baseline (percentage points). CI: 95% image-cluster bootstrap. "
              "p: exact McNemar; Holm-adjusted within each family.", "",
              "| Family | Split | Baseline | Method | dAcc [95% CI] | dF1 | dRecall | dYes-rate | fixed/broken | p (Holm) |",
              "|---|---|---|---|---|---:|---:|---:|---|---:|"]
    for c in result["comparisons"]:
        ci = c["ci95_pp"]["accuracy"]
        lines.append(f"| {c['family']} | {c['split']} | {c['baseline']} | {c['method']} | "
                     f"{c['delta_pp']['accuracy']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | {c['delta_pp']['f1']:+.2f} | "
                     f"{c['delta_pp']['recall']:+.2f} | {c['delta_pp']['yes_rate']:+.2f} | {c['corrected']}/{c['harmed']} | "
                     f"{c['p_holm']:.3g} |")
    if result["sampling"]:
        s = result["sampling"]
        lines += ["", "## VCD sampling protocol (the paper's Regular baseline protocol)", "",
                  f"Seeds: {s['seeds']}; temperature {s['settings']['temperature']}, top-p {s['settings']['top_p']}, "
                  f"top-k {s['settings']['top_k']}. Mean ± SD over seeds.", "",
                  "| Split | Method | Acc | Recall | F1 | Paper Acc | Paper F1 |", "|---|---|---|---|---|---:|---:|"]
        for split in SPLITS:
            for m, v in s["metrics"][split].items():
                key = "regular" if m == "regular" else "iri"
                paper = result["paper"][split][key]
                lines.append(f"| {split} | {m} | {v['accuracy']['mean']:.2f} ± {v['accuracy']['sd']:.2f} | "
                             f"{v['recall']['mean']:.2f} ± {v['recall']['sd']:.2f} | {v['f1']['mean']:.2f} ± {v['f1']['sd']:.2f} | "
                             f"{paper['accuracy']:.2f} | {paper['f1']:.2f} |")
    lines += ["", "Controls: `threshold_control` is regular decoding with a single yes/no logit-margin threshold "
              "chosen on calibration data under the same recall floor as the tuned injections. "
              "`threshold_rate[m]` matches method m's calibration yes-rate. An injection that does not beat these "
              "controls is not evidence of improved visual grounding.", ""]
    return "\n".join(lines)
