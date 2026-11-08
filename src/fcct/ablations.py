"""Explicit Table 6 controls; unsupported/ambiguous variants are never invented."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .experiment import run_evaluation
from .io import read_jsonl, sha256, write_json, write_jsonl
from .report import binary_metrics


def ablation_settings(config, plan, names):
    by_id = {case["id"]: case for case in plan["settings"]}
    if len(by_id) != len(plan["settings"]):
        raise ValueError("Duplicate ablation IDs")
    results = []
    for name in names:
        if name not in by_id:
            raise ValueError(f"Unknown ablation: {name}")
        case = by_id[name]
        if not case["supported"]:
            raise ValueError(f"Ablation {name} is not implemented: {case['reason']}")
        modified = copy.deepcopy(config)
        # Each case starts from the paper's main IRI configuration so ablations
        # do not accidentally inherit another case's disabled component/range.
        modified.update(attention_scale=0.26, mlp_scale=0.16, k_source=3, k_target=10,
                        normalize=True, rr_weighting=True, iri_allowed_layers=None)
        modified.update(case["overrides"])
        results.append((case, modified))
    return results


def run_ablations(backend, config, repo, plan_path: Path, names, *, limit=None, allow_partial=False):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    settings = ablation_settings(config, plan, names)
    root = repo / config["output_root"]
    destination = root / "ablations"
    profile = root / "profile.json"
    rows = [r for r in read_jsonl(repo / config["data_root"] / "pope.jsonl") if r["split"] == "adversarial"]
    if len(rows) != 3000:
        raise ValueError("Table 6 requires all 3000 original COCO POPE adversarial questions")
    manifest = destination / "adversarial_manifest.jsonl"
    write_jsonl(manifest, rows)
    for case, modified in settings:
        case_root = destination / case["id"]
        modified["output_root"] = str(case_root.relative_to(repo)).replace("\\", "/")
        predictions = case_root / "pope.jsonl"
        print(f"Ablation: {case['label']}", flush=True)
        run_evaluation(backend, modified, repo, manifest, predictions, profile,
                       limit=limit, allow_partial=allow_partial)
        observed = read_jsonl(predictions)
        measured = binary_metrics(observed, case["method"])
        result = {"id": case["id"], "label": case["label"], "method": case["method"],
                  "all_questions_processed": len(observed) == len(rows),
                  "full_trace_profile": json.loads(profile.read_text())["complete"],
                  "measured": measured, "paper": case["paper"], "exact_reproduction": False,
                  "overrides": case["overrides"], "plan_sha256": sha256(plan_path),
                  "predictions_sha256": sha256(predictions), "profile_sha256": sha256(profile),
                  "delta_pp": {key: measured[key] - case["paper"][key] for key in ("accuracy", "f1")}}
        write_json(case_root / "metrics.json", result)
    return write_ablation_report(plan, plan_path, destination)


def write_ablation_report(plan, plan_path, destination):
    records = []
    for case in plan["settings"]:
        path = destination / case["id"] / "metrics.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["plan_sha256"] != sha256(plan_path):
                raise ValueError("Ablation plan changed; regenerate metrics under the current plan")
        else:
            record = {"id": case["id"], "label": case["label"],
                      "status": "not_run" if case["supported"] else "not_implemented",
                      "reason": case.get("reason"), "paper": case["paper"]}
        records.append(record)
    supported = [r for r in records if r.get("status") != "not_implemented"]
    complete = all(r.get("all_questions_processed") and r.get("full_trace_profile") for r in supported)
    summary = {"status": "independent_table6_controls", "exact_reproduction": False,
               "all_supported_controls_completed": complete, "entire_table6_reproduced": False,
               "interpretation_notes": plan["interpretation_notes"], "settings": records}
    write_json(destination / "ablation_summary.json", summary)
    lines = ["# Table 6: independent IRI ablation controls", "",
             "Published parameters are frozen. Original implementation and hidden-state intervention semantics remain unavailable.", "",
             "| Setting | Questions | Measured accuracy | Paper accuracy | Measured F1 | Paper F1 | State |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for row in records:
        measured = row.get("measured")
        state = ("complete" if row.get("all_questions_processed") and row.get("full_trace_profile") else "partial") if measured else row["status"]
        accuracy = f"{measured['accuracy']:.2f}" if measured else "-"
        f1 = f"{measured['f1']:.2f}" if measured else "-"
        count = str(measured["n"]) if measured else "0"
        lines.append(f"| {row['label']} | {count} | {accuracy} | {row['paper']['accuracy']:.2f} | {f1} | {row['paper']['f1']:.2f} | {state} |")
    lines += ["", "## Interpretation choices", "", *[f"- {note}" for note in plan["interpretation_notes"]]]
    (destination / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {key: summary[key] for key in ("status", "exact_reproduction", "all_supported_controls_completed", "entire_table6_reproduced")}
