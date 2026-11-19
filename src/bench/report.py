"""Paper-vs-measured report for every LLaVA-1.5-7B table (1, 2, 3, 4, 6, 7)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fcct.io import read_jsonl, write_json

from .metrics import chair_metrics, mme_scores, mmhal_metrics, nltk_ready, pope_metrics
from .runner import ablation_cells, cell_path, cells, manifest

SPLITS = ("random", "popular", "adversarial")


def _texts(path, rows):
    if not path.is_file():
        return None
    by_key = {r["key"]: r["text"] for r in read_jsonl(path)}
    if any(r["key"] not in by_key for r in rows):
        return None  # incomplete cell: never report partial results
    return [by_key[r["key"]] for r in rows]


def _aggregate(values):
    """Mean and sample SD over seeds for a list of flat metric dicts."""
    keys = values[0].keys()
    out = {}
    for k in keys:
        if isinstance(values[0][k], (int, float)):
            v = [x[k] for x in values]
            out[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
    return out


def collect(cfg, repo):
    results = {"pope": {}, "mme": {}, "chair": {}, "mmhal": {}, "ablations": {}, "chair_status": "ok"}
    for benchmark in ("pope", "mme", "chair", "mmhal"):
        try:
            _, rows = manifest(cfg, repo, benchmark)
        except ValueError:
            continue
        grouped = {}
        for method, protocol, seed in cells(cfg, benchmark):
            texts = _texts(cell_path(cfg, repo, benchmark, method, protocol, seed), rows)
            if texts is None:
                continue
            if benchmark == "pope":
                value = {}
                for split in SPLITS:
                    idx = [i for i, r in enumerate(rows) if r["split"] == split]
                    m = pope_metrics([rows[i]["label"] for i in idx], [texts[i] for i in idx])
                    value.update({f"{split}_{k}": v for k, v in m.items()})
            elif benchmark == "mme":
                s = mme_scores(rows, texts)
                value = {**{f"mme_{k}": v for k, v in s["subtasks"].items()},
                         "hallucination_total": s["hallucination_total"],
                         "perception_total": s["perception_total"], "cognition_total": s["cognition_total"]}
            elif benchmark == "chair":
                captions = repo / cfg["output_root"] / "chair_captions" / f"{method}__{protocol}__s{seed}.jsonl"
                captions.parent.mkdir(parents=True, exist_ok=True)
                captions.write_text("".join(json.dumps({"image_id": r["image_id"], "caption": t}) + "\n"
                                            for r, t in zip(rows, texts)), encoding="utf-8")
                source = repo / cfg["data_root"] / "sources/chair"
                if not (source / "chair.py").is_file() or not (source / "annotations").is_dir():
                    results["chair_status"] = "evaluator assets missing: run scripts/setup_bench.py --chair"
                    continue
                nltk_ready(repo / ".venv/nltk_data")
                value = chair_metrics(source / "chair.py", source / "annotations",
                                      repo / cfg["output_root"] / "chair_evaluator.pkl", captions)
            else:
                judge = repo / cfg["output_root"] / "judge" / f"{method}__{protocol}__s{seed}__{cfg['mmhal']['judge_model']}.jsonl"
                if not judge.is_file():
                    continue
                ratings = [r["rating"] for r in read_jsonl(judge)]
                if len(ratings) != 96:
                    continue
                m = mmhal_metrics(ratings)
                value = {"score": m["score"], "hallucination_rate": m["hallucination_rate"]}
            grouped.setdefault((method, protocol), []).append(value)
        for (method, protocol), values in grouped.items():
            results[benchmark].setdefault(method, {})[protocol] = _aggregate(values)
    try:
        _, adv = manifest(cfg, repo, "pope_adversarial")
        grouped = {}
        for method, protocol, seed in ablation_cells(cfg, repo):
            texts = _texts(cell_path(cfg, repo, "pope_adversarial", method, protocol, seed), adv)
            if texts is not None:
                grouped.setdefault((method, protocol), []).append(pope_metrics([r["label"] for r in adv], texts))
        for (method, protocol), values in grouped.items():
            results["ablations"].setdefault(method, {})[protocol] = _aggregate(values)
    except ValueError:
        pass
    latency = repo / cfg["output_root"] / "latency.json"
    results["latency"] = json.loads(latency.read_text()) if latency.is_file() else None
    return results


def _cell(stat, digits=2):
    if stat is None:
        return "—"
    if stat["n"] > 1:
        return f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f}"
    return f"{stat['mean']:.{digits}f}"


def render(cfg, repo, results, paper):
    names = paper["method_map"]
    protocols = list(cfg["protocols"])
    lines = ["# Paper evaluation matrix — LLaVA-1.5-7B", "",
             "Measured values use the official metric code (see `src/bench/metrics.py`). Sampling rows are "
             "mean ± SD over seeds. Paper rows are reference values; their decoding protocol is not stated in the "
             "paper (Regular/VCD rows are copied from the VCD paper, which used sampling).", "",
             "Provenance notes:", *[f"- {n}" for n in paper["provenance_notes"]], ""]

    def table(title, columns, getter, paper_rows, digits=2):
        lines.extend([f"## {title}", "", "| Method | Protocol | " + " | ".join(columns) + " |",
                      "|---|---|" + "---:|" * len(columns)])
        for method, row in paper_rows.items():
            lines.append(f"| **paper {method}** | (reported) | "
                         + " | ".join(f"{v:.{digits}f}" if isinstance(v, (int, float)) else f"{v}" for v in row) + " |")
        for method in results_methods(getter):
            for protocol in protocols:
                vals = [getter(method, protocol, c) for c in range(len(columns))]
                if all(v is None for v in vals):
                    continue
                label = f"{method}" + (" (R-FCCT, ours)" if method == "riri" else f" ≙ {names.get(method, '')}")
                lines.append(f"| {label} | {protocol} | " + " | ".join(_cell(v, digits) for v in vals) + " |")
        lines.append("")

    def results_methods(getter):
        return getter(None, None, None)

    # Table 1 — MME hallucination subset
    mme = results["mme"]
    keys1 = ["mme_existence", "mme_count", "mme_position", "mme_color", "hallucination_total"]
    def g1(method, protocol, c):
        if method is None:
            return list(mme)
        return mme.get(method, {}).get(protocol, {}).get(keys1[c])
    table("Table 1 — MME hallucination subset", ["Exist.", "Count", "Pos.", "Color", "Total"], g1,
          {k: v for k, v in paper["table1_mme_hallucination"].items() if k != "columns"}, 1)

    # Table 2 — POPE
    pope = results["pope"]
    for split in SPLITS:
        def g2(method, protocol, c, split=split):
            if method is None:
                return list(pope)
            return pope.get(method, {}).get(protocol, {}).get(f"{split}_{['accuracy', 'f1', 'precision', 'recall', 'yes_ratio'][c]}")
        table(f"Table 2 — POPE {split}", ["Acc", "F1", "Prec", "Rec", "Yes%"], g2,
              {k: v + ["", "", ""] for k, v in paper["table2_pope"][split].items()})

    # Table 3 — CHAIR
    chair = results["chair"]
    def g3(method, protocol, c):
        if method is None:
            return list(chair)
        return chair.get(method, {}).get(protocol, {}).get(["CHAIRs", "CHAIRi", "Recall", "Len"][c])
    if results.get("chair_status", "ok") != "ok":
        lines.append(f"> CHAIR: {results['chair_status']}\n")
    table("Table 3 — CHAIR (max_new_tokens 512)", ["CHAIRs↓", "CHAIRi↓", "Recall↑", "Len"], g3,
          {k: v for k, v in paper["table3_chair"].items() if k != "columns"}, 1)

    # Table 4 — MMHal-Bench
    mmhal = results["mmhal"]
    def g4(method, protocol, c):
        if method is None:
            return list(mmhal)
        return mmhal.get(method, {}).get(protocol, {}).get(["score", "hallucination_rate"][c])
    table(f"Table 4 — MMHal-Bench (judge: {cfg['mmhal']['judge_model']}; MHumanEval needs human annotators)",
          ["Score↑", "VH%↓"], g4, {k: v[:2] for k, v in paper["table4_mmhal"].items() if k != "columns"})

    # Table 6 — ablations
    plan = json.loads((repo / cfg["ablations"]["plan"]).read_text(encoding="utf-8"))
    lines += ["## Table 6 — ablations (POPE adversarial)", "",
              "| Setting | Paper Acc | Paper F1 | Protocol | Acc | F1 |", "|---|---:|---:|---|---:|---:|"]
    for case in plan["settings"]:
        measured = results["ablations"].get(f"abl:{case['id']}", {})
        if not measured:
            state = "not implemented" if not case["supported"] else "not run"
            lines.append(f"| {case['label']} | {case['paper']['accuracy']} | {case['paper']['f1']} | {state} | — | — |")
        for protocol, stats in measured.items():
            lines.append(f"| {case['label']} | {case['paper']['accuracy']} | {case['paper']['f1']} | {protocol} | "
                         f"{_cell(stats['accuracy'])} | {_cell(stats['f1'])} |")
    lines.append("")

    # Table 7 — latency
    lines += ["## Table 7 — latency (greedy)", "", "| Method | TTFT ms | TPOT ms | Adv. Acc |", "|---|---:|---:|---:|"]
    for method, row in paper["table7_latency"].items():
        if method != "columns":
            lines.append(f"| **paper {method}** | {row[0]} | {row[1]} | {row[2]} |")
    if results["latency"]:
        for method, v in results["latency"]["methods"].items():
            acc = pope.get(method, {}).get("greedy", {}).get("adversarial_accuracy")
            lines.append(f"| {method} | {v['ttft_ms']:.1f} | {v['tpot_ms']:.1f} | {_cell(acc)} |")
        lines.append(f"\nHardware: {results['latency']['environment'].get('gpu')}; the paper used A100 80GB.")
    lines += ["", "Table 5 (Qwen2-VL-7B, InternVL2-8B) and the Qwen-VL-Chat / LLaVA-NeXT columns need model "
              "adapters that are not implemented; see README."]
    return "\n".join(lines) + "\n"


def build(cfg, repo):
    paper = json.loads((repo / "configs/paper_tables_llava15.json").read_text(encoding="utf-8"))
    results = collect(cfg, repo)
    out = repo / cfg["output_root"] / "report"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "results.json", results)
    (out / "report.md").write_text(render(cfg, repo, results, paper), encoding="utf-8")
    return {"report": str(out / "report.md")}
