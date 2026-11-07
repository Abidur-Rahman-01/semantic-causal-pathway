"""Evaluate local Ollama vision models on reviewed probe and held-out variants."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from run_experiment import resolve_path, dataset_answer_score
from scoring.dataset_scorers import normalize_answer, SCORER_VERSIONS
from semantic_circuits.transforms import transformation_family, validity_gate

EVALUATION_VERSION = "reference_answers_roles_v2"


def query_ollama(model: str, image_path: Path, question: str, ollama_url: str) -> str:
    payload = {
        "model": model,
        "prompt": f"Answer with a single word or short phrase.\nQuestion: {question}\nAnswer:",
        "images": [base64.b64encode(image_path.read_bytes()).decode("ascii")],
        "stream": False, "options": {"temperature": 0.0, "num_predict": 16, "seed": 17},
    }
    response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=120)
    response.raise_for_status()
    answer = response.json().get("response", "").strip()
    if not answer:
        raise ValueError("Ollama returned an empty answer")
    return answer


def reviewed_variants(row: dict) -> list[dict]:
    variants = [v for v in row.get("variants", []) if validity_gate(v)[0]]
    if not variants or any(v.get("analysis_role") not in {"probe", "heldout"} for v in variants):
        raise ValueError(f"{row['sample_id']}: reviewed variants need explicit analysis_role")
    families = {role: {transformation_family(v["transform"]) for v in variants if v["analysis_role"] == role}
                for role in ("probe", "heldout")}
    if not all(families.values()) or families["probe"] & families["heldout"]:
        raise ValueError(f"{row['sample_id']}: probe/heldout families must be present and disjoint")
    return variants


def evaluate_model_on_manifest(model: str, manifest_rows: list[dict], manifest_path: Path,
                               data_root: Path, output_path: Path, max_samples: int | None = None,
                               ollama_url: str = "http://localhost:11434") -> dict:
    from benchmark_manifest import normalize_manifest_row
    selected = [normalize_manifest_row(r) for r in (manifest_rows[:max_samples] if max_samples else manifest_rows)]
    ids = {r["sample_id"] for r in selected}
    if len(ids) != len(selected):
        raise ValueError("Manifest contains duplicate sample IDs")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    records = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if (rec.get("evaluation_version") != EVALUATION_VERSION or rec.get("manifest_sha256") != manifest_hash
                    or rec.get("model_id") != model):
                raise ValueError("Existing predictions use a different protocol/model/manifest; choose a new output directory")
            if rec["sample_id"] in records:
                raise ValueError("Duplicate sample in existing predictions")
            records[rec["sample_id"]] = rec
    # Check every input before making model calls.
    for row in selected:
        if not row.get("answers"):
            raise ValueError(f"{row['sample_id']}: reference answers are required")
        for item in [row, *reviewed_variants(row)]:
            if not resolve_path(item["image"], manifest_path, data_root).is_file():
                raise FileNotFoundError(item["image"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as sink:
        for row in selected:
            sid = row["sample_id"]
            if sid in records:
                continue
            started = time.perf_counter()
            dataset, refs = row.get("dataset"), row["answers"]
            def answer(item):
                return query_ollama(model, resolve_path(item["image"], manifest_path, data_root), row["question"], ollama_url)
            baseline = answer(row)
            base_score = dataset_answer_score(baseline, refs, dataset)
            predictions = {"probe": [], "heldout": []}
            # API errors stop this sample before writing; transport errors are not model failures.
            for variant in reviewed_variants(row):
                value = answer(variant)
                predictions[variant["analysis_role"]].append({
                    "transform": variant["transform"], "answer": value,
                    "vqa_consensus_score": dataset_answer_score(value, refs, dataset),
                    "answer_matches_baseline": normalize_answer(value, dataset) == normalize_answer(baseline, dataset),
                })
            probes, heldout = predictions["probe"], predictions["heldout"]
            consistency = sum(v["answer_matches_baseline"] for v in probes) / len(probes)
            rec = {"evaluation_version": EVALUATION_VERSION, "manifest_sha256": manifest_hash,
                   "sample_id": sid, "image_id": row.get("image_id"), "dataset": dataset,
                   "analysis_split": row.get("analysis_split"), "model_id": model,
                   "question": row["question"], "gold_answers": refs,
                   "baseline_answer": baseline, "vqa_consensus_score": base_score,
                   "probe_behavior_consistency": consistency, "probe_variant_predictions": probes,
                   "heldout_variant_predictions": heldout,
                   "heldout_failure": int(any(v["vqa_consensus_score"] < 0.5 for v in heldout)) if base_score >= 0.5 else None,
                   "wall_seconds": time.perf_counter() - started, "scorer_versions": SCORER_VERSIONS}
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
            records[sid] = rec
            print(f"[{len(records)}/{len(selected)}] {sid}: score={base_score:.3f}", flush=True)
    evaluated = [records[sid] for sid in ids]
    eligible = [r for r in evaluated if r["heldout_failure"] is not None]
    return {"model": model, "total_evaluated": len(evaluated),
            "baseline_accuracy": sum(r["vqa_consensus_score"] for r in evaluated) / len(evaluated) if evaluated else None,
            "eligible_baseline_correct": len(eligible),
            "heldout_failure_rate": sum(r["heldout_failure"] for r in eligible) / len(eligible) if eligible else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT / ".venv/data/vqa_v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ollama_clean")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--url", default="http://localhost:11434")
    args = parser.parse_args()
    if args.samples is not None and args.samples < 1:
        parser.error("--samples must be positive")
    manifest = args.manifest.resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    summaries = []
    for model in args.models:
        clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model)
        summaries.append(evaluate_model_on_manifest(model, rows, manifest, args.data_root.resolve(),
                         args.output_dir / f"{clean_name}_results.jsonl", args.samples, args.url))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
