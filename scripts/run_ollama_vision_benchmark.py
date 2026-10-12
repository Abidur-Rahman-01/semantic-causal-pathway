"""Automated Comparative Benchmark for Vision-Capable Ollama Models.

Evaluates multimodal models (e.g., llama3.2-vision:11b, llava:13b) against
the 1,000-image study manifest (.venv/data/vqa_v2/approved_study_1000_manifest.jsonl).
Measures:
  1. Baseline VQA accuracy (consensus score)
  2. Behavioral invariance across probe variants (photometric JPEG)
  3. Generalization robustness across held-out variants (geometric translation)
  4. Failure flip rate
  5. Latency per image
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_experiment import resolve_path, dataset_answer_score


def get_installed_vision_models(ollama_url: str = "http://localhost:11434") -> list[str]:
    """Query local Ollama server and return only models with 'vision' capability."""
    try:
        res = requests.get(f"{ollama_url}/api/tags", timeout=10)
        res.raise_for_status()
        data = res.json()
        vision_models = []
        for m in data.get("models", []):
            caps = m.get("capabilities", [])
            if "vision" in caps:
                vision_models.append(m["name"])
        return vision_models
    except Exception as exc:
        print(f"Warning: Failed to query Ollama API: {exc}")
        return ["llama3.2-vision:11b", "llava:13b"]


def query_ollama(model: str, image_path: Path, question: str, ollama_url: str = "http://localhost:11434") -> str:
    """Send image and question to Ollama VLM and return trimmed answer."""
    with image_path.open("rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "Answer the question directly using only a single word or a short phrase.\n"
        f"Question: {question}\n"
        "Answer:"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 16,
        },
    }

    resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json().get("response", "").strip()
    # Normalize clean single line answer
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    first_line = lines[0] if lines else raw
    if ":" in first_line and len(first_line.split(":")[0]) < 10:
        first_line = first_line.split(":", 1)[1].strip()
    return first_line.rstrip(".")


def evaluate_model_on_manifest(
    model: str,
    manifest_rows: list[dict],
    manifest_path: Path,
    data_root: Path,
    output_path: Path,
    max_samples: int | None = None,
) -> dict:
    """Evaluate one Ollama vision model across the manifest."""
    print(f"\n{'='*70}\nEvaluating Model: {model}\n{'='*70}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = {}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed[str(r["sample_id"])] = r

    target_rows = manifest_rows[:max_samples] if max_samples else manifest_rows
    done_count = len(completed.keys() & {str(r['sample_id']) for r in target_rows})
    print(f"Total samples to evaluate: {len(target_rows)}. Already done: {done_count}")

    records = list(completed.values())
    count = len(completed)

    with output_path.open("a", encoding="utf-8") as sink:
        for idx, row in enumerate(target_rows, 1):
            sid = str(row["sample_id"])
            if sid in completed:
                continue

            img_path = resolve_path(row["image"], manifest_path, data_root)
            q = row["question"]
            gold = row.get("candidate_answers") or row.get("answers")
            ds = row.get("dataset")

            t0 = time.perf_counter()
            try:
                base_ans = query_ollama(model, img_path, q)
            except Exception as e:
                print(f"  [Error] {sid} baseline failed: {e}")
                continue

            base_score = dataset_answer_score(base_ans, gold, ds)

            # Evaluate variants
            variants_results = []
            probe_matches = []
            heldout_matches = []

            for v in row.get("variants", []):
                v_path = resolve_path(v["image"], manifest_path, data_root)
                v_type = v.get("family", "probe" if "jpeg" in v.get("name", "").lower() else "heldout")
                try:
                    v_ans = query_ollama(model, v_path, q)
                except Exception as e:
                    v_ans = ""
                v_score = dataset_answer_score(v_ans, gold, ds)
                matches_base = (v_ans.lower().strip() == base_ans.lower().strip())
                v_res = {
                    "name": v.get("name"),
                    "family": v_type,
                    "answer": v_ans,
                    "score": v_score,
                    "matches_baseline": matches_base,
                }
                variants_results.append(v_res)
                if "probe" in v_type or "jpeg" in v.get("name", "").lower():
                    probe_matches.append(matches_base)
                else:
                    heldout_matches.append(v_score is not None and v_score >= 0.5)

            elapsed = time.perf_counter() - t0

            # Generalization failure: was baseline correct (>=0.5) but failed on any heldout variant (<0.5)?
            heldout_failure = None
            if base_score is not None and base_score >= 0.5 and heldout_matches:
                heldout_failure = int(not all(heldout_matches))

            rec = {
                "sample_id": sid,
                "dataset": ds,
                "analysis_split": row.get("analysis_split"),
                "question": q,
                "model": model,
                "baseline_answer": base_ans,
                "baseline_score": base_score,
                "probe_invariant": all(probe_matches) if probe_matches else None,
                "heldout_failure": heldout_failure,
                "variants": variants_results,
                "latency_seconds": elapsed,
            }

            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
            records.append(rec)
            count += 1
            print(f"  [{count}/{len(target_rows)}] {sid} ({ds}): Ans='{base_ans}', Score={base_score}, HeldoutFail={heldout_failure}, Time={elapsed:.2f}s", flush=True)

    # Compute aggregate summary
    eligible = [r for r in records if (r.get("baseline_score") or 0.0) >= 0.5]
    fails = [r for r in eligible if r.get("heldout_failure") == 1]
    invariants = [r for r in eligible if r.get("probe_invariant") is True]

    summary = {
        "model": model,
        "total_evaluated": len(records),
        "baseline_accuracy": len(eligible) / max(len(records), 1),
        "eligible_baseline_correct": len(eligible),
        "probe_invariance_rate": len(invariants) / max(len(eligible), 1),
        "heldout_failure_rate": len(fails) / max(len(eligible), 1),
        "mean_latency_seconds": sum(r["latency_seconds"] for r in records) / max(len(records), 1),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run comparative Ollama vision benchmark.")
    parser.add_argument("--samples", type=int, default=None, help="Limit number of samples for quick comparison.")
    parser.add_argument("--wait-for-study", action="store_true", help="Wait for active study orchestrator to finish.")
    parser.add_argument("--models", nargs="+", default=None, help="Explicit list of Ollama models to test.")
    args = parser.parse_args()

    manifest_path = ROOT / ".venv/data/vqa_v2/approved_study_1000_manifest.jsonl"
    data_root = ROOT / ".venv/data/vqa_v2"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_rows = [json.loads(line) for line in f if line.strip()]

    # Filter to vision-capable models
    available_vision = get_installed_vision_models()
    print(f"Installed Ollama vision models: {available_vision}")

    if args.models:
        target_models = [m for m in args.models if m in available_vision]
        skipped = [m for m in args.models if m not in available_vision]
        if skipped:
            print(f"Skipping non-vision / uninstalled models: {skipped}")
    else:
        # Default to 11B to 14B vision models
        target_models = [m for m in ["llama3.2-vision:11b", "llava:13b"] if m in available_vision]

    if not target_models:
        print("Error: No vision models found to evaluate.")
        return

    print(f"Candidate Vision Models to Benchmark: {target_models}")

    if args.wait_for_study:
        print("Checking if primary study orchestrator is running...")
        import subprocess
        while True:
            proc = subprocess.run(
                ["powershell", "-Command", "Get-Process python -ErrorAction SilentlyContinue | Where-Object { (Get-CimInstance Win32_Process -Filter \"ProcessId = $($_.Id)\").CommandLine -match 'run_fresh_1000_study.py' }"],
                capture_output=True, text=True
            )
            if not proc.stdout.strip():
                print("Primary study orchestrator has finished! Proceeding to Ollama benchmark.")
                break
            print("Primary study orchestrator is still active. Waiting 60s...")
            time.sleep(60)

    # Verify each model on sample 1 before launching full evaluation
    sample_probe_row = manifest_rows[0]
    sample_probe_img = resolve_path(sample_probe_row["image"], manifest_path, data_root)
    sample_probe_q = sample_probe_row["question"]

    verified_models = []
    for model in target_models:
        print(f"\nProbing functionality for {model}...")
        try:
            test_ans = query_ollama(model, sample_probe_img, sample_probe_q)
            print(f"  [PASS] {model} verified functional (Response: '{test_ans}')")
            verified_models.append(model)
        except Exception as exc:
            print(f"  [DISCARD] {model} failed verification probe ({exc}). Skipping as not helpful.")

    if not verified_models:
        print("Error: No verified functional vision models available.")
        return

    print(f"\nRunning benchmark on verified models: {verified_models}")

    summaries = []
    for model in verified_models:
        clean_name = model.replace(":", "_").replace("/", "_")
        out_file = ROOT / "outputs" / "ollama_benchmark" / f"{clean_name}_results.jsonl"
        summary = evaluate_model_on_manifest(
            model=model,
            manifest_rows=manifest_rows,
            manifest_path=manifest_path,
            data_root=data_root,
            output_path=out_file,
            max_samples=args.samples,
        )
        summaries.append(summary)

    # Print Final Comparison Table
    print("\n" + "=" * 80)
    print(" FINAL COMPARATIVE BENCHMARK: OLLAMA VISION MODELS ")
    print("=" * 80)
    print(f"{'Model':<24} | {'VQA Acc':<9} | {'Probe Invariance':<18} | {'Held-out Fail':<14} | {'Latency':<8}")
    print("-" * 80)
    for s in summaries:
        print(f"{s['model']:<24} | {s['baseline_accuracy']*100:<8.1f}% | {s['probe_invariance_rate']*100:<17.1f}% | {s['heldout_failure_rate']*100:<13.1f}% | {s['mean_latency_seconds']:<6.2f}s")
    print("=" * 80)

    report_path = ROOT / "outputs" / "ollama_benchmark" / "comparative_vision_benchmark_report.json"
    report_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nComparative report saved to: {report_path}")


if __name__ == "__main__":
    main()
