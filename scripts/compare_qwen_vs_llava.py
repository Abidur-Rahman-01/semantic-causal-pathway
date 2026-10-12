"""Compare 1,000-image study results between Qwen2.5-VL-7B and Ollama LLaVA-13B."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load Qwen results
qwen_file = ROOT / ".venv/outputs/study_1000/models/qwen2_5_vl_7b/semantic_pathway_results.jsonl"
llava_file = ROOT / "outputs/ollama_benchmark/llava_13b_results.jsonl"

qwen_rows = [json.loads(line) for line in qwen_file.open(encoding="utf-8")]
llava_rows = [json.loads(line) for line in llava_file.open(encoding="utf-8")]

print("=" * 80)
print(" HEAD-TO-HEAD COMPARISON: QWEN2.5-VL-7B vs. OLLAMA LLAVA-13B (N = 1,000) ")
print("=" * 80)
print(f"Total Evaluated Instances: Qwen = {len(qwen_rows)}, LLaVA = {len(llava_rows)}")


def get_model_stats(rows, is_qwen=False):
    by_split = defaultdict(list)
    by_dataset = defaultdict(list)

    for r in rows:
        split = r.get("analysis_split")
        ds = r.get("dataset")
        by_split[split].append(r)
        by_dataset[ds].append(r)

    def compute(subset):
        if is_qwen:
            eligible = [r for r in subset if (r.get("vqa_consensus_score") or 0.0) >= 0.5]
            fails = [r for r in eligible if r.get("heldout_failure") == 1]
            lat = sum(r.get("wall_seconds", 0.0) for r in subset) / max(len(subset), 1)
        else:
            eligible = [r for r in subset if (r.get("baseline_score") or 0.0) >= 0.5]
            fails = [r for r in eligible if r.get("heldout_failure") == 1]
            lat = sum(r.get("latency_seconds", 0.0) for r in subset) / max(len(subset), 1)

        return {
            "total": len(subset),
            "correct": len(eligible),
            "accuracy": len(eligible) / max(len(subset), 1),
            "failures": len(fails),
            "failure_rate": len(fails) / max(len(eligible), 1),
            "latency": lat,
        }

    overall = compute(rows)
    splits = {s: compute(r_list) for s, r_list in by_split.items()}
    datasets = {d: compute(r_list) for d, r_list in by_dataset.items()}
    return {"overall": overall, "splits": splits, "datasets": datasets}


qwen_st = get_model_stats(qwen_rows, is_qwen=True)
llava_st = get_model_stats(llava_rows, is_qwen=False)

print("\n1. OVERALL COHORT PERFORMANCE (1,000 IMAGES):")
print("-" * 80)
print(f"{'Metric':<35} | {'Qwen2.5-VL-7B (PyTorch)':<20} | {'LLaVA-13B (Ollama)':<20}")
print("-" * 80)
print(f"{'Clean Baseline VQA Accuracy':<35} | {qwen_st['overall']['accuracy']*100:<19.1f}% | {llava_st['overall']['accuracy']*100:<19.1f}%")
print(f"{'Baseline Correct Instances':<35} | {qwen_st['overall']['correct']:<19d}  | {llava_st['overall']['correct']:<19d}")
print(f"{'Generalization Failure Rate':<35} | {qwen_st['overall']['failure_rate']*100:<19.1f}% | {llava_st['overall']['failure_rate']*100:<19.1f}%")
print(f"{'Total Failure Events':<35} | {qwen_st['overall']['failures']:<19d}  | {llava_st['overall']['failures']:<19d}")
print(f"{'Average Latency per Sample':<35} | {qwen_st['overall']['latency']:<18.1f}s | {llava_st['overall']['latency']:<18.1f}s")
print("-" * 80)

print("\n2. PERFORMANCE BY DATASET:")
print("-" * 80)
for ds in ["VQA v2", "GQA"]:
    q_d = qwen_st["datasets"].get(ds, {})
    l_d = llava_st["datasets"].get(ds, {})
    print(f"[{ds} - 500 Images]")
    print(f"  VQA Accuracy         : Qwen = {q_d.get('accuracy', 0)*100:.1f}%, LLaVA = {l_d.get('accuracy', 0)*100:.1f}%")
    print(f"  Held-out Failure Rate: Qwen = {q_d.get('failure_rate', 0)*100:.1f}%, LLaVA = {l_d.get('failure_rate', 0)*100:.1f}%")

print("\n3. PERFORMANCE BY PARTITION:")
print("-" * 80)
for sp in ["train", "validation", "test"]:
    q_s = qwen_st["splits"].get(sp, {})
    l_s = llava_st["splits"].get(sp, {})
    name = "Locked Test (400)" if sp == "test" else f"{sp.capitalize()} ({q_s.get('total', 0)})"
    print(f"[{name}]")
    print(f"  VQA Accuracy         : Qwen = {q_s.get('accuracy', 0)*100:.1f}%, LLaVA = {l_s.get('accuracy', 0)*100:.1f}%")
    print(f"  Held-out Failure Rate: Qwen = {q_s.get('failure_rate', 0)*100:.1f}%, LLaVA = {l_s.get('failure_rate', 0)*100:.1f}%")
print("=" * 80)
