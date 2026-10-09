import json
from pathlib import Path

import sys

target_files = [Path(sys.argv[1])] if len(sys.argv) > 1 else [
    Path(".venv/outputs/semantic_pathway_results.jsonl"),
    Path("outputs/gqa_pathway/semantic_pathway_results.jsonl"),
]

for path in target_files:
    if not path.exists():
        continue
    print(f"\n{'#' * 60}\nRESULTS FROM: {path}\n{'#' * 60}")
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        print("=" * 60)
        print(f"Sample ID: {r['sample_id']}")
        print(f"Question: {r['question']}")
        print(f"Model Baseline Answer: {r['baseline_answer']}")
        print(f"Gold Human Answers: {r.get('gold_answers', [])[:5]}")
        print(f"Baseline Candidate Distribution: {r['baseline_candidate_distribution']}")
        ext = r['external_intervention']
        for k, v in ext.items():
            if isinstance(v, (int, float)):
                print(f"  External {k}: {v:.4f}")
        print(f"CECA (mean distribution alignment): {r['ceca_distribution_mean']:.4f}")
        print(f"CECA (scalar target diagnostic): {r['ceca_scalar_mean_diagnostic']:.4f}")
        print(f"CPS (Hard Head-Set Jaccard): {r['cps_hard']:.4f}")
        print(f"CPS (Weighted Restoration Consistency): {r['cps_weighted']:.4f}")
        print(f"SCC Score (CECA x weighted CPS): {r['scc']:.4f}")
        print(f"Variant Flip / Failure: {r['semantic_variant_failure']}")
        print("\nVariant Pathways:")
        for v in r['variant_pathways']:
            t = v['transform']
            ans = v['answer']
            match = v['answer_matches_baseline']
            ceca_d = f"{v['ceca_distribution']:.4f}" if v['ceca_distribution'] is not None else "None"
            num_med = len(v['mediator_recovery']['mediators'])
            print(f"  - {t:<22} -> answer: {ans!r:<12} (match={match}) | CECA: {ceca_d} | Mediators: {num_med}")
        
        cost = r['cost']
        peak_gb = cost.get('cuda_peak_memory_bytes', 0) / (1024 ** 3)
        print(f"\nResource Cost: Forward passes: {cost['model_forward_calls']}, Wall time: {r['wall_seconds']:.1f}s, Peak VRAM: {peak_gb:.2f} GB")
