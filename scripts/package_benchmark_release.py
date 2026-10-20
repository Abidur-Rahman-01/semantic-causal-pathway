"""Package the complete benchmark release directory structure conforming to Section 14.

Creates:
benchmark/
  schemas/
  manifests/
  annotations/
  configs/
  scoring/
  scripts/
  outputs/
  README.md
  DATA_CARD.md
  MODEL_CARD.md
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark"


def package_benchmark():
    print(f"Creating benchmark release directory at {BENCHMARK_DIR}...")
    BENCHMARK_DIR.mkdir(exist_ok=True)

    # 1. schemas/
    schemas_dir = BENCHMARK_DIR / "schemas"
    schemas_dir.mkdir(exist_ok=True)
    shutil.copyfile(
        ROOT / "schemas" / "benchmark_manifest.schema.json",
        schemas_dir / "benchmark_manifest.schema.json",
    )

    # 2. manifests/
    manifests_dir = BENCHMARK_DIR / "manifests"
    manifests_dir.mkdir(exist_ok=True)
    manifest_sources = [
        ("pilot_200_manifest.jsonl", ROOT / "archives" / "pathway_200_run" / "approved_pilot_manifest.jsonl"),
        ("confirmatory_500_manifest.jsonl", ROOT / "data" / "confirmatory_500" / "approved_confirmatory_manifest.jsonl"),
        ("confirmatory_1000_manifest.jsonl", ROOT / "data" / "study_1000" / "approved_study_1000_manifest.jsonl"),
    ]
    for target_name, src_path in manifest_sources:
        if src_path.is_file():
            shutil.copyfile(src_path, manifests_dir / target_name)
            print(f"  Copied manifest: {target_name}")

    # 3. annotations/
    annotations_dir = BENCHMARK_DIR / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    for fname in ("variant_reviews.jsonl", "rejected_variants.jsonl"):
        src = ROOT / "annotations" / fname
        if src.is_file():
            shutil.copyfile(src, annotations_dir / fname)
            print(f"  Copied annotation: {fname}")

    # 4. configs/
    configs_dir = BENCHMARK_DIR / "configs"
    configs_dir.mkdir(exist_ok=True)
    for fname in ("benchmark_default.json", "benchmark_ablation_plan.json"):
        src = ROOT / "configs" / fname
        if src.is_file():
            shutil.copyfile(src, configs_dir / fname)
            print(f"  Copied config: {fname}")

    # 5. scoring/
    scoring_dir = BENCHMARK_DIR / "scoring"
    scoring_dir.mkdir(exist_ok=True)
    for fname in ("__init__.py", "dataset_scorers.py", "vqa_scorer.py", "gqa_scorer.py"):
        src = ROOT / "scoring" / fname
        if src.is_file():
            shutil.copyfile(src, scoring_dir / fname)
            print(f"  Copied scoring module: {fname}")

    # 6. scripts/
    scripts_dir = BENCHMARK_DIR / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    for fname in ("benchmark_manifest.py", "report_benchmark.py", "generate_benchmark_figures.py", "run_benchmark.py"):
        src = ROOT / "scripts" / fname
        if src.is_file():
            shutil.copyfile(src, scripts_dir / fname)
            print(f"  Copied script: {fname}")

    # 7. outputs/
    outputs_dir = BENCHMARK_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "outputs" / "run_index.json", outputs_dir / "run_index.json")

    runs_dest = outputs_dir / "runs"
    runs_dest.mkdir(exist_ok=True)
    for run_name in ("2026-10-05_qwen25vl7b_pilot200", "2026-10-06_qwen25vl7b_confirmatory500", "2026-10-10_qwen25vl7b_confirmatory1000"):
        src_run = ROOT / "outputs" / "runs" / run_name
        dest_run = runs_dest / run_name
        if src_run.is_dir():
            shutil.copytree(src_run, dest_run, dirs_exist_ok=True)
            print(f"  Copied run: {run_name}")

    print("Benchmark release files copied successfully!")


if __name__ == "__main__":
    package_benchmark()
