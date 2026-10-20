# Image Evidence Stability Benchmark: Complete Runbook & Research Package

This package is the self-contained runbook and execution framework for the **Image Evidence Stability Benchmark**. It is engineered so you can clone the repository, run the experiment on another computer, and trace every reported metric and figure back to locked manifests, raw predictions, and input checksums.

---

## 1. Research Question & Claim Boundaries

### Core Question
> *When a vision-language model gives the same answer after a meaning-preserving image change, does it still rely on the visual evidence needed to answer the question, and does that reliance persist under a different, held-out transformation?*

### Prespecified Benchmark Protocol
- **Primary Outcome:** Failure on at least one accepted held-out variant (`heldout_failure = 1` if score < 0.5 among baseline-correct examples).
- **Secondary Outcome:** Held-out answer flip (`heldout_answer_flip = 1` if answer string changes).
- **Eligibility Cohort:** Examples where original image prediction receives dataset score `>= 0.5`. (Baseline-incorrect examples are tracked separately and never counted as transformation failures).
- **Probe Variants:** Human-approved photometric changes (JPEG quality 92/98) used to evaluate answer consistency and white-box pathway metrics.
- **Held-out Variants:** Disjoint geometric translations (5% shifts) used exclusively to measure primary failure outcomes.
- **Split & Resampling Unit:** `image_id`. All questions, masks, controls, and variants for an image stay in the same split with zero cross-split leakage.
- **Candidate Pathway Metrics:** CPS, CECA, and SCC are evaluated as candidate measurements; empirical results indicate they do not outperform black-box baselines. Do not claim they predict failures without fresh locked confirmatory evidence.

---

## 2. Directory Structure

```
benchmark/
  ├── schemas/
  │   └── benchmark_manifest.schema.json      # Draft 2020-12 manifest JSON Schema
  ├── manifests/
  │   ├── pilot_200_manifest.jsonl            # 200-sample balanced manifest
  │   ├── confirmatory_500_manifest.jsonl     # 500-sample confirmatory manifest
  │   └── confirmatory_1000_manifest.jsonl    # 1,000-sample confirmatory manifest
  ├── annotations/
  │   ├── variant_reviews.jsonl               # 2,800 reviewed candidate variants across 4 gates
  │   └── rejected_variants.jsonl             # 2,000 rejected variants with audit reasons
  ├── configs/
  │   ├── benchmark_default.json              # Frozen benchmark protocol configuration
  │   └── benchmark_ablation_plan.json        # Prespecified ablation factors and questions
  ├── scoring/
  │   ├── __init__.py                         # Scoring module entry point
  │   ├── vqa_scorer.py                       # VQAv2 soft consensus scoring (v2.0)
  │   ├── gqa_scorer.py                       # GQA exact match & normalization scoring (v1.2)
  │   └── dataset_scorers.py                  # Scorer dispatch & version tracking
  ├── scripts/
  │   ├── benchmark_manifest.py               # Manifest validator, splitter, and hasher
  │   ├── report_benchmark.py                 # Full outcomes, predictors, metrics & report generator
  │   ├── generate_benchmark_figures.py       # Deterministic generator for all 9 essential figures
  │   └── run_benchmark.py                    # Unified CLI runner (validate, split, report, verify)
  ├── outputs/
  │   ├── run_index.json                      # Comprehensive provenance ledger of all runs
  │   └── runs/
  │       ├── 2026-10-05_qwen25vl7b_pilot200/
  │       ├── 2026-10-06_qwen25vl7b_confirmatory500/
  │       └── 2026-10-10_qwen25vl7b_confirmatory1000/
  ├── DATA_CARD.md                            # Comprehensive data provenance & review gates
  ├── MODEL_CARD.md                           # Model configurations, hooks & limitations
  └── README.md                               # This runbook
```

---

## 3. Step-by-Step Execution Guide

### Prerequisites
Install Python 3.10+ and dependencies:
```bash
pip install -r ../requirements.txt
# Requires: torch, transformers, pillow, scikit-learn, numpy, matplotlib
```

### Step 1: Validate Manifest & Review Gates
Check manifest IDs, split isolation, image and mask dimensions, and review approval gates:
```bash
python scripts/run_benchmark.py validate manifests/confirmatory_1000_manifest.jsonl
```

### Step 2: Deterministically Split by Image ID
Assign train/val/test splits safely at the `image_id` level:
```bash
python scripts/run_benchmark.py split manifests/confirmatory_1000_manifest.jsonl \
  --output manifests/confirmatory_1000_split.jsonl \
  --seed 17 --train 0.6 --val 0.2 --test 0.2
```

### Step 3: Run Model Inference (Locked Protocol)
To execute inference under frozen settings with Qwen2.5-VL-7B-Instruct:
```bash
python ../scripts/run_confirmatory_study.py \
  --config configs/benchmark_default.json \
  --manifest manifests/confirmatory_1000_manifest.jsonl \
  --output-dir outputs/runs/2026-10-21_qwen25vl7b_v1
```

### Step 4: Generate Outcomes, Predictors, Metrics, Figures & Reports
Once predictions are saved, run the reporting engine:
```bash
python scripts/run_benchmark.py report \
  outputs/runs/2026-10-21_qwen25vl7b_v1/predictions.jsonl \
  --manifest manifests/confirmatory_1000_manifest.jsonl \
  --config configs/benchmark_default.json \
  --output-dir outputs/runs/2026-10-21_qwen25vl7b_v1 \
  --run-id 2026-10-21_qwen25vl7b_v1 \
  --model-revision "Qwen/Qwen2.5-VL-7B-Instruct" \
  --bootstrap-repeats 2000 \
  --seed 17
```
This single command automatically generates:
1. `outcomes.csv`: Sample-level outcomes table
2. `predictors.csv`: Predictor table with black-box and white-box features
3. `metrics.json`: Bootstrap AUROC, AUPRC, Brier, CIs, and paired differences
4. `report.json`: Machine-readable metadata and metrics
5. `report.md`: Markdown summary report with all 8 tables
6. `checksums.sha256`: SHA-256 digests of all run artifacts
7. `figures/`: All 9 essential figures generated deterministically

### Step 5: Verify Run Integrity
Verify that every generated file matches its SHA-256 checksum:
```bash
python scripts/run_benchmark.py verify outputs/runs/2026-10-21_qwen25vl7b_v1
```

---

## 4. Run Directory Standard Layout

Every benchmark run directory contains:
```
outputs/runs/<RUN_ID>/
  ├── run_config.json                 # Frozen JSON run configuration
  ├── manifest.jsonl                  # Frozen evaluation manifest
  ├── predictions.jsonl               # Raw sample-level & variant predictions
  ├── outcomes.csv                    # Sample-level primary/secondary outcomes
  ├── predictors.csv                  # One row per sample with all candidate predictors
  ├── metrics.json                    # Cohort metrics with grouped bootstrap CIs
  ├── report.md                       # Formatted report with 8 essential tables
  ├── report.json                     # Structured evaluation metadata
  ├── logs/                           # Inference stdout/stderr logs
  ├── checksums.sha256                # Exact SHA-256 hashes of all artifacts
  └── figures/                        # 9 publication-grade figures
      ├── sample_flow.png
      ├── transformation_examples.png
      ├── baseline_performance_and_probe_consistency.png
      ├── heldout_failure_rate.png
      ├── behavior_consistency_vs_evidence_necessity.png
      ├── predictive_performance_roc_prc.png
      ├── calibration_reliability_plot.png
      ├── ablations_paired_differences.png
      └── whitebox_metric_availability.png
```

---

## 5. Summary of Historical Runs (Reconciled)

| Run ID | Dataset (N) | Eligible Test | Failures (%) | Probe Inconsist. AUROC | Low-SCC AUROC (95% CI) | Status |
|---|---:|---:|---:|---:|---:|---|
| `2026-10-05_qwen25vl7b_pilot200` | VQA+GQA (200) | 34 | 7 (20.6%) | 0.656 | 0.321 (0.124, 0.548) | Verified & Reconciled |
| `2026-10-06_qwen25vl7b_confirmatory500` | VQA+GQA (500) | 57 | 9 (15.8%) | 0.718 | 0.529 (0.337, 0.722) | Verified & Reconciled |
| `2026-10-10_qwen25vl7b_confirmatory1000` | VQA+GQA (1,000) | 250 | 27 (10.8%) | 0.638 | 0.447 (0.334, 0.551) | Verified & Reconciled |

**Core Finding:** Across all 3 independent cohorts, Low-SCC does not outperform black-box probe inconsistency in predicting held-out transformation failures.
