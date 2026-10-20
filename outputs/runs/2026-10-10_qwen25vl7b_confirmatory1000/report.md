# Benchmark Run & Evaluation Report

This report was automatically compiled according to **Image Evidence Stability Benchmark Protocol v1.0**.
Every reported metric is traced directly to locked manifests, predictions, and grouped image bootstrap resampling.

## Table 1: Run Metadata & Checkpoint Revisions

| Attribute | Value |
|---|---|
| Run ID | `2026-10-10_qwen25vl7b_confirmatory1000` |
| Model & Revision | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Scorer Version | `vqav2_soft_consensus_v2.0_gqa_v1.2` |
| Analysis Unit | `image_id` (zero leakage across splits) |
| Code Commit | `c3b3859bfee3c3d17e27700f8a1ee9c5a202e784` |
| Predictions File SHA-256 | `7dd0e2058322fa8e...` |
| Manifest SHA-256 | `b53672346a7f5cf3...` |
| Config SHA-256 | `8ea8e22e0a3246f9...` |
| Timestamp (UTC) | `2026-10-20T23:17:22.092869+00:00` |
| Bootstrap Resamples | 100 (seed=17) |

## Table 2: Benchmark Sample Flow & Exclusions

| Stage | Count | Notes |
|---|---:|---|
| Proposed Candidate Variants | 4000 | Generated before review |
| Human-Reviewed Variants | 4000 | Evaluated across 4 quality gates |
| Accepted Variants | 4000 | All 4 gates passed + disjoint family |
| Rejected Variants | 0 | Logged with explicit rejection reason |
| Unreviewed Variants | 0 | Excluded from evaluation |
| Evaluated Prediction Samples | 1000 | Unique image-question instances |
| Eligible (Original Correct >= 0.5) | 610 | Cohort for primary failure analysis |
| Baseline Incorrect (Excluded) | 390 | Not counted as transformation failures |

## Table 8: Rejected Variant Counts & Reasons

| Rejection Reason | Count |
|---|---:|

## Table 3 & 4: Performance & Failure Rates by Cohort

| Split:Dataset | Total N | Baseline Acc | Eligible N | Probe Consist. | Failures | Failure Prev. | Flips | Flip Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test:GQA:Qwen/Qwen2.5-VL-7B-Instruct | 200 | 54.0% | 108 | 0.292 | 13 | 12.0% | 89 | 82.4% |
| test:VQA v2:Qwen/Qwen2.5-VL-7B-Instruct | 200 | 71.0% | 142 | 0.297 | 14 | 9.9% | 132 | 93.0% |
| test:all:Qwen/Qwen2.5-VL-7B-Instruct | 400 | 62.5% | 250 | 0.295 | 27 | 10.8% | 221 | 88.4% |
| train:GQA:Qwen/Qwen2.5-VL-7B-Instruct | 200 | 59.5% | 119 | 0.302 | 12 | 10.1% | 102 | 85.7% |
| train:VQA v2:Qwen/Qwen2.5-VL-7B-Instruct | 200 | 59.5% | 119 | 0.253 | 19 | 16.0% | 105 | 88.2% |
| train:all:Qwen/Qwen2.5-VL-7B-Instruct | 400 | 59.5% | 238 | 0.278 | 31 | 13.0% | 207 | 87.0% |
| validation:GQA:Qwen/Qwen2.5-VL-7B-Instruct | 100 | 57.0% | 57 | 0.240 | 5 | 8.8% | 47 | 82.5% |
| validation:VQA v2:Qwen/Qwen2.5-VL-7B-Instruct | 100 | 65.0% | 65 | 0.240 | 9 | 13.8% | 58 | 89.2% |
| validation:all:Qwen/Qwen2.5-VL-7B-Instruct | 200 | 61.0% | 122 | 0.240 | 14 | 11.5% | 105 | 86.1% |

## Table 5: Predictor Performance (AUROC, AUPRC, Brier, CIs)

### Cohort: `test:GQA:Qwen/Qwen2.5-VL-7B-Instruct` (Eligible N=108, Failures=13)

| Predictor | N | AUROC (95% CI) | AUPRC (95% CI) | Raw Brier | Train-Calib Brier | Paired Δ AUROC vs Low-SCC (95% CI) |
|---|---:|---|---|---:|---:|---|
| probe_behavioral_inconsistency | 108 | 0.652 (0.529, 0.744) | 0.168 (0.086, 0.261) | 0.495 | 0.103 | +0.185 (-0.015, +0.383) |
| low_confidence | 108 | 0.599 (0.481, 0.719) | 0.150 (0.090, 0.267) | 0.165 | 0.106 | +0.134 (-0.064, +0.329) |
| normalized_entropy | 108 | 0.587 (0.486, 0.700) | 0.144 (0.093, 0.255) | 0.224 | 0.106 | +0.119 (-0.067, +0.311) |
| external_necessity | 108 | 0.547 (0.426, 0.691) | 0.206 (0.096, 0.442) | 0.117 | 0.106 | +0.096 (-0.104, +0.296) |
| control_adjusted_necessity | 108 | 0.449 (0.284, 0.605) | 0.159 (0.076, 0.341) | 0.119 | 0.106 | -0.010 (-0.246, +0.267) |
| low_cps | 108 | 0.459 (0.341, 0.569) | 0.117 (0.076, 0.190) | 0.812 | 0.106 | -0.023 (-0.072, +0.019) |
| low_scc | 90 | 0.495 (0.366, 0.611) | 0.121 (0.062, 0.207) | 0.814 | 0.108 | Reference |

### Cohort: `test:VQA v2:Qwen/Qwen2.5-VL-7B-Instruct` (Eligible N=142, Failures=14)

| Predictor | N | AUROC (95% CI) | AUPRC (95% CI) | Raw Brier | Train-Calib Brier | Paired Δ AUROC vs Low-SCC (95% CI) |
|---|---:|---|---|---:|---:|---|
| probe_behavioral_inconsistency | 142 | 0.641 (0.534, 0.741) | 0.133 (0.076, 0.194) | 0.518 | 0.089 | +0.253 (-0.036, +0.481) |
| low_confidence | 142 | 0.498 (0.385, 0.622) | 0.116 (0.067, 0.274) | 0.182 | 0.096 | +0.137 (-0.088, +0.359) |
| normalized_entropy | 142 | 0.522 (0.402, 0.654) | 0.109 (0.066, 0.203) | 0.238 | 0.095 | +0.171 (-0.059, +0.372) |
| external_necessity | 142 | 0.513 (0.349, 0.642) | 0.119 (0.064, 0.258) | 0.101 | 0.093 | +0.136 (-0.105, +0.326) |
| control_adjusted_necessity | 142 | 0.508 (0.341, 0.678) | 0.123 (0.056, 0.280) | 0.101 | 0.093 | +0.134 (-0.131, +0.345) |
| low_cps | 142 | 0.384 (0.245, 0.541) | 0.080 (0.046, 0.138) | 0.839 | 0.093 | +0.006 (-0.022, +0.045) |
| low_scc | 111 | 0.380 (0.237, 0.623) | 0.089 (0.060, 0.164) | 0.834 | 0.104 | Reference |

### Cohort: `test:all:Qwen/Qwen2.5-VL-7B-Instruct` (Eligible N=250, Failures=27)

| Predictor | N | AUROC (95% CI) | AUPRC (95% CI) | Raw Brier | Train-Calib Brier | Paired Δ AUROC vs Low-SCC (95% CI) |
|---|---:|---|---|---:|---:|---|
| probe_behavioral_inconsistency | 250 | 0.646 (0.554, 0.714) | 0.148 (0.094, 0.210) | 0.508 | 0.095 | +0.211 (+0.115, +0.358) |
| low_confidence | 250 | 0.544 (0.455, 0.637) | 0.116 (0.080, 0.204) | 0.174 | 0.100 | +0.126 (-0.003, +0.244) |
| normalized_entropy | 250 | 0.553 (0.464, 0.633) | 0.117 (0.079, 0.184) | 0.232 | 0.100 | +0.136 (+0.002, +0.251) |
| external_necessity | 250 | 0.526 (0.419, 0.617) | 0.132 (0.083, 0.243) | 0.108 | 0.098 | +0.102 (-0.074, +0.262) |
| control_adjusted_necessity | 250 | 0.479 (0.374, 0.576) | 0.124 (0.075, 0.235) | 0.109 | 0.098 | +0.055 (-0.149, +0.255) |
| low_cps | 250 | 0.433 (0.346, 0.549) | 0.096 (0.072, 0.134) | 0.827 | 0.099 | +0.002 (-0.026, +0.025) |
| low_scc | 201 | 0.447 (0.342, 0.544) | 0.101 (0.066, 0.151) | 0.825 | 0.105 | Reference |

## Table 6: White-Box Metric Availability & Undefined Accounting

| Metric | Defined N | Undefined N | Defined % | Notes / Reason Codes |
|---|---:|---:|---:|---|
| `cps_hard` | 1000 | 0 | 100.0% | Undefined preserved as null; no zero-imputation |
| `cps_weighted` | 1000 | 0 | 100.0% | Undefined preserved as null; no zero-imputation |
| `ceca` | 802 | 198 | 80.2% | Undefined preserved as null; no zero-imputation |
| `scc` | 802 | 198 | 80.2% | Undefined preserved as null; no zero-imputation |

## Table 7: Compute Cost & Model Calls

- Mean model forward calls per sample: `395.945`
- Mean wall seconds per sample: `114.90268154149997`

## Essential Figures

All figures below were generated deterministically from the saved data files:
- `figures/sample_flow.png`: Sample flow and quality exclusion funnel
- `figures/transformation_examples.png`: Reviewed transformation variants and evidence masks
- `figures/baseline_performance_and_probe_consistency.png`: Original score vs probe consistency
- `figures/heldout_failure_rate.png`: Held-out failure rate by split and cohort
- `figures/behavior_consistency_vs_evidence_necessity.png`: Behavioral consistency vs external evidence necessity
- `figures/predictive_performance_roc_prc.png`: ROC and Precision-Recall evaluation curves
- `figures/calibration_reliability_plot.png`: Platt-scaled calibration reliability diagram
- `figures/ablations_paired_differences.png`: Paired ablation effect differences
- `figures/whitebox_metric_availability.png`: White-box metric availability accounting
