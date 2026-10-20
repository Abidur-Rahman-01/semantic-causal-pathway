# Benchmark Run & Evaluation Report

This report was automatically compiled according to **Image Evidence Stability Benchmark Protocol v1.0**.
Every reported metric is traced directly to locked manifests, predictions, and grouped image bootstrap resampling.

## Table 1: Run Metadata & Checkpoint Revisions

| Attribute | Value |
|---|---|
| Run ID | `2026-10-06_qwen25vl7b_confirmatory500` |
| Model & Revision | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Scorer Version | `vqav2_soft_consensus_v2.0_gqa_v1.2` |
| Analysis Unit | `image_id` (zero leakage across splits) |
| Code Commit | `c3b3859bfee3c3d17e27700f8a1ee9c5a202e784` |
| Predictions File SHA-256 | `ab904f6464f2a0d8...` |
| Manifest SHA-256 | `0d59ca4b1e7a96ef...` |
| Config SHA-256 | `a0a86b3f94a703fc...` |
| Timestamp (UTC) | `2026-10-20T23:16:26.190490+00:00` |
| Bootstrap Resamples | 100 (seed=17) |

## Table 2: Benchmark Sample Flow & Exclusions

| Stage | Count | Notes |
|---|---:|---|
| Proposed Candidate Variants | 2000 | Generated before review |
| Human-Reviewed Variants | 2000 | Evaluated across 4 quality gates |
| Accepted Variants | 2000 | All 4 gates passed + disjoint family |
| Rejected Variants | 0 | Logged with explicit rejection reason |
| Unreviewed Variants | 0 | Excluded from evaluation |
| Evaluated Prediction Samples | 500 | Unique image-question instances |
| Eligible (Original Correct >= 0.5) | 303 | Cohort for primary failure analysis |
| Baseline Incorrect (Excluded) | 197 | Not counted as transformation failures |

## Table 8: Rejected Variant Counts & Reasons

| Rejection Reason | Count |
|---|---:|

## Table 3 & 4: Performance & Failure Rates by Cohort

| Split:Dataset | Total N | Baseline Acc | Eligible N | Probe Consist. | Failures | Failure Prev. | Flips | Flip Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test:GQA:Qwen/Qwen2.5-VL-7B-Instruct | 50 | 56.0% | 28 | 0.360 | 2 | 7.1% | 22 | 78.6% |
| test:VQA v2:Qwen/Qwen2.5-VL-7B-Instruct | 50 | 58.0% | 29 | 0.290 | 7 | 24.1% | 25 | 86.2% |
| test:all:Qwen/Qwen2.5-VL-7B-Instruct | 100 | 57.0% | 57 | 0.325 | 9 | 15.8% | 47 | 82.5% |
| train:GQA:Qwen/Qwen2.5-VL-7B-Instruct | 150 | 61.3% | 92 | 0.290 | 10 | 10.9% | 75 | 81.5% |
| train:VQA v2:Qwen/Qwen2.5-VL-7B-Instruct | 150 | 66.7% | 100 | 0.250 | 10 | 10.0% | 91 | 91.0% |
| train:all:Qwen/Qwen2.5-VL-7B-Instruct | 300 | 64.0% | 192 | 0.270 | 20 | 10.4% | 166 | 86.5% |
| validation:GQA:Qwen/Qwen2.5-VL-7B-Instruct | 50 | 48.0% | 24 | 0.220 | 2 | 8.3% | 20 | 83.3% |
| validation:VQA v2:Qwen/Qwen2.5-VL-7B-Instruct | 50 | 60.0% | 30 | 0.290 | 4 | 13.3% | 26 | 86.7% |
| validation:all:Qwen/Qwen2.5-VL-7B-Instruct | 100 | 54.0% | 54 | 0.255 | 6 | 11.1% | 46 | 85.2% |

## Table 5: Predictor Performance (AUROC, AUPRC, Brier, CIs)

### Cohort: `test:GQA:Qwen/Qwen2.5-VL-7B-Instruct` (Eligible N=28, Failures=2)

| Predictor | N | AUROC (95% CI) | AUPRC (95% CI) | Raw Brier | Train-Calib Brier | Paired Δ AUROC vs Low-SCC (95% CI) |
|---|---:|---|---|---:|---:|---|
| probe_behavioral_inconsistency | 28 | 0.615 (0.341, 0.825) | 0.092 (0.048, 0.273) | 0.464 | 0.067 | +0.488 (+0.076, +0.856) |
| low_confidence | 28 | 0.500 (0.270, 0.739) | 0.101 (0.053, 0.334) | 0.144 | 0.070 | +0.333 (+0.059, +0.543) |
| normalized_entropy | 28 | 0.519 (0.270, 0.796) | 0.106 (0.053, 0.362) | 0.197 | 0.070 | +0.333 (+0.059, +0.543) |
| external_necessity | 28 | 0.750 (0.409, 1.000) | 0.567 (0.062, 1.000) | 0.061 | 0.068 | +0.595 (+0.336, +0.855) |
| control_adjusted_necessity | 28 | 0.904 (0.741, 1.000) | 0.643 (0.132, 1.000) | 0.061 | 0.068 | +0.738 (+0.576, +0.909) |
| low_cps | 28 | 0.154 (0.037, 0.333) | 0.062 (0.037, 0.170) | 0.873 | 0.068 | +0.024 (+0.000, +0.079) |
| low_scc | 23 | 0.167 (0.000, 0.364) | 0.077 (0.044, 0.208) | 0.861 | 0.081 | Reference |

### Cohort: `test:VQA v2:Qwen/Qwen2.5-VL-7B-Instruct` (Eligible N=29, Failures=7)

| Predictor | N | AUROC (95% CI) | AUPRC (95% CI) | Raw Brier | Train-Calib Brier | Paired Δ AUROC vs Low-SCC (95% CI) |
|---|---:|---|---|---:|---:|---|
| probe_behavioral_inconsistency | 29 | 0.672 (0.494, 0.845) | 0.327 (0.168, 0.665) | 0.362 | 0.201 | +0.008 (-0.371, +0.262) |
| low_confidence | 29 | 0.630 (0.347, 0.821) | 0.374 (0.152, 0.727) | 0.193 | 0.202 | +0.017 (-0.274, +0.237) |
| normalized_entropy | 29 | 0.623 (0.352, 0.809) | 0.355 (0.146, 0.697) | 0.229 | 0.201 | -0.008 (-0.280, +0.210) |
| external_necessity | 29 | 0.591 (0.366, 0.776) | 0.292 (0.150, 0.549) | 0.239 | 0.203 | -0.067 (-0.406, +0.169) |
| control_adjusted_necessity | 29 | 0.344 (0.164, 0.638) | 0.255 (0.103, 0.500) | 0.240 | 0.203 | -0.269 (-0.607, +0.108) |
| low_cps | 29 | 0.545 (0.314, 0.772) | 0.261 (0.123, 0.488) | 0.720 | 0.204 | +0.000 (-0.042, +0.069) |
| low_scc | 24 | 0.622 (0.425, 0.819) | 0.366 (0.187, 0.610) | 0.682 | 0.235 | Reference |

### Cohort: `test:all:Qwen/Qwen2.5-VL-7B-Instruct` (Eligible N=57, Failures=9)

| Predictor | N | AUROC (95% CI) | AUPRC (95% CI) | Raw Brier | Train-Calib Brier | Paired Δ AUROC vs Low-SCC (95% CI) |
|---|---:|---|---|---:|---:|---|
| probe_behavioral_inconsistency | 57 | 0.649 (0.488, 0.765) | 0.211 (0.121, 0.327) | 0.412 | 0.135 | +0.121 (-0.160, +0.425) |
| low_confidence | 57 | 0.616 (0.452, 0.763) | 0.226 (0.137, 0.458) | 0.169 | 0.137 | +0.079 (-0.143, +0.274) |
| normalized_entropy | 57 | 0.604 (0.442, 0.758) | 0.209 (0.127, 0.387) | 0.213 | 0.136 | +0.061 (-0.170, +0.283) |
| external_necessity | 57 | 0.694 (0.534, 0.860) | 0.336 (0.143, 0.597) | 0.151 | 0.137 | +0.158 (-0.160, +0.445) |
| control_adjusted_necessity | 57 | 0.498 (0.302, 0.711) | 0.296 (0.104, 0.578) | 0.152 | 0.137 | -0.044 (-0.381, +0.392) |
| low_cps | 57 | 0.472 (0.331, 0.709) | 0.151 (0.085, 0.273) | 0.795 | 0.137 | +0.009 (-0.025, +0.038) |
| low_scc | 47 | 0.529 (0.271, 0.750) | 0.212 (0.077, 0.443) | 0.770 | 0.160 | Reference |

## Table 6: White-Box Metric Availability & Undefined Accounting

| Metric | Defined N | Undefined N | Defined % | Notes / Reason Codes |
|---|---:|---:|---:|---|
| `cps_hard` | 500 | 0 | 100.0% | Undefined preserved as null; no zero-imputation |
| `cps_weighted` | 500 | 0 | 100.0% | Undefined preserved as null; no zero-imputation |
| `ceca` | 393 | 107 | 78.6% | Undefined preserved as null; no zero-imputation |
| `scc` | 393 | 107 | 78.6% | Undefined preserved as null; no zero-imputation |

## Table 7: Compute Cost & Model Calls

- Mean model forward calls per sample: `393.46`
- Mean wall seconds per sample: `112.2361887240006`

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
