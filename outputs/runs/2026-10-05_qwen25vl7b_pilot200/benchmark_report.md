# Benchmark Run & Evaluation Report

This report was automatically compiled according to **Image Evidence Stability Benchmark Protocol v1.0**.
Every reported metric is traced directly to locked manifests, predictions, and grouped image bootstrap resampling.

## Table 1: Run Metadata & Checkpoint Revisions

| Attribute | Value |
|---|---|
| Run ID | `2026-10-05_qwen25vl7b_pilot200` |
| Model & Revision | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Scorer Version | `vqav2_soft_consensus_v2.0_gqa_v1.2` |
| Analysis Unit | `image_id` (zero leakage across splits) |
| Code Commit | `c3b3859bfee3c3d17e27700f8a1ee9c5a202e784` |
| Predictions File SHA-256 | `594076cd85c7faf2...` |
| Manifest SHA-256 | `176b2b0a2160920a...` |
| Config SHA-256 | `c771b3f37e8d28bb...` |
| Timestamp (UTC) | `2026-10-20T23:15:22.514714+00:00` |
| Bootstrap Resamples | 100 (seed=17) |

## Table 2: Benchmark Sample Flow & Exclusions

| Stage | Count | Notes |
|---|---:|---|
| Proposed Candidate Variants | 800 | Generated before review |
| Human-Reviewed Variants | 800 | Evaluated across 4 quality gates |
| Accepted Variants | 800 | All 4 gates passed + disjoint family |
| Rejected Variants | 0 | Logged with explicit rejection reason |
| Unreviewed Variants | 0 | Excluded from evaluation |
| Evaluated Prediction Samples | 200 | Unique image-question instances |
| Eligible (Original Correct >= 0.5) | 169 | Cohort for primary failure analysis |
| Baseline Incorrect (Excluded) | 31 | Not counted as transformation failures |

## Table 8: Rejected Variant Counts & Reasons

| Rejection Reason | Count |
|---|---:|

## Table 3 & 4: Performance & Failure Rates by Cohort

| Split:Dataset | Total N | Baseline Acc | Eligible N | Probe Consist. | Failures | Failure Prev. | Flips | Flip Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test:GQA:Qwen/Qwen2.5-VL-7B-Instruct | 19 | 73.7% | 14 | 0.368 | 1 | 7.1% | 11 | 78.6% |
| test:VQA v2:Qwen/Qwen2.5-VL-7B-Instruct | 21 | 95.2% | 20 | 0.214 | 6 | 30.0% | 17 | 85.0% |
| test:all:Qwen/Qwen2.5-VL-7B-Instruct | 40 | 85.0% | 34 | 0.287 | 7 | 20.6% | 28 | 82.4% |
| train:GQA:Qwen/Qwen2.5-VL-7B-Instruct | 59 | 79.7% | 47 | 0.280 | 13 | 27.7% | 39 | 83.0% |
| train:VQA v2:Qwen/Qwen2.5-VL-7B-Instruct | 61 | 91.8% | 56 | 0.205 | 25 | 44.6% | 49 | 87.5% |
| train:all:Qwen/Qwen2.5-VL-7B-Instruct | 120 | 85.8% | 103 | 0.242 | 38 | 36.9% | 88 | 85.4% |
| validation:GQA:Qwen/Qwen2.5-VL-7B-Instruct | 22 | 72.7% | 16 | 0.341 | 5 | 31.2% | 15 | 93.8% |
| validation:VQA v2:Qwen/Qwen2.5-VL-7B-Instruct | 18 | 88.9% | 16 | 0.361 | 3 | 18.8% | 15 | 93.8% |
| validation:all:Qwen/Qwen2.5-VL-7B-Instruct | 40 | 80.0% | 32 | 0.350 | 8 | 25.0% | 30 | 93.8% |

## Table 5: Predictor Performance (AUROC, AUPRC, Brier, CIs)

### Cohort: `test:GQA:Qwen/Qwen2.5-VL-7B-Instruct` (Eligible N=14, Failures=1)

| Predictor | N | AUROC (95% CI) | AUPRC (95% CI) | Raw Brier | Train-Calib Brier | Paired Δ AUROC vs Low-SCC (95% CI) |
|---|---:|---|---|---:|---:|---|
| probe_behavioral_inconsistency | 14 | 0.423 (0.250, 0.684) | 0.100 (0.083, 0.356) | 0.500 | 0.113 | +0.200 (-0.150, +0.562) |
| low_confidence | 14 | 0.615 (0.333, 0.846) | 0.167 (0.111, 0.500) | 0.144 | 0.109 | +0.300 (-0.050, +0.745) |
| normalized_entropy | 14 | 0.615 (0.333, 0.846) | 0.167 (0.111, 0.500) | 0.195 | 0.108 | +0.300 (-0.050, +0.745) |
| external_necessity | 14 | 0.385 (0.112, 0.615) | 0.111 (0.083, 0.333) | 0.071 | 0.108 | +0.000 (-0.363, +0.545) |
| control_adjusted_necessity | 14 | 0.385 (0.259, 0.500) | 0.071 (0.071, 0.214) | 0.071 | 0.108 | -0.300 (-0.600, -0.100) |
| low_cps | 14 | 0.231 (0.077, 0.504) | 0.091 (0.080, 0.333) | 0.829 | 0.108 | +0.000 (+0.000, +0.000) |
| low_scc | 11 | 0.300 (0.100, 0.600) | 0.125 (0.100, 0.375) | 0.810 | 0.129 | Reference |

### Cohort: `test:VQA v2:Qwen/Qwen2.5-VL-7B-Instruct` (Eligible N=20, Failures=6)

| Predictor | N | AUROC (95% CI) | AUPRC (95% CI) | Raw Brier | Train-Calib Brier | Paired Δ AUROC vs Low-SCC (95% CI) |
|---|---:|---|---|---:|---:|---|
| probe_behavioral_inconsistency | 20 | 0.577 (0.382, 0.760) | 0.337 (0.145, 0.596) | 0.525 | 0.227 | +0.333 (-0.065, +0.615) |
| low_confidence | 20 | 0.607 (0.369, 0.826) | 0.366 (0.201, 0.721) | 0.249 | 0.232 | +0.212 (-0.231, +0.575) |
| normalized_entropy | 20 | 0.679 (0.457, 0.882) | 0.427 (0.231, 0.879) | 0.206 | 0.240 | +0.303 (-0.231, +0.808) |
| external_necessity | 20 | 0.607 (0.342, 0.839) | 0.403 (0.195, 0.776) | 0.307 | 0.231 | +0.273 (-0.455, +0.733) |
| control_adjusted_necessity | 20 | 0.393 (0.186, 0.674) | 0.322 (0.166, 0.637) | 0.303 | 0.231 | +0.030 (-0.554, +0.568) |
| low_cps | 20 | 0.500 (0.210, 0.725) | 0.308 (0.139, 0.557) | 0.684 | 0.231 | +0.061 (-0.091, +0.333) |
| low_scc | 14 | 0.303 (0.020, 0.682) | 0.199 (0.077, 0.441) | 0.778 | 0.231 | Reference |

### Cohort: `test:all:Qwen/Qwen2.5-VL-7B-Instruct` (Eligible N=34, Failures=7)

| Predictor | N | AUROC (95% CI) | AUPRC (95% CI) | Raw Brier | Train-Calib Brier | Paired Δ AUROC vs Low-SCC (95% CI) |
|---|---:|---|---|---:|---:|---|
| probe_behavioral_inconsistency | 34 | 0.598 (0.428, 0.744) | 0.244 (0.094, 0.481) | 0.515 | 0.180 | +0.304 (+0.085, +0.598) |
| low_confidence | 34 | 0.630 (0.454, 0.790) | 0.266 (0.134, 0.526) | 0.206 | 0.181 | +0.238 (-0.147, +0.578) |
| normalized_entropy | 34 | 0.651 (0.497, 0.791) | 0.280 (0.158, 0.560) | 0.202 | 0.185 | +0.262 (-0.147, +0.661) |
| external_necessity | 34 | 0.550 (0.337, 0.806) | 0.255 (0.126, 0.546) | 0.210 | 0.180 | +0.155 (-0.346, +0.596) |
| control_adjusted_necessity | 34 | 0.442 (0.261, 0.753) | 0.224 (0.098, 0.500) | 0.208 | 0.180 | -0.060 (-0.378, +0.333) |
| low_cps | 34 | 0.481 (0.229, 0.687) | 0.205 (0.096, 0.397) | 0.744 | 0.181 | +0.048 (-0.019, +0.152) |
| low_scc | 25 | 0.321 (0.087, 0.574) | 0.141 (0.070, 0.366) | 0.792 | 0.186 | Reference |

## Table 6: White-Box Metric Availability & Undefined Accounting

| Metric | Defined N | Undefined N | Defined % | Notes / Reason Codes |
|---|---:|---:|---:|---|
| `cps_hard` | 200 | 0 | 100.0% | Undefined preserved as null; no zero-imputation |
| `cps_weighted` | 200 | 0 | 100.0% | Undefined preserved as null; no zero-imputation |
| `ceca` | 155 | 45 | 77.5% | Undefined preserved as null; no zero-imputation |
| `scc` | 155 | 45 | 77.5% | Undefined preserved as null; no zero-imputation |

## Table 7: Compute Cost & Model Calls

- Mean model forward calls per sample: `345.63157894736844`
- Mean wall seconds per sample: `92.07604927894825`

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
