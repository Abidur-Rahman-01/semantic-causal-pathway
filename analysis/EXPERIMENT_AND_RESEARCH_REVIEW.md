# Latest FCCT/IRI experiment audit and research direction

Reviewed 12 November 2026. Source archive: `outputs.zip`, SHA-256 `9eed6d9a293671db15884127302955853c9f62e33532e3eee0a6399ae419ec9d`.

## Verdict

The archive supplies evidence of a completed supported LLaVA-1.5-7B independent reconstruction: 1,371 tracing records and 9,000 paired POPE predictions. It supersedes the older status file saying no scientific run was completed. The main experiment operated and produced real, auditable prediction artifacts. It did not meet the whole paper's experimental scope, establish exact reproduction, or reproduce the full magnitude of the reported improvements.

Archive files were extracted to `analysis/latest_archive/`, preserving the preexisting outputs. Archive timestamps place the completed main report on 9 November 2026; they are timestamps supplied by the archive, not independent execution attestation.

## Independent verification

`analysis/audit_latest.py` recomputes metrics from answer text, checks IDs, compares stored hashes, examines tracing statistics, and performs image-cluster bootstrap analysis. Results are in `analysis/latest_audit_metrics.json`.

- 9,000 unique evaluation IDs, 3,000 per split, 500 evaluation image IDs, no empty generations.
- 1,371 unique tracing IDs; 1,300 completed, 70 excluded because the object is outside the crop, one excluded for an undefined/filtered gap.
- Zero overlap between tracing and evaluation image IDs.
- Prediction/report hash, profile/report hash, trace/profile hash, trace-metadata/profile hash, and evaluation/profile hash all agree.
- Trace and evaluation metadata identify the same code hashes.
- Every recomputed split/method confusion matrix and metric exactly matches comparison.json.
- A separate full patch arithmetic check verified 849,696 numeric RR values with zero discrepancies; 23,904 patch records have null values. All 1,300 final-layer last-token hidden restorations exactly recover their recorded clean probabilities. Total patch records: 873,600.

These checks establish internal consistency. The archive does not include original evaluation/tracing manifests, images, model weights, or the run's source snapshot. Thus image/reference-label correctness and correspondence of recorded source hashes to the executed source cannot be independently established here. Hash consistency is not proof of execution history. The current source passed 30 tests in the preceding review, but that does not validate the archived source revision.

## Measured results

| POPE split | Baseline accuracy | IRI accuracy | Delta pp | Baseline F1 | IRI F1 | Delta pp |
|---|---:|---:|---:|---:|---:|---:|
| Random | 89.03 | 89.33 | +0.30 | 88.96 | 88.94 | -0.02 |
| Popular | 85.60 | 87.07 | +1.47 | 85.99 | 86.90 | +0.91 |
| Adversarial | 79.20 | 81.27 | +2.07 | 80.95 | 82.08 | +1.13 |

Mean split accuracy improves 84.61 to 85.89 (+1.28 pp). Mean split F1 improves 85.30 to 85.97 (+0.67 pp). These are descriptive means; questions and images are shared across splits, so 9,000 rows are not 9,000 independent observations.

Image-cluster bootstrap, 10,000 resamples, seed 17, gives accuracy-change 95% percentile intervals: random [-0.30,+0.93] pp; popular [+0.77,+2.20]; adversarial [+1.33,+2.80]. These intervals describe image sampling uncertainty conditional on this one run, not corruption-seed/model/protocol uncertainty. The random-split improvement is not clearly distinguishable from zero under this analysis.

## Comparison with the paper

Published values are from [Table 2 of the frozen paper](https://arxiv.org/html/2511.05923v3).

| Split | Paper IRI accuracy | Local IRI accuracy | Local minus paper | Paper baseline-to-IRI accuracy gain | Local gain |
|---|---:|---:|---:|---:|---:|
| Random | 89.76 | 89.33 | -0.43 pp | +6.47 pp | +0.30 pp |
| Popular | 87.67 | 87.07 | -0.60 pp | +5.79 pp | +1.47 pp |
| Adversarial | 85.17 | 81.27 | -3.90 pp | +6.21 pp | +2.07 pp |

Random and popular IRI accuracy/F1 are within the repository's one-point tolerance. Adversarial IRI accuracy/F1 miss by 3.90/2.10 pp. Only five of twelve baseline/IRI accuracy/F1 comparisons pass the configured tolerance. Tolerance is a local agreement criterion, not statistical equivalence or exact reproduction.

Baseline mismatch is substantial: random accuracy is 5.74 pp above the paper and popular accuracy 3.72 pp above it. Similar final IRI scores therefore do not imply reproduction of the intervention effect. Prompt, runtime, decoding, evaluation aggregation, and independently chosen tracing inputs are possible explanations, not established causes.

## The intervention trades false positives for false negatives

In each split, true positives decrease 1,326 to 1,287 and false negatives increase 174 to 213. Recall drops 88.4% to 85.8% (-2.6 pp). False-positive counts improve:

| Split | Baseline FP | IRI FP | Corrected answers | Newly wrong answers |
|---|---:|---:|---:|---:|
| Random | 155 | 107 | 48 | 39 |
| Popular | 258 | 175 | 83 | 39 |
| Adversarial | 450 | 349 | 101 | 39 |

Corrected answers means baseline-wrong/IRI-correct count. Net gains after subtracting harms are respectively 9, 44, and 62 questions. Shared positive-question results across splits must not be counted as independent replications.

The binary outputs show fewer false object-presence claims, accompanied by more missed present objects. They do not alone establish reduced free-form hallucination, improved visual grounding, or retained descriptive coverage. A conservative answer shift is compatible with the observed pattern; causal attribution needs additional controls.

## Coverage of the whole paper

| Paper expectation | Evidence in newest archive | Assessment |
|---|---|---|
| Figure 3, LLaVA tracing | 32 layers, three components, seven groups, 873,600 patch records | Executed; inputs independently reconstructed and estimates unstable |
| Figure 3, Qwen-VL-Chat | None | Not met |
| Figure 4, 3,000-VQA attention analysis | No attention artifacts | Not met |
| LLaVA POPE | All 9,000 paired answers | Executed; partial numerical agreement |
| Other paper model results | None | Not met |
| MME | None | Not met |
| CHAIR captions/evaluation | None | Not met |
| MMHal-Bench judge evaluation | None | Not met |
| MHumanEval human evaluation | None | Not met |
| Table 6 ablations | Without-MLP case has only five questions; remaining cases not run or unsupported | Not met |
| VCD/OPERA/PAI/VTI comparisons | None | Not met |
| Table 7 matched latency | No matched timing study | Not met |
| Preservation of other abilities | No relevant completed evaluation | Not established |

The five-question ablation's 100% score has no evidential value for reproducing Table 6. Old status/preflight summaries in the archive predate the full run and should not determine completion status.

## Recovery-rate instability: a concrete finding

RR is `(patched-corrupted)/(clean-corrupted)`. Keeping signed nonzero denominators follows the implemented equation; it also produces a fragile mean when denominators approach zero. A large RR is not automatically an error or proof of a strong typical effect.

- Of 1,300 completed questions, 369 have negative corruption gaps and 931 positive gaps. Noise improves affirmative probability for approximately 28.4% of completed questions, complicating the interpretation of “recovery.”
- Thirteen completed questions have absolute gaps below 1e-5; 117 are below .001.
- For attention layer 13, sample trace-39357-77 has gap -0.00000309944 and patched-minus-corrupted probability -0.0662816, producing RR +21,385.02. Its positive ratio accompanies a decrease in affirmative probability, not successful recovery of damaged performance in the usual positive-gap sense.
- Removing image 39357 changes layer 13's image-mean RR from +6.291 to -2.268.
- The attention source top three are [14,13,17] by descending score. Removing the largest 1% absolute image-mean contributions separately per layer changes them to [19,30,24]. This is a sensitivity diagnostic, not a proposed validated estimator.
- In 2,000 image bootstrap resamples, original attention sources 13/17 appear in the top three only 63.85%/52.35% of the time. MLP source 31 appears only 46.75% of the time.
- Attention layer 14 has mean RR 27.76; multiplying by .26 gives a nominal source coefficient around 7.22 before norm preservation. Final normalization preserves magnitude, not intervention direction or semantic fidelity.
- MLP source layer 31 has no strictly later target layer. It cannot contribute under the causal-order rule; only two of the three selected MLP sources can actually inject.

These observations do not disprove the original paper. They identify sensitivity of this reconstruction and a plausible reason to investigate estimator reliability before tuning injection strength. Establishing that estimator instability causes benchmark degradation requires new controlled runs.

## Recent peer-reviewed literature and novelty boundaries

Search cutoff: 12 November 2026. This is a targeted search, not an exhaustive systematic review. Publisher-access restrictions limited several articles to indexed publisher abstracts/excerpts; no unavailable full-text evidence is claimed.

| Work / venue | Verified date or status | What it already covers | Implication |
|---|---|---|---|
| [MLS-Steer, Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231226001773) | 7 Apr 2026; DOI 10.1016/j.neucom.2026.132780 | Multi-scale visual interventions and adaptive cosine-weighted fusion | Multi-scale adaptive steering alone is insufficient novelty |
| [Causal Decoding for Hallucination-Resistant MLLMs, TMLR](https://openreview.net/pdf/ab0a7b2df406569b130e33d21f831de5e10a59b3.pdf) | Published Feb 2026 on accepted PDF | Causal decoding targeting spurious object mentions | Causal decoding alone is already covered |
| [Hallucination backtracking, Scientific Reports](https://www.nature.com/articles/s41598-026-52390-1) | 4 Jun 2026 | Visual attention trajectories, origin localization, regeneration; discusses domain recalibration | Token-aware correction and adaptive grounding signals are established directions |
| [CoSE, Information Fusion](https://www.sciencedirect.com/science/article/pii/S156625352600357X) | Publisher listing Nov 2026; DOI 10.1016/j.inffus.2026.104478 | Semantic manifold connectivity and retrieved latent conditioning for lightweight models | Generic geometry-aware enhancement is crowded |
| [Find what you missed, Knowledge-Based Systems](https://www.sciencedirect.com/science/article/pii/S0950705126013158) | 27 Sep 2026; DOI 10.1016/j.knosys.2026.116589 | Weak causal signals for visual-token pruning; explicitly distinguishes those from strict causal identification | Token dependency recovery alone is not a clean new contribution |

For novelty checks, peer-reviewed conferences must also be included even though the requested primary emphasis is journals. [Causal Route Gating, ICML 2026](https://proceedings.mlr.press/v306/cheng26d.html) already separates visual and text routes with decision-aligned interventions. [Mass-Tail Attribution Entropy, Findings of IJCNLP-AACL, 6–10 Nov 2026](https://aclanthology.org/2026.findings-aacl.174/) explicitly studies task-dependent hallucination regimes and recall-preserving decoding. Thus “adaptive IRI that preserves recall” is too broad as a novelty claim.

## Recommended niche

**Reliability of causal layer selection under weak and reversed visual corruption, with intervention evaluated at matched object recall.**

Candidate title: *When Recovery Rates Mislead: Stable Causal Layer Selection for Vision–Language Interventions*.

Research question: Does an uncertainty-aware estimate of layer causal utility transfer better across corruption distributions and improve hallucination metrics at matched recall, compared with ranking raw mean probability ratios?

The proposed distinction is uncertainty in the causal-effect estimator and ranking, not merely predictive uncertainty, extra visual attention, or another steering rule. This search did not establish a paper resolving this exact combination; novelty remains provisional until broader full-text and citation-chain review.

Three contributions could form a coherent study:

1. **A reproducible measurement study.** Separate positive, reversed, and weak-gap corruption; measure leave-one-image influence, rank confidence, corruption-seed stability, precision sensitivity, and transfer between corruption families. Include controlled object edits as a complement to Gaussian noise; report editing artifacts and avoid calling synthetic corruption a faithful natural counterfactual by assumption.
2. **A stable selection method.** Compare raw means, medians/trimmed estimators, uncertainty-aware shrinkage, and a regularized pooled contrast such as `sum(d_i * n_i)/(sum(d_i*d_i)+lambda)`, where d is clean-minus-corrupted and n patched-minus-corrupted. This pooled slope weights strong gaps differently and estimates a different quantity than mean RR; its assumptions must be explicit. Require a later reachable target for every source and select source-target pairs jointly. Tune only on disjoint calibration data.
3. **A behavioral validation.** Compare hallucination at matched recall, matched answer-positive rate, and matched compute. Measure both corrections and newly introduced errors. Use threshold/calibration-only controls to test whether gains exceed conservative answer bias. Any yes/no probability scoring requires new score logging; archived answer text cannot supply calibrated confidence or ECE.

Keep ratio robustness as the primary contribution. Reachable source-target selection and recall matching are necessary controls, not three independent claims of algorithmic novelty.

## Feasible experimental plan

**Phase A: existing artifacts, no new model inference.** Reproduce this audit; inspect influential examples; bootstrap all layer rankings; compare estimators descriptively. Do not select the winning method on the existing POPE test scores.

**Phase B: small controlled pilot.** Select 100–150 calibration images, disjoint from POPE. Use at least three noise seeds and several corruption strengths; stratify positive/reversed/weak gaps. Limit initial interventions to last-token attention/MLP, then expand after confirming the mechanism. On a precision subset compare FP16 with higher precision where hardware permits, because very small probability differences need numerical scrutiny.

**Phase C: frozen evaluation.** Freeze estimator, layer selection, hyperparameters, and gating on calibration data. Evaluate LLaVA plus a second distinct model family; COCO POPE plus captioning with official CHAIR/AMBER-style coverage metrics and a held-out visual domain. Group all resampling and splits by image; shared positives across POPE splits are not independent data. Include baseline IRI, random matched interventions, reachable-pair-only IRI, robust-estimator-only IRI, and calibration/threshold-only controls. Add current causal/adaptive baselines where code and compute permit. Report latency and offline calibration cost separately.

**Success criterion, declared before running:** reduced hallucination at a fixed recall budget, more stable source rankings across independently generated corruptions, and gains beyond threshold-only controls on a second model/domain. A +1 pp held-out F1 improvement with no more than .5 pp recall loss could be a pilot target, not a guaranteed result. Use image-cluster intervals and distinguish dataset uncertainty from repeated-run uncertainty.

**Falsification:** if stable rankings do not improve held-out behavior, or if probability-threshold adjustment reproduces the gain, do not claim improved grounding. A careful negative result about unreliable mechanistic estimators can still be valuable, but it needs multiple models, corruption families, and controlled tests.

## Extension path

After validating the core result: spatial/attribute/relation hallucination; domain-shifted images; quantization-induced ranking changes; video temporal evidence; multilingual prompts. Each should test the same estimator-transfer hypothesis rather than accumulate unrelated features. Quantization is particularly practical: reuse one checkpoint at different precisions and ask whether selected intervention sites are stable, while separating changed model behavior from measurement noise.

The project now has a substantive experimental foundation. The strongest immediate research opportunity is to explain and remedy the observed instability without purchasing lower hallucination through reduced object recall.
