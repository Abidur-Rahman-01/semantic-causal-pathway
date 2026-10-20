# Image Evidence Stability Benchmark Protocol

**Protocol version:** 1.0 (implementation baseline)  
**Status:** protocol template; a run becomes confirmatory only after its manifest, roles, model, scorer, and analysis config are frozen and hashed before inference.

## Research question

When a vision-language model gives the same answer after a meaning-preserving image change, does it still rely on the visual evidence needed to answer the question, and does that reliance persist under a different, held-out transformation?

The benchmark evaluates behavior under the declared images, questions, interventions, and transformations. It does not by itself establish general out-of-distribution performance, hallucination detection, human-like reasoning, or a complete causal account of model computation.

## Units, eligibility, and outcomes

- **Primary unit:** `image_id`. A benchmark release must use one question per image for the confirmatory analysis. If multiple questions per image are included in an extension, all inference, splits, and bootstrap resampling remain grouped by image.
- **Eligibility:** determine from the original-image prediction using the dataset's specified scorer. The default confirmatory cohort is examples with original score `>= 0.5`. Report all examples and eligible examples separately. Do not call a baseline-incorrect example a transformation failure.
- **Primary outcome:** among eligible examples, `heldout_failure = 1` if the score on any human-approved held-out variant is below `0.5`; otherwise it is `0`. If there is no approved held-out variant or a score is unavailable, the outcome is undefined and the reason is recorded.
- **Secondary outcome:** answer flip from the original answer on any approved held-out variant, reported separately from correctness failure.
- A model may change its answer and remain correct, or retain its answer and be incorrect; these outcomes must not be conflated.

## Transformation roles

- **Probe variants** are human-reviewed, meaning-preserving changes used for answer consistency and white-box pathway measurements. Probe families and exact transformations are declared in the run config before inference.
- **Held-out variants** belong to transformation families disjoint from the probe families. They are used only for final outcome measurement. Do not use held-out predictions to select metrics, thresholds, transformations, or eligibility rules.
- The schema-v1 white-box track requires at least two accepted probe variants, two accepted held-out variants, and three accepted, reviewed spatial controls per sample. A study with different minimums must declare them in a versioned protocol/config before inference.
- Every proposal remains in the review audit. Approval requires explicit decisions for answerability, evidence preservation, relation preservation, artifact introduction, reviewer identity, and review date. A failed or missing gate means the variant is not approved.
- The current default example is photometric JPEG probes and geometric translations as held-out variants. This is a specific transformation-family evaluation, not a general OOD test.

## Leakage, splits, and data integrity

- Assign `train`, `validation`, and `test` at the `image_id` level before model inference. Every question, image variant, evidence mask, control mask, and prediction for an image inherits that split.
- Freeze `candidate_answers` before model inference and without consulting gold answers or model predictions. Preserve the review record and timestamp.
- Probe/held-out roles and family assignments are frozen before inference and must be disjoint within a sample and across the declared run protocol.
- Record canonical image SHA-256 hashes. A duplicate image hash may not occur across analysis splits; duplicate source IDs and duplicate sample IDs are errors.
- Fit/calibrate on train only; use validation only for choices declared in the frozen config; evaluate test once. This tooling cannot technically prevent a researcher from reopening test results, so the run ledger must record access and deviations.

## Tracks and candidate metrics

### Track A: black-box

Report original-image score, raw answer, probe predictions and answer consistency, held-out predictions and scores, held-out failure, held-out answer flip, and available confidence/entropy. If the service supports edited-image queries, report evidence-deletion effects with the intervention documented.

### Track B: white-box

Track B includes Track A and may additionally report mediator-search configuration, mediator sets, hard and weighted CPS, CECA, SCC, intervention and mask types, undefined counts/reasons, forward-pass count, wall time, and peak memory. CPS, CECA, and SCC are candidate measures; no predictive value is assumed. Undefined values remain null and are never silently imputed as perfect or failed scores.

## Analysis and reporting

- Report sample flow and outcome prevalence by dataset, model, and split.
- Report AUROC, average precision (AUPRC), Brier score, and image-grouped bootstrap 95% percentile intervals. Report failure prevalence alongside AUPRC and the number of valid bootstrap replicates. AUROC is null when a cohort or resample has one outcome class.
- The sampling unit for confidence intervals is `image_id`. When several rows share an image, sample image groups with replacement and include all their rows each time.
- Predictor comparisons are paired by image. Any significance test, multiplicity adjustment, and primary comparison must be prespecified; point-estimate differences alone do not establish superiority.
- Report official scorer name/version. VQAv2 consensus and GQA scoring are distinct; a custom subset is not an official leaderboard result.
- Hash and record protocol, config, manifest, code commit, model/checkpoint revision, and scorer version in every run report.

## Claim boundary

The primary claim is about a measured answer/evidence behavior under approved transformations. The benchmark does not establish that an identified head set is the model's complete reasoning circuit. Activation patching estimates effects under its intervention and token-alignment assumptions. Descriptive dissociation between output and mediator metrics is not evidence that CPS identifies true internal reasoning. Any predictive claim requires a fresh locked evaluation and comparison against declared baselines.
