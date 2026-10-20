# Running and releasing the benchmark

This workflow adds a reproducible black-box track and optional white-box pathway measurements to the current Qwen pipeline. It does not run inference during preparation. Run the commands on the machine that has the dataset and model dependencies.

## Freeze a manifest

1. Start from image/question records with stable `sample_id`, `image_id`, dataset, image path, question, gold answers, and candidate answer bins reviewed before inference.
2. Generate variant candidates with `scripts/prepare_variants.py`. Review the image candidates and complete every review column, including `artifact_introduced`, reviewer, date, role, and rejection reason. Apply the sheet with `scripts/apply_variant_review.py`.
3. Convert the reviewed records to schema v1 using [`schemas/benchmark_manifest.schema.json`](../schemas/benchmark_manifest.schema.json). Schema v1 keeps auditable fields for reviewer/date, acceptance, rejection, role, family, and asset paths. Candidate answer metadata must identify the frozen protocol and reviewer.
4. Assign grouped splits before inference if the sample manifest has no frozen split:

   ```bash
   python scripts/benchmark_manifest.py split input.jsonl --output split.jsonl --data-root DATA_ROOT --seed 17 --train 0.6 --validation 0.2 --test 0.2 --summary-out split_summary.json
   ```

   This deterministically hashes `image_id` within each dataset. Do not regenerate splits after seeing model outcomes.
5. Seal assets and validate the immutable input:

   ```bash
   python scripts/benchmark_manifest.py seal split.jsonl --data-root DATA_ROOT --output frozen_manifest.jsonl
   python scripts/benchmark_manifest.py validate frozen_manifest.jsonl --data-root DATA_ROOT --require-schema-v1 --require-hashes --summary-out manifest_summary.json
   ```

Validation checks IDs, grouped split isolation, duplicate image content across splits, candidate bins, role/family disjointness, two approved variants per role, at least three reviewed controls, review fields, file existence, and image/mask dimensions. Keep the source manifest and all rejected candidates.

## Run inference

The existing `scripts/run_confirmatory_study.py` and `scripts/run_fresh_1000_study.py` now accept schema-v1 rows through a compatibility normalizer. Set the desired manifest, data root, model revision, run output directory, and protocol fields in a copied config. Avoid overwriting historical run artifacts. Run the black-box track using an API-compatible prediction JSONL if the model does not expose activations; the unified reporter accepts the same sample IDs and image groups.

The output includes the resolved model revision, raw baseline candidate log probabilities, probe factual/deleted/patched distributions, held-out answers and dataset scores, a separate answer-flip outcome, undefined-metric reasons, and compute metadata. The scorer currently used by the runner is `dataset_answer_score` in `scripts/run_experiment.py`: VQAv2 uses consensus scoring and GQA uses normalized exact/contained match. Record the code commit as the scorer version; this is a local evaluation scorer, not a leaderboard submission. Supply the resolved checkpoint revision from the prediction records to the report command.

## Rebuild reports

After inference, generate the predictor table and report from the sample-level JSONL:

```bash
python scripts/report_benchmark.py predictions.jsonl --manifest frozen_manifest.jsonl --config frozen_config.json --run-id RUN_ID --model-revision MODEL_REVISION --scoring-version dataset_answer_score@CODE_COMMIT --output-dir report
```

This writes `predictor_table.csv`, `benchmark_report.json`, and `benchmark_report.md`. It reports by dataset, model, analysis split, and question type; uses `image_id` grouped bootstrap; leaves AUROC undefined for one-class samples; records valid resample counts; and reports paired AUROC differences against low-SCC. AUROC/AUPRC use raw scores. Raw Brier is shown as a bounded-score diagnostic; train-only univariate logistic calibration is also fit for each predictor when both outcomes and enough examples are available. Validation/test calibrated Brier is out-of-sample; train calibrated Brier is in-sample and must not be presented as generalization performance.

## Ablations and release

The frozen starting plan is [`configs/benchmark_ablation_plan.json`](../configs/benchmark_ablation_plan.json). Give each ablation a unique config and output folder, preserve the same image splits, and compare paired outcomes by `image_id`. Do not select a best setting from the locked test results.

Package schema, manifests/IDs, review sheets, configs, scoring code, reports, and environment lockfiles. Follow VQAv2/GQA image redistribution terms; where image redistribution is not allowed, release source IDs and retrieval instructions and check that derived annotations may be shared. Document model weights and the extra activation access required by Track B.
