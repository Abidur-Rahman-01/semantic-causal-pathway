# Benchmark data card and current run audit

This checkout contains artifacts from distinct 200-, 500-, and 1,000-image studies. They are not one pooled experiment. See [`outputs/run_index.json`](../outputs/run_index.json) for per-run provenance and artifact paths.

## Intended use

Evaluate VLM answer and evidence-use behavior on human-reviewed image transformations, with an optional white-box track for activation-patching-derived metrics. Results apply only to the declared dataset sample, model revision, scorer, intervention, and transformation families.

## Data and annotation

The project references VQAv2 and GQA examples. Source image redistribution is subject to each dataset's license. A release should include source IDs and retrieval instructions where images cannot legally be redistributed. Evidence regions, controls, candidate answer bins, and variant decisions require reviewer identity/date and an audit trail. Detector/segmenter output is a proposal, not an approval or ground-truth label.

## Splits and analysis unit

Use `image_id` as the split and bootstrap group. The prior studies report one question per image. All variants, masks, controls, and predictions for an image stay in one analysis split. The new schema records both the source dataset split and analysis split.

## Known audit limitations in this checkout

- The archived 200-image study includes a manifest and raw results, but its report and evaluation JSON disagree on some metrics. A separate AUROC=1.0 evaluation file has no matching auditable inputs and must not be cited as confirmed.
- The two 500-image summary JSON files disagree materially on the locked-test SCC AUROC. The corresponding manifest and raw per-sample predictions are not present.
- The 1,000-image report and summary JSON are present, but the raw Qwen prediction JSONL and manifest cited by the report are absent. LLaVA raw predictions and comparison outputs are also absent.
- Consequently, the run index distinguishes reported summaries from independently reproducible results. Restore raw artifacts before using the word “verified” or making a priority claim.

## Scoring

The inference scripts use `dataset_answer_score` for VQAv2 consensus and GQA scoring. Record the scorer's code revision with each run. Do not treat a small custom sample as an official benchmark leaderboard score. The held-out correctness-failure outcome and answer-flip outcome are distinct.

## Known limitations

Transformations and deletion interventions can introduce artifacts. Activation patching identifies effects under a particular model, prompt, tokenization, hook location, and intervention; it does not establish a complete reasoning circuit. CPS, CECA, and SCC are candidate metrics whose construct validity and predictive value remain open questions.
