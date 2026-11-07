# Running and releasing the benchmark

Use [the project README](../README.md) for the current preparation/review workflow.
The study runner accepts native reviewed manifests and schema-v1 manifests. It
normalizes schema-v1 evidence, candidate-answer reviews, controls, and variant roles
without manufacturing reviewer identities or acceptance decisions.

For schema-v1 release inputs:

```powershell
python pathway.py split input.jsonl --output split.jsonl --seed 17
python pathway.py seal split.jsonl --data-root DATA_ROOT --output frozen_manifest.jsonl
python pathway.py validate frozen_manifest.jsonl --data-root DATA_ROOT
```

Set `manifest`, `data_root`, model revisions, and a new output directory in
`configs/pathway.json`. Run `python pathway.py run --check-only` before inference.
The default analysis phases are train/validation. Freeze scientific settings before
setting `allow_test_split=true` and invoking `run --split test`. Historical sample
exposure must be audited separately from within-manifest image isolation.

After inference:

```powershell
python pathway.py report predictions.jsonl --manifest frozen_manifest.jsonl --config configs/pathway.json --run-id RUN_ID --model-revision MODEL_COMMIT --output-dir outputs/runs/RUN_ID
python pathway.py verify outputs/runs/RUN_ID
```

The reporter produces `outcomes.csv`, `predictors.csv`, `metrics.json`, `report.json`,
`report.md`, figures, frozen inputs, and `checksums.sha256`. Undefined metrics remain
undefined; grouped bootstrap and train-only calibration are recorded with the run.
The current scorer label applies to newly evaluated predictions. For historical
predictions, explicitly pass their original `--scoring-version`; reporting does not
rerun inference or silently repair saved scores.

To share a release, include the source, frozen inputs, run directory, environment
requirements, and dataset/model licensing references. Refer to external source
datasets instead of redistributing restricted images. The `benchmark/` directory
contains historical artifacts; canonical source code is kept only at repository root.
