# Cleanup and migration

The repository now has one public entry point (`pathway.py`), one active study
configuration (`configs/pathway.json`), and one implementation of each scorer and
benchmark utility. The active implementation retains all existing core modules:
question requirements, Grounding DINO/SAM2, optional CLIP diagnostics, T1-T4 variants,
spatial controls, human-review application, external necessity/sufficiency,
attention-head activation capture and patching, greedy/exact mediator search,
CPS/CECA/SCC, grouped inference, calibration, failure prediction, and LoRA training.

Removed 58 files: duplicate release code, superseded sample-count-specific runners,
auto-approval dataset builders, fixed-path download/monitor/publishing helpers, a
prototype notebook, duplicate findings, and the separate legacy comparative analyzer.
The exact inventory is [cleanup_removed_files.json](cleanup_removed_files.json).
Historical experiment configurations moved to `archives/configs/`.
The original architecture, literature review, raw datasets, model weights, manifests,
annotations, benchmark run checksums, and raw results remain available.

| Previous workflow | Current command |
|---|---|
| Separate pilot/500/1000 inference runners | `python pathway.py run --config configs/pathway.json` |
| Auto-approved dataset builders | `data`, `sample`, proposal commands, and explicit human review commands |
| Separate model comparisons and paper analyzers | `python pathway.py report ...` |
| Hardcoded synthetic failure dataset | `python pathway.py failure-data PREDICTIONS --output FILE` |
| Hardcoded Ollama study | `python pathway.py ollama --manifest FILE --models MODEL` |
| Repeated CV training scripts | `folds`, followed by `train` for each fold |
| Copied benchmark scripts/scorers | Canonical `scripts/` and `scoring/` only |

Evaluation corrections made while consolidating:

- All maintained evaluators share `scoring/`. VQA consensus averages over reference
  holdouts; GQA no longer accepts substrings such as `not red` for the answer `red`.
- Ollama uses gold reference answers rather than candidate options, honors explicit
  roles, and emits a schema compatible with the shared report pipeline.
- Evaluation errors are not converted to model failure events. Existing Ollama
  predictions with incompatible provenance cannot be silently resumed.
- The primary runner fingerprints its manifest, config, and scorer versions before
  resuming. Fresh runs cannot append duplicates with `--no-resume`.
- Failure-predictor input uses observed labels and preserves image groups/splits.
- Annotation export preserves actual recorded review metadata and reasons.
- Summary statistics exclude undefined values rather than silently replacing them.
- New reports emit canonical filenames without duplicate aliases.

Historical metrics have not been regenerated or endorsed by this cleanup. Genuine
human reviews and a prior-exposure audit remain necessary for publication claims.
The conceptual architecture includes optional future repair; this cleanup does not
claim to implement capabilities that were absent from the original code.
