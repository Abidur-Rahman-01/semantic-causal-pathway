# FCCT / IRI reproduction and the R-FCCT reliability extension

This repository contains two things:

* **Part A, FCCT/IRI reproduction** (`fcct.py`, `src/fcct/`). An independent reconstruction of
  *Causal Tracing of Object Representations in Large Vision Language Models: Mechanistic
  Interpretability and Hallucination Mitigation* (Li et al., AAAI-26,
  [arXiv:2511.05923v3](https://arxiv.org/abs/2511.05923v3)) on LLaVA-1.5-7B with COCO POPE. The
  authors' repository is empty, so unpublished settings are reconstructed and documented.
* **Part B, R-FCCT** (`rfcct.py`, `src/rfcct/`). An extension that leaves FCCT's definitions and
  IRI's equations untouched. It addresses the limitations found in Part A: contrastive
  (present + absent object) decision-margin tracing, stable causal layer selection, held-out
  strength tuning, protocol-matched evaluation, and prior-shift controls.

**Read first:** [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md) covers the paper's novelty,
the reproduction verdict, the seven issues found, related peer-reviewed work, and the R-FCCT
hypotheses with their failure criteria.

## Headline status

| | Result |
|---|---|
| Part A pipeline | Ran end to end on LLaVA-1.5-7B POPE (9,000 paired answers, 1,371 traces); integrity verified |
| Paper reproduced? | **No.** Greedy baseline is 4–6 pp above the paper; the IRI gain is +0.3/+1.5/+2.1 pp versus the paper's +6.5/+5.8/+6.2 |
| Main explanation found | The paper's "Regular" POPE row equals the VCD paper's *sampling* baseline to two decimals |
| Part B | Implemented and tested end to end on a tiny model (42/42 tests). **Not yet run on the 7B model; no R-FCCT results exist yet** |

## Hardware and runtime

* NVIDIA GPU with ≥ 17 GiB free memory (FP16 LLaVA-1.5-7B with eager attention). Tested on RTX 3090
  (24 GB) and RTX 5080. Apple Silicon and CPU are only for the unit tests.
* Disk: about 14 GB model + about 1 GB images + outputs (the Part A trace file alone is about 110 MB).
* Runtimes below are rough planning figures derived from forward-pass counts, not measurements:
  Part A takes about 1–2 days. Part B (8 stages) takes about 12–16 GPU-hours on a 3090: tracing about
  181k forward passes, tuning about 145k, greedy evaluation about 42k short generations, sampling about
  54k. Every stage resumes: rerun the same command after an interruption.

## 1. Installation

Linux / macOS (bash):

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

RTX 50-series (Blackwell) GPUs need a CUDA 12.8 build (the archived run used
`torch 2.11.0+cu128`); install from `https://download.pytorch.org/whl/cu128` instead.
Below, `python` means the virtual-environment interpreter (`.venv/bin/python` or
`.\.venv\Scripts\python.exe`).

## 2. Assets (shared by Parts A and B)

```bash
python scripts/setup_fcct.py --paper --pope --annotations-fast --model   # paper PDF, POPE, COCO val2014 annotations, pinned LLaVA weights
python fcct.py prepare --download-images                                 # 500 tracing + 500 POPE images, frozen manifests
python -m unittest discover -s tests -v                                  # 42 tests: math, hooks, tiny-model pipelines
```

The model is pinned to `llava-hf/llava-1.5-7b-hf@b234b804…` with SHA-256 locks
(`docs/fcct_model.lock.json`). The Hugging Face cache lives in `.venv/hf_home`.

## 3. Part A — FCCT/IRI reproduction

```bash
python fcct.py check          # assets, checksums, GPU memory
python -u fcct.py run         # trace -> profile -> paired regular/IRI POPE -> report
python fcct.py ablations --all   # optional: Table 6 controls (about 12 x 3,000 generations)
```

Outputs: `outputs/fcct/{trace.jsonl, profile.json, pope.jsonl}` and
`outputs/fcct/report/{report.md, comparison.json, fcct_curves.png}`.
The completed run is archived under `analysis/latest_archive/` (git-ignored) and audited by
`analysis/audit_latest.py`, whose output is `analysis/latest_audit_metrics.json`.
Part B's `paper_iri` variant reads `outputs/fcct/profile.json`. If you only have the archive, copy
`analysis/latest_archive/outputs/fcct/profile.json` to `outputs/fcct/profile.json` first.

## 4. Part B — R-FCCT pipeline

One command runs everything in order:

```bash
python rfcct.py prepare --download-images   # contrastive-trace / calibration / test manifests (+200 new COCO images)
python -u rfcct.py run                      # all stages below, then the report
```

Or stage by stage (same result; each stage checks that its inputs are complete and unchanged):

| # | Command | Data used | Output (`outputs/rfcct/`) | Purpose |
|---|---|---|---|---|
| 1 | `rfcct.py noise` | 200 tracing questions | `noise_calibration.json` | Pick σ by a pre-declared "measurable drop" criterion |
| 2 | `rfcct.py trace` | 500 tracing images, about 2,742 questions (present + absent) | `contrastive_trace.jsonl` | Clean/corrupted/patched decision margins, last-token MHSA & MLP, 32 layers |
| 3 | `rfcct.py select` | traces | `plans.json` | Ratio-of-means scores, bootstrap stability, reachable plans for 5 variants |
| 4 | `rfcct.py tune` | 200 calibration images (1,200 Q) | `tune_margins.jsonl` | Score the λ grid for every variant |
| 5 | `rfcct.py freeze` | tune results | `frozen.json` | Choose λ (max accuracy, recall loss ≤ 1 pp) and threshold controls |
| 6 | `rfcct.py evaluate` | POPE test (9,000 Q; about 6,000 unique prompts) | `greedy.jsonl` | Greedy generation + first-token margin for 7 methods |
| 7 | `rfcct.py sample` | POPE test | `sampling.jsonl` | VCD protocol: temperature 1, top-p 1, top-k 50, 3 seeds |
| 8 | `rfcct.py report` | all of the above | `report/report.md`, `report/results.json` | Tables, paired McNemar + cluster-bootstrap CIs, Holm, paper comparison |

`--limit N` on stages 2, 4, 6 and 7 processes N more items, for quick checks. Later stages refuse
to start until earlier ones are complete. `rfcct.py run --skip-sampling` omits stage 7.
To recompute a stage after changing settings, delete that stage's output file. Resumption is
refused, never silently mixed, when code, config, inputs or environment change.

**Data roles never overlap:** tracing images, calibration images and POPE test images are
disjoint (checked in `prepare`). Every choice (σ, layers, λ, thresholds) is made before the
test set is touched.

### Methods compared

| Method | Layers from | λ |
|---|---|---|
| `regular` | — | — |
| `paper_iri@paper` | Part A RR profile, paper rule | paper (0.26, 0.16) |
| `paper_iri@tuned` | same | calibration-tuned |
| `faithful_rr@tuned` | paper Eq. 2 restricted to measurable drops | tuned |
| `riri_present_only@tuned` | R-FCCT without absent objects | tuned |
| `riri_no_stability@tuned` | R-FCCT without bootstrap | tuned |
| `riri@tuned` | **full R-FCCT** | tuned |
| `threshold_control` | regular decoding + a calibrated yes/no margin threshold | — |

The primary, Holm-corrected comparisons are `riri@tuned` against `regular`, `paper_iri@paper`,
`paper_iri@tuned`, and `threshold_control`. All settings live in
[`configs/rfcct_llava15.json`](configs/rfcct_llava15.json). Freeze it before running.

## 5. Interpreting results

* Compare against the *protocol-matched* baseline. The paper's Regular numbers come from sampling
  and should be compared with the sampling section of the report, not with greedy rows.
* Report recall and yes-rate alongside accuracy. A method that only says "No" more often is matched
  by `threshold_control`.
* POPE splits share their 1,500 positive questions, so the three splits are not independent
  replications.
* Hypotheses and pre-declared failure conditions are in `docs/RESEARCH_REPORT.md` §5.

## Repository layout

```text
fcct.py, src/fcct/            Part A: FCCT tracing, IRI, POPE evaluation, report (unchanged)
rfcct.py, src/rfcct/          Part B: data.py (manifests), tracing.py (contrastive tracing, noise),
                              stats.py (estimators, bootstrap, McNemar, Holm), plan.py (PlannedIRI),
                              pipeline.py (stages), report.py
configs/                      fcct_llava15.json (Part A), rfcct_llava15.json (Part B),
                              fcct_paper_targets.json, fcct_ablation_plan.json
scripts/setup_fcct.py         Pinned downloads with checksums
tests/                        test_fcct.py (30), test_rfcct.py (12)
docs/RESEARCH_REPORT.md       Findings, literature, hypotheses
docs/fcct_reproduction.md     Part A protocol details and reconstruction choices
analysis/                     Audit of the completed Part A run
data/, outputs/               Generated (git-ignored)
```
