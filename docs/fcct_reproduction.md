# FCCT / IRI independent reconstruction

Target: [arXiv:2511.05923v3](https://arxiv.org/abs/2511.05923v3), AAAI 2026.
The author [FCCT-IRI repository](https://github.com/QimingLi-scir/FCCT-IRI)
was empty at inspection on 2026-11-07. This implementation is a reconstruction.
No numerical reproduction is claimed until actual inference and comparison finish.

## Scope and completion requirements

The executable path currently covers LLaVA-1.5-7B FCCT tracing, deriving IRI layer
weights from those traces, paired original COCO POPE evaluation, last-token object
attention statistics, and 12 named Table 6 controls. It does not
reproduce the entire paper. In particular:

| Paper item | Local support | Still required for the original result |
|---|---|---|
| Figure 3, LLaVA FCCT | Three component outputs, seven token categories, 32 layers, curve plotting | Original subset, token assignments, noise specification and RR aggregation |
| Figure 3, Qwen-VL-Chat | No adapter | Original Qwen-VL-Chat checkpoint/backend; Qwen2.5-VL is a different model |
| Figure 4, attention | Attention mass, per-token and object-relative statistics plus plot/report | Original 3,000 VQAs and attention normalization protocol |
| Tables 1–2, IRI | Published equations and LLaVA hyperparameters; original POPE evaluation | Original author layer RRs, exact prompt/decoding and implementation |
| Table 1, MME | Reference targets recorded; no evaluator integration | Official MME assets, task pairing and evaluation protocol |
| Table 3, CHAIR | Reference targets recorded; no evaluator integration | Original caption subset/prompts and official CHAIR evaluator |
| Table 4, MMHal-Bench | Reference targets recorded; no judge run | Original GPT-4 judge snapshot, instructions, decoding and outputs |
| Table 4, MHumanEval | Reference targets recorded; no human scores fabricated | Original 146 examples and three real human annotators |
| Table 5, additional models | Not implemented | Qwen2-VL-7B and InternVL2-8B adapters and unreported IRI settings |
| Table 6, ablations | Executable 12-setting control matrix with printed parameters; hidden-state variant explicitly unsupported | Original hidden-state intervention and precise control semantics, plus completed runs |
| Table 7, latency | Not reproduced | Original hardware and measurement protocol |

## Installation and public assets

Use the repository's Python 3.11 environment. The versions tested locally are in
`requirements-fcct.txt`; the authors' package versions are unknown. On a new CUDA
12.4 environment, install PyTorch before the other requirements:

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-fcct.txt
```

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/setup_fcct.py --paper --pope --annotations --model
.\.venv\Scripts\python.exe fcct.py prepare --download-images
.\.venv\Scripts\python.exe fcct.py check
```

The model is `llava-hf/llava-1.5-7b-hf`, pinned to
`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`. These are HF-converted weights;
equivalence to the original author's runtime has not been verified. Model weights
require approximately 14 GB of disk space. The official COCO annotation archive
and selected original images require additional space. Downloads use atomic
temporary files; failed downloads do not count as complete assets. Rerunning setup
reuses completed files; interrupted individual HTTP downloads restart.

Original POPE files are pinned to AoiDragon/POPE commit
`08d957b917e5a378a2f99d35b6293c536a66298b` and checked against upstream Git
blob hashes. They contain 3,000 questions per split, 9,000 total. POPEv2 is not used.
The downloaded source lock records SHA-256 hashes. COCO images retain their
original licenses; see `data/fcct/dataset_card.json` after preparation.

For a smaller annotation transfer, use `--annotations-fast` in place of
`--annotations`. It reads the original validation-instances member using HTTP
byte ranges and verifies its ZIP CRC. The full archive also contains much larger
training annotations, which are unnecessary for this implementation.

Preparation uses annotated objects in 500 seeded COCO val2014 images and excludes
all POPE image IDs. Up to six distinct categories are queried per image. This is
a new tracing set: the paper's original IDs and question list are unavailable.
The question count may be below 3,000; the paper separately reports 3,000 VQAs
for its attention visualization, and does not give its tracing question count.
Every image is decoded, checked against COCO dimensions, and hashed. Images are
stored in `data/fcct/images/` and fetched from the official COCO bucket as needed.

The prepared local reconstruction contains **500 tracing images / 1,371 questions**
and **500 separate POPE images / 9,000 questions**, with zero image overlap.
The frozen manifest and annotation checksums are recorded in
`docs/fcct_dataset_card.json`. This is a newly selected tracing subset.

## Execution

The default configuration is `configs/fcct_llava15.json`. GPU execution uses FP16
without quantization or CPU offloading. The preflight requires at least 17 GiB
free GPU memory; this is a conservative estimate, not a guarantee against OOM.
Cached activations are stored on CPU. Vision projections are computed once per
clean/corrupted input and reused across decoder patching runs.

An eligible question requires up to 674 decoder forward passes: one clean, one
corrupted, and 32 × 3 × 7 patch runs. Empty categories are recorded with null RR.
One 3090 will take substantial time for a full run. Engineering measurements are
needed before estimating its total duration. The paper used eight A100 80 GB GPUs.

```powershell
# Complete the supported path using already prepared assets:
.\.venv\Scripts\python.exe fcct.py run

# Or run stages separately:
.\.venv\Scripts\python.exe fcct.py trace
.\.venv\Scripts\python.exe fcct.py profile
.\.venv\Scripts\python.exe fcct.py evaluate
.\.venv\Scripts\python.exe fcct.py report --tolerance 1.0
```

The architecture audit is available without loading the 7B checkpoint:

```powershell
.\.venv\Scripts\python.exe fcct.py audit
.\.venv\Scripts\python.exe fcct.py --require-exact run
```

The second command currently refuses execution and identifies missing original
artifacts and unimplemented experiments. This is a strict reproducibility check,
not a request for permission. Ordinary `run` remains an independent reconstruction.

Figure 4 and Table 6 controls can be executed separately:

```powershell
# The paper's separate 3,000-VQA manifest must be supplied first.
.\.venv\Scripts\python.exe fcct.py attention --manifest data/fcct/attention.jsonl

# Engineering-only analysis of the existing 1,371 tracing questions:
.\.venv\Scripts\python.exe fcct.py attention --manifest data/fcct/trace.jsonl --allow-partial

# Run after producing a complete version-0.2 trace/profile.
.\.venv\Scripts\python.exe fcct.py ablations --list
.\.venv\Scripts\python.exe fcct.py ablations --all
.\.venv\Scripts\python.exe fcct.py ablations --setting without_mlp last_ten

# Combine the supported stages once the separate attention manifest is available:
.\.venv\Scripts\python.exe fcct.py run --attention-manifest data/fcct/attention.jsonl --include-ablations
```

Attention reports are under `outputs/fcct/attention/attention_report.md`; ablation
results are under `outputs/fcct/ablations/ablation_report.md`. The matrix evaluates
the 3,000 adversarial COCO POPE questions and freezes the printed Table 6 parameters.
It does not tune coefficients to recover the published scores. Each case currently
generates its own matched regular/IRI answers. The hidden-state variant is not run
because its two coefficients and intervention placement are not defined by the text.

`--config` is a global option and goes before the command. Use a new output root
when changing scientific settings, code, packages, or datasets. A configuration,
code, manifest, environment, or RR-profile change rejects resuming existing results.
Whole samples are written atomically; errors never become model failures. A limit
counts additional questions on a resume, not total questions.

```powershell
# Small engineering check. It cannot establish agreement with the paper.
.\.venv\Scripts\python.exe fcct.py trace --limit 1
.\.venv\Scripts\python.exe fcct.py profile --allow-partial
.\.venv\Scripts\python.exe fcct.py evaluate --limit 6 --allow-partial
.\.venv\Scripts\python.exe fcct.py report
```

A partial profile is useful only if the traced question has a positive corruption
gap. Otherwise trace further questions; do not fill missing RRs with constants.
Use a separate output root for partial-profile evaluation before evaluating with
a subsequently completed profile. Report tolerance defaults to one percentage
point. It is an explicit comparison criterion, not a guarantee of agreement.
An otherwise completed report/run exits with code 3 if the measured metrics
fall outside that tolerance; execution success does not imply paper agreement.

## Method and reconstruction choices

The code restores complete MHSA outputs after output projection, complete MLP
outputs, or decoder-layer hidden outputs. It never substitutes individual attention
heads or patches teacher-forced answers. It scores one configured `Yes` token
(ID 3869 in the pinned tokenizer) with
the full-vocabulary next-token softmax and computes unbounded probability-based
RR. Equation 2 is evaluated for both positive and negative nonzero denominators.
Only zero denominators are undefined by default. An optional `positive_only` gap
policy and a minimum absolute gap can be configured as explicit extra filters.
The profile records sign counts and both image-weighted and question-weighted RR
averages; `rr_aggregation` selects which profile is used for IRI. No undefined value
is coerced to zero. Version-0.1 traces/profiles must be regenerated because their
positive-only filtering differs from the revised equation-based default.

The seven category assignments use COCO category bounding-box unions transformed
through CLIP's resize and center crop. Any patch overlap counts. Objects completely
removed by cropping are excluded. Crowd annotations are excluded. Early/late visual positions occur before the
first/after the last object patch; nonobject patches between these boundaries are
unassigned. Text spans are aligned with tokenizer offsets, including multiword
object names. The final token is exclusive to the last-token group.
Each trace records these unassigned positions. Explicit token groups can instead
be included in a manifest row with `token_ids_sha256`, the fingerprint of the exact
encoded prompt IDs; a mismatch is rejected. This does not infer the authors' masks.

IRI independently ranks the last-token attention/MLP layer RRs. Each component
uses its top three source layers and top ten target layers; an injection only uses
sources in strictly earlier layers of the same forward. Each batch vector's L2
norm is preserved. Original source outputs are cached before injection, and caches
are reset at each forward, including autoregressive cached decoding. Hyperparameters
are attention scale 0.26 and MLP scale 0.16 for LLaVA-1.5-7B.

Noise sigma 0.2, RGB-space corruption, clipping/rounding, seeded image selection,
bbox grounding, within-image then across-image averaging,
stable ranking ties, capitalized answer token, Vicuna system prompt, and binary
generation budget are **local choices**. They are enumerated in the configuration
and copied into provenance. They must not be described as recovered author settings.
No parameters are fitted to match published test scores.

POPE parsing follows the original upstream evaluator's first-sentence No/not/no
rule, including its unusual treatment of empty outputs as affirmative; empty
generation counts are reported. Metrics join predictions to references by stable
sample IDs. Partial evaluations never pass numerical-reproduction criteria.
This implementation evaluates the original COCO-only split. The paper does not
specify Table 2's dataset/run aggregation, and its rounded accuracy values do not
all correspond to single integer counts over 3,000 questions. Table comparisons
therefore remain provisional even after completing this local benchmark.

Published reference numbers are isolated in `configs/fcct_paper_targets.json`.
The paper's printed CHAIR formula names conflict with standard CHAIR metric names;
no local substitute CHAIR calculation is provided or silently relabeled.
Use the [original CHAIR implementation](https://github.com/LisaAnne/Hallucination/blob/master/utils/chair.py)
when adding that benchmark.

## Validation and artifacts

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The FCCT tests include real, randomly initialized tiny Transformers LLaVA decoder
integration: final hidden restoration recovers the clean probability, cached vision
projection matches an ordinary forward, tuple-valued attention patching works,
hooks clean up after exceptions, and IRI survives cached generation without stale
cross-forward values. Numerical checks cover normalization, unbounded/undefined
RR, grounding after center crop, deterministic noise, image-balanced profiles,
resume rejection, and upstream POPE scoring. These are implementation checks,
not experiments on the trained 7B checkpoint.

Version 0.2 was validated with 30 FCCT tests (29 in the full suite followed by
one added ablation-report regression test). The tests now include a complete
tracing loop and attention extraction on a real tiny Transformers decoder,
signed RR, externally supplied token assignments, equal-weight controls,
profile compatibility and strict exact-reproduction gating. A historical
version-0.1 check on the downloaded, trained 7B checkpoint passed token alignment, vision-cache
equivalence, final-hidden restoration, hook removal, and generation. Its record is
`outputs/fcct_engineering_smoke.json`: the configured affirmative token is 3869,
clean probability 0.9673872, corrupted probability 0.9488930, and probability after
final-hidden restoration 0.9673872. This one-image check does not estimate benchmark
accuracy or establish reproduction. The lowercase-token diagnostic is preserved
separately to document why the affirmative token choice was corrected.

Artifacts are under `data/fcct/` and `outputs/fcct/`. Completed runs save raw
probabilities, token groups, patch RRs,
exclusion counts, fitted curves, support per layer/group, paired generated text,
code/config/data hashes, package versions, and a measured-vs-published report.
Copies of the downloaded source and model checksum locks are kept in
`docs/fcct_sources.lock.json` and `docs/fcct_model.lock.json` for review.

The final local readiness report is `outputs/fcct_preflight.json`. All dataset
and model checks passed; the physical GPU had about 2.6 GiB free while Ollama
held its model. Full tracing/evaluation has not been run. The current state and
remaining scope are recorded in `outputs/fcct_reproduction_status.json`.

For exact reproduction, the missing author code, image/question list, corruption
parameters, original layer RRs, model revisions, prompts/decoding settings, evaluation
subsets and judge/annotation artifacts must become available. Obtaining these is
necessary before an exact-results claim can be justified. The authoritative
paper-to-code coverage matrix is `docs/fcct_architecture_audit.md`; regenerate it
with `python fcct.py audit` after implementation changes.
