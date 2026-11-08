# FCCT / IRI

Independent reconstruction of **Causal Tracing of Object Representations in Large
Vision Language Models: Mechanistic Interpretability and Hallucination Mitigation**,
[arXiv:2511.05923v3](https://arxiv.org/abs/2511.05923v3), AAAI 2026.

The implemented workflow covers LLaVA-1.5-7B causal tracing, Intermediate
Representation Injection, paired original COCO POPE evaluation, last-token object
attention analysis, and 12 named Table 6 controls. Numerical
agreement with the paper has not been established. Other paper models and benchmarks
remain outside the implemented scope; see [the reproduction guide](docs/fcct_reproduction.md).

## Run the prepared experiment

Run PowerShell commands from this repository:

```powershell
.\.venv\Scripts\python.exe fcct.py check
.\.venv\Scripts\python.exe fcct.py audit
.\.venv\Scripts\python.exe -u fcct.py run
```

The readiness check requires at least 17 GiB of free physical GPU memory. If Ollama
is holding GPU memory, close its clients and unload the model you are using before
starting the experiment.

`run` performs tracing, fits layer recovery rates, generates regular and IRI POPE
answers, and writes a measured-versus-published report. Run the same command after
an interruption to resume completed samples. Keep the code, configuration, packages,
and datasets unchanged while resuming.

The prepared dataset contains 500 tracing images with 1,371 object questions and
500 separate POPE images with 9,000 questions. The image sets have zero overlap.
The pinned model and all images required by the prepared FCCT/POPE workflow are
already downloaded in this workspace.

Version 0.2 uses the signed, nonzero denominator in Equation 2. Traces and profiles
from version 0.1 must be regenerated. Passing `--require-exact` refuses execution
while the original artifacts and remaining paper experiments are unavailable.

For the Table 6 controls, run `python fcct.py ablations --all` after completing the
main trace/profile stages. For Figure 4, provide the separate 3,000-VQA manifest:
`python fcct.py attention --manifest data/fcct/attention.jsonl`. The original
attention manifest is not included in the independently selected tracing set.
See [the architecture audit](docs/fcct_architecture_audit.md) for paper coverage.

## Project files

```text
fcct.py                      Experiment command line
FCCT_IRI_paper.md             Paper reference
configs/fcct_llava15.json     Frozen experiment settings
configs/fcct_paper_targets.json Published comparison values
configs/fcct_ablation_plan.json Table 6 parameters and unsupported-variant reasons
src/fcct/                    Tracing, IRI, data, provenance, and reporting
scripts/setup_fcct.py         Public asset downloads
scripts/smoke_fcct.py         Trained-checkpoint engineering check
tests/test_fcct.py            FCCT tests
docs/fcct_reproduction.md     Protocol, limitations, and detailed commands
docs/fcct_*lock.json          Downloaded asset checksums
data/fcct/                   Images, original annotations, and frozen manifests
outputs/fcct/                Scientific run artifacts, when generated
outputs/fcct_*.json           Engineering checks and readiness records
.venv/                       Python environment and pinned LLaVA cache
```

## Installation and assets

For a new Python 3.11 environment, install the appropriate PyTorch CUDA build first,
then install `requirements.txt`, which includes the pinned FCCT requirements.
The tested Windows build is PyTorch 2.6.0 with CUDA 12.4:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/setup_fcct.py --paper --pope --annotations-fast --model
.\.venv\Scripts\python.exe fcct.py prepare --download-images
```

## Validation and results

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite tests numerical recovery, IRI normalization and cache handling, real tiny
Transformers decoder integration, dataset isolation, download integrity, and reporting.
A separate trained 7B checkpoint check is saved in `outputs/fcct_engineering_smoke.json`.
Engineering checks do not establish reproduction of the paper's benchmark results.

Completed scientific runs produce `outputs/fcct/report/report.md`,
`outputs/fcct/report/comparison.json`, and `outputs/fcct/report/fcct_curves.png`.
The original author subset, corruption details, and evaluation artifacts are still
needed to justify an exact reproduction claim.
