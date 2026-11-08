# FCCT/IRI paper-to-code architecture audit

**Exact numerical reproduction has not been achieved.** This audit checks implementation coverage; it does not substitute tests for paper experiments.

Source: the supplied AAAI-26 conference text and https://arxiv.org/abs/2511.05923v3.

| Paper item | Coverage | Finding | Code |
|---|---|---|---|
| Equation 1: residual stream | implemented | Uses the pretrained LLaVA decoder; component hooks observe projected MHSA, MLP, and decoder-layer outputs. | `src/fcct/core.py` |
| Equation 2: recovery rate | implemented | Full-vocabulary affirmative next-token probability; signed nonzero denominators by default; no clipping. Zero denominators are undefined. | `src/fcct/core.py` |
| Equations 3-5: IRI | implemented | Independent component rankings, strict source < target ordering, last-token source outputs, RR weighting and per-vector L2 norm preservation. | `src/fcct/core.py` |
| LLaVA main hyperparameters | implemented | k1=3, k2=10, attention scale=0.26, MLP scale=0.16. | `configs/fcct_llava15.json` |
| Figure 3: component/layer/category tracing | implemented_with_unverified_inputs | Three components, 32 layers, seven categories; original sample list, grounding and aggregation are not available. | `src/fcct/experiment.py` |
| Seven category assignments | independent_interpretation | COCO boxes mapped through the CLIP crop; unassigned visual/text positions are recorded. Exact assignments can be supplied with an encoded-token checksum. | `src/fcct/llava.py` |
| Figure 4: last-token object attention | implemented_with_unverified_inputs | Saves attention mass, per-token and object-relative values. Requires a separate 3000-question manifest for the reported sample count; author normalization is unspecified. | `src/fcct/attention.py` |
| Table 2: LLaVA regular and IRI POPE | implemented_with_unverified_protocol | Original 9000-question COCO POPE. The paper's dataset/run aggregation, prompts and decoding snapshot remain unverified. | `src/fcct/report.py` |
| Table 6: named controls | partially_implemented | 12 supported settings with the printed coefficients; first/last-ten and equal-weight controls state their interpretation. Hidden-state IRI is not invented. | `configs/fcct_ablation_plan.json` |
| Qwen-VL-Chat, LLaVA-NeXT, Qwen2-VL-7B, InternVL2-8B | not_implemented | Dedicated checkpoint-specific adapters, tracing and evaluation are still required. Qwen2.5-VL is not a replacement. | - |
| Table 1: MME | not_implemented | Official assets, paired-question evaluator and all model/method results are required. | - |
| Table 3: CHAIR | not_implemented | Original image subset and caption prompts plus original CHAIR evaluator are required; max_new_tokens=512 is reported. | - |
| Table 4: MMHal-Bench | blocked_external_and_not_implemented | 96 questions plus the original GPT-4 judge version, instructions and outputs are required. | - |
| Table 4: MHumanEval | blocked_external_and_not_implemented | 146 examples and three actual human annotators; automated scores cannot replace this experiment. | - |
| VCD, OPERA, PAI and VTI baselines | not_implemented | Original baseline implementations and matched decoding/evaluation are required. | - |
| Table 7: TTFT/TPOT | not_implemented | Matched measurement protocol and original 8 x A100 80GB hardware experiment are unavailable. | - |

## Original artifacts still required

- Original 500-image tracing subset, question list, object regions and exact token-category assignments
- Gaussian noise domain, standard deviation, clipping, RNG and corruption-selection protocol
- Original checkpoint/backend revisions, prompt templates, affirmative token ID and decoding settings
- RR aggregation, numerical filtering and original per-layer RR profiles
- Original separate 3000-VQA attention manifest and attention normalization
- IRI source-cache and prefill/decoding intervention details; hidden-state ablation semantics
- Full benchmark subsets, baseline configurations, judge snapshot and human annotations

## Verified corrections in version 0.2

- Equation 2 now retains negative denominators. Only a zero denominator is undefined by default; positive-only filtering is explicit and optional.
- Profile files expose image-weighted and question-weighted RR averages and their supports, because the paper does not state its aggregation.
- Token coverage records unassigned positions; externally supplied category assignments require an exact encoded-token checksum.
- Figure 4 analysis saves multiple explicit attention normalizations instead of claiming an undocumented one is the authors' choice.
- Table 6 controls freeze the printed coefficients; the ambiguous hidden-state variant is explicitly unsupported.
- Profiles from old tracing semantics or incompatible noise/token/aggregation settings are rejected.

The original author repository is empty as checked on 2026-11-08. Numerical targets must never be copied into measured outputs.
