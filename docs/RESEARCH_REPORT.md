# FCCT/IRI: reproduction verdict, issues found, related work, and the R-FCCT extension

Prepared 20 November 2026. Paper: Li et al., *Causal Tracing of Object Representations in Large Vision
Language Models: Mechanistic Interpretability and Hallucination Mitigation*, AAAI-26 (pp. 31645–31653),
[arXiv:2511.05923v3](https://arxiv.org/abs/2511.05923v3). The FCCT definitions (Eq. 2, the seven token
categories, the three runs) and IRI (Eqs. 3–5) are treated as fixed. Nothing below changes them; the
extension adds measurement and selection around them.

## 1. What is novel in the paper

1. **FCCT** — the first causal-tracing study of LVLMs that corrupts the *image* with Gaussian noise
   and restores activations at a grid of **7 token categories × 3 components (MHSA, MLP, hidden
   state) × all 32 layers**, scored by the Recovery Rate (Eq. 2).
2. **Findings** — (i) the last token's MHSA in middle layers (~layer 15) aggregates cross-modal
   object information; (ii) MLPs show a three-stage progression (visual object tokens → textual
   object tokens → last token); (iii) hidden states shift from visual-centric to task-centric.
3. **IRI** — training-free injection of the last token's MHSA/MLP outputs from the top-k₁ RR
   *source* layers into the top-k₂ RR *target* layers (l > k), weighted by RR, norm-preserved.
   The paper reports state-of-the-art results on POPE, MME, CHAIR, MMHal-Bench and MHumanEval,
   with no latency cost.

The causal-tracing method is the core novelty. IRI is how it is put to use: tracing scores choose
where to intervene.

## 2. Did the reproduction succeed end to end?

**It ran end to end for the supported scope. It did not reproduce the paper.**

| Paper element | Status in this repository |
|---|---|
| FCCT tracing, LLaVA-1.5-7B (Fig. 3a–c) | Ran: 1,371 questions, 873,600 patch records |
| IRI on POPE, LLaVA-1.5-7B (Table 2) | Ran: all 9,000 paired answers |
| Qwen-VL-Chat tracing, other 4 models | Not run (no adapters) |
| MME, CHAIR, MMHal-Bench, MHumanEval | Not run |
| Table 6 ablations | Not run (only 5 questions of one setting) |
| Figure 4 attention (3,000 VQAs) | Not run (manifest not public) |
| Table 7 latency | Not run |

Integrity checks pass: confusion matrices recompute exactly from answer text, hashes chain
trace → profile → predictions, and there is zero tracing/test image overlap
(`analysis/latest_audit_metrics.json`).

**Measured vs. published (POPE, LLaVA-1.5-7B):**

| Split | Regular acc (ours / paper) | IRI acc (ours / paper) | IRI gain (ours / paper) |
|---|---|---|---|
| Random | 89.03 / 83.29 | 89.33 / 89.76 | **+0.30 / +6.47** |
| Popular | 85.60 / 81.88 | 87.07 / 87.67 | **+1.47 / +5.79** |
| Adversarial | 79.20 / 78.96 | 81.27 / 85.17 | **+2.07 / +6.21** |

The final IRI numbers on random and popular land within 1 pp of the paper, but that is a
coincidence of two offsets. Our baseline is 4–6 pp *higher* and our gain is 3–20× *smaller*. The
image-cluster bootstrap 95% CI for the random-split gain is [−0.30, +0.93] pp, which includes zero.
IRI lowers recall from 88.4% to 85.8% on every split. It removes false positives, but it also creates
39 new false negatives.

## 3. Issues found (with evidence)

**I1. The paper's "Regular" baseline is the VCD sampling baseline.** FCCT Table 2's LLaVA-1.5 Regular
row (83.29/81.33, 81.88/80.06, 78.96/77.57) and its VCD row (87.73/87.16, 85.38/85.06, 80.88/81.33)
are identical, to two decimals, to Table 1 of [VCD (CVPR 2024)](https://arxiv.org/abs/2311.16922).
VCD's caption says "Regular decoding denotes direct sampling" (temperature 1, averaged over runs).
Its recall is only 72.8%, against 88.4% for greedy decoding. FCCT does not state IRI's decoding
protocol. If IRI used greedy decoding, most of the reported +6 pp is the gap between sampling and
greedy decoding, not IRI. This explains our baseline gap exactly. It is a protocol confound in the
paper, not a fault in the reconstruction.

The same holds for MME (Table 1). FCCT's Regular row (175.7 / 124.7 / 114.0 / 151.0) is VCD's
*sampling* Regular (175.67 ± 7.51, 124.67 ± 19.59, 114.00 ± 9.32, 151.00 ± 10.45) rounded. FCCT
prints the total as 565.4 where VCD has 565.33. FCCT's MME VCD row (592.0 total) does *not* match
VCD's own main table (604.66). It is close to VCD's appendix ablation value at T = 500 (591.67).
FCCT's CHAIR Regular row (52.8 / 15.9 / 77.3 / 93.4) appears in neither PAI (greedy 46.2 / 13.8 for
7B) nor OPERA, so its protocol cannot be recovered. MMHal-Bench's official judge is `gpt-4-0314`
at temperature 0, scored by the `rating: k` parse. The judge itself is non-deterministic, as its
README notes. All of these facts were verified on 20 November 2026 against the VCD paper (arXiv
v1), the VCD, OPERA and PAI repositories, and the MMHal-Bench repository.

**I2. Hyperparameters were tuned on the test benchmark.** The Table 6 caption says "λa and λm
individually optimized" on MS-COCO POPE. No held-out split is described.

**I3. The corruption barely corrupts.** With σ = 0.2 (pixel units), the median |p_clean − p_corrupted|
is 0.018. Only 332/1,300 traced questions drop by more than 0.05, and noise *raises* p(Yes) for 369
(28%). The paper says noise should "induce measurable drops", but σ is unreported.

**I4. The RR estimator is ill-conditioned.** The ratio has near-zero and negative denominators.
One question gives RR = +21,385 (gap −3.1e-6). Removing one image flips attention-layer-13 mean RR
from +6.29 to −2.27. Five of the six selected source layers appear in the bootstrap top-3 only 47–64% of the time; attention layer 14 is the exception at 99%.
Mean RR for attention layer 14 is 27.8, so 0.26 × RR = 7.2 effective scale. The paper's λ values
presume RR ≈ [0, 1] ("close to 1 indicates a strong causal effect"). The reconstruction's v0.2
policy (keep any non-zero signed gap) departs from the paper's "measurable drops" wording.

**I5. Some selected layers are wasted.** IRI ranks sources and targets independently. MLP source 31
has no later target, and attention targets 3, 11 and 13 have no earlier source. In practice IRI
uses 2 of 3 MLP sources and 7 of 10 attention targets.

**I6. Tracing never looks at hallucinations.** FCCT traces only *present* objects with p(Yes).
Object hallucination is a wrong "Yes" for an *absent* object, which the tracing never observes.
Layers chosen to restore "Yes" are injected into every question. This matches the observed trade:
fewer false positives, more false negatives.

**I7. Tracing and evaluation prompts differ.** Tracing used "…Please answer this question with one
word (Yes or No)." The archived POPE run used the bare question, so the answers are sentences
("Yes, there is…"). The paper does not state its POPE prompt. VCD's code appends " Please answer
this question with one word."

## 4. Recent peer-reviewed work directly relevant to FCCT's core novelty

Verified in this session (venue confirmed on the publisher/proceedings page):

| Work | Venue | Relation to FCCT | Consequence |
|---|---|---|---|
| [FCCT/IRI](https://ojs.aaai.org/index.php/AAAI/article/view/40431) | AAAI 2026 | The base paper | — |
| [HIRE: Hallucination-aware Intermediate Representation Edit](https://proceedings.iclr.cc/paper_files/paper/2026/hash/41fd34e25626cde2dd249e5b11fb547f-Abstract-Conference.html) | ICLR 2026 | Edits intermediate representations against hallucination | Strongest intermediate-representation competitor; code is public ([ASGO-MM/HIRE](https://github.com/ASGO-MM/HIRE)) |
| [Causal Route Gating](https://proceedings.mlr.press/v306/cheng26d.html) | ICML 2026 | Gradient-estimated visual/text route effects; suppresses conflicting text routes | Closest "decision-aligned causal intervention"; do not claim to be first there |
| [CausalLens](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.html) | CVPR 2026 | Sensitivity-guided head amplification in mid layers (L10–20) | Independently supports FCCT's mid-layer finding; "reliable heads" is prior art |
| [VCD](https://arxiv.org/abs/2311.16922) | CVPR 2024 | Gaussian-noised image contrast; source of FCCT's baseline numbers | Required baseline; defines the sampling protocol |

From the earlier project review (`analysis/NOVELTY_RESEARCH_PROPOSAL.md`), not re-verified this
session: Does Localization Inform Editing? (NeurIPS 2023), Best Practices of Activation Patching
(ICLR 2024), VISTA (ICML 2025), Activation Steering Decoding (ACL 2025), Vision-Language
Introspection (ACL 2026), Mass-Tail Attribution Entropy (Findings AACL 2026).

Closest unrefereed preprint: [Dual-Pathway Circuits of Object Hallucination](https://arxiv.org/abs/2605.13156).
It patches visual tokens across five VLMs to find a grounding pathway and a hallucination pathway.
R-FCCT must be explicitly distinguished from it. The distinction is that R-FCCT keeps FCCT's
restoration operator and uses absent-object traces to *select IRI layers*. It does not characterize
circuits.

Searches were targeted, not a systematic review. Before submission, read the full text of HIRE,
Causal Route Gating and the dual-pathway preprint, and check papers citing FCCT.

## 5. The extension: R-FCCT (code in `src/rfcct/`, run with `rfcct.py`)

Each component targets one issue. FCCT's three runs, restoration operator and Eq. 2 are kept, and
IRI Eqs. 3–5 are executed unchanged by `PlannedIRI`. A test checks that `PlannedIRI` is
bit-identical to the original `IRI` under the paper's plan.

| Component | Fixes | What it does |
|---|---|---|
| Corruption calibration | I3 | Smallest σ for which ≥80% of clean-correct questions drop and the median drop is ≥1 logit. The criterion is fixed in the config before tracing. |
| **Contrastive decision-margin tracing** | I6 | Traces present *and* POPE-style absent objects (random/popular/adversarial) with s = y·(logit Yes − logit No). Keeps a trace only if the clean answer is correct and corruption lowers s by ≥ 0.5. p(Yes) is still recorded, so Eq. 2 is computable on the same runs. |
| Ratio-of-means causal score | I4 | β_l = Σ n / Σ d per class, image-balanced. Every d > 0, so the denominator cannot cancel. Classes are averaged. |
| Stability selection + LCB weights | I4 | 1,000-draw image-cluster bootstrap. Sources need a positive 5% lower bound and are ranked by top-k inclusion frequency. The weight is the lower bound, not the noisy mean. |
| Reachable plans | I5 | Every source gets a later target, and targets lie after the earliest source. |
| Held-out strength tuning | I2 | λ grid scored on 200 *new* COCO images (1,200 questions). Maximizes accuracy subject to recall loss ≤ 1 pp. POPE is never used for choices. |
| Protocol-matched evaluation | I1, I7 | Greedy decoding *and* the VCD sampling protocol (3 seeds) with the VCD prompt suffix. Sampled Regular should land near the paper's 83.29 if I1 is right. |
| Prior-shift controls | I6 | Regular decoding with a yes/no-margin threshold chosen on calibration data, under the same recall floor or matched yes-rate. If IRI cannot beat this, its gain is a bias shift, not better grounding. |
| Paired statistics | — | Exact McNemar and image-cluster bootstrap CIs, with Holm correction within the predeclared primary family. |

**Same evaluation matrix as the paper.** `bench.py` evaluates Regular, VCD, IRI and R-FCCT
(`riri`) on every LLaVA-1.5-7B benchmark the paper uses: MME (Table 1), POPE (Table 2), CHAIR
(Table 3), MMHal-Bench (Table 4), the ablations (Table 6) and latency (Table 7). It uses the
official metric code under both decoding protocols. R-FCCT's strength is tuned only on POPE-style
calibration data. Its CHAIR, MME and MMHal results therefore test transfer from the discriminative
probe to generation, which a reviewer will ask about.

Ablations are built into the pipeline: `paper_iri` (reconstruction, paper λ or tuned λ),
`faithful_rr` (paper Eq. 2 with the measurable-drop filter), `riri_present_only` (no absent
objects, which isolates I6), `riri_no_stability` (no bootstrap) and `riri` (full method).

### Hypotheses and how they can fail (declare before running)

* **H1 (protocol):** sampled Regular ≈ paper Regular (within ~1 pp), and greedy Regular is ~4–6 pp
  higher. *Fails if* sampled Regular is far from 83.29: then I1 is not the explanation.
* **H2 (estimator):** R-FCCT source layers have bootstrap inclusion ≥ 0.9, against 0.47–0.64 for
  five of six reconstruction sources. *Fails if* inclusion stays low.
* **H3 (contrastive):** `riri` beats `riri_present_only` and `paper_iri@tuned` in accuracy, with
  smaller recall loss. *Fails if* there is no significant difference after Holm correction.
* **H4 (grounding):** `riri` beats `threshold_control`. *If it fails,* report honestly that IRI
  behaves like a calibrated prior shift on POPE. That is still a publishable negative finding.

## 6. Publication assessment

The strongest framing is *"Does causal tracing choose good interventions? Reliability of FCCT-guided
injection in LVLMs."* Contribution (a) is the documented protocol confound (I1) and test-set tuning
(I2). Contribution (b) is the estimator and selection failures (I3–I6). Contribution (c) is
R-FCCT with held-out, protocol-matched evaluation and prior-shift controls.

The present evidence and code are enough for (a) and (b) on LLaVA-1.5 POPE. To make a credible case
at ACL/EMNLP main, you need:

1. Run R-FCCT end to end. The results decide the story, so nothing is claimed yet.
2. A second model family (Qwen-VL-Chat or LLaVA-NeXT, both in the paper). This needs a new
   adapter.
3. A generative benchmark (CHAIR with recall/length reported) to show the gains are not POPE-only.
4. Contact the FCCT authors about the decoding protocol before asserting I1 in print. The numerical
   identity with VCD is a fact; how IRI itself was decoded is not stated.

No acceptance outcome can be guaranteed. A careful replication with a confirmed confound and a
statistically controlled fix is the kind of paper reviewers value, provided the claims stay
exactly as strong as the evidence.

## 7. Limitations of this extension

* Single model (LLaVA-1.5-7B) and a single benchmark (COCO POPE). CHAIR, MME and MMHal are not
  integrated.
* Last-token MHSA/MLP tracing only: exactly what IRI uses, but not the full 7×3 FCCT grid (use
  `fcct.py trace` for that).
* Absent-object traces depend on noise *raising* hallucination, which is VCD's premise. Traces
  where noise does not lower correctness are excluded and counted, not hidden.
* Gaussian noise is a synthetic corruption, not a natural counterfactual.
* Calibration negatives use COCO label statistics computed without POPE test images. POPE's own
  negatives came from its authors' pipeline, so the distributions are close but not identical.
* All numerical claims about R-FCCT await the GPU run. The tests verify correctness on a tiny model,
  not effect sizes.
