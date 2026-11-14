**Research proposal: When Does Causal Tracing Predict Useful Steering?**

**Recommended direction:** Intervention-matched selection for vision–language hallucination mitigation, with explicit measurement of transfer from restoration scores to deployed intervention effects.

**Prepared:** 13 November 2026. **Status:** research design, not an implemented method or completed paper. **Scope of this deliverable:** documentation only. **Target:** evidence and conceptual clarity appropriate for a competitive top-tier submission; no acceptance or novelty guarantee is possible.

**The decision in one paragraph.** Build a paper around the gap between *where restoring an activation repairs a corrupted computation* and *where injecting an activation improves an ordinary prediction*. Establish whether this gap exists across models, explain how score instability and intervention geometry contribute, and select interventions using estimates of their actual deployment effects. Evaluate benefit under an explicit budget for newly introduced errors and at matched recall and descriptive coverage. Robust recovery-rate aggregation is an essential comparison inside this project, but is unlikely to carry a strong paper alone.

**Provisional title:** *From Restoration to Intervention: Measuring and Improving the Transfer of Causal Traces in Vision–Language Models*.

**One-sentence candidate contribution:** A reproducible framework for measuring whether causal restoration scores transfer to useful activation interventions, together with a selector that directly measures the deployed intervention and validates its benefit–harm trade-off on independent data.

The word “transfer” here refers to transfer between experimental operators and input conditions. It does not imply transferring one model’s layer numbers into another model.

**1. What the repository establishes—and what it does not**

The current implementation reconstructs FCCT and IRI using LLaVA-1.5-7B. The model remains pretrained; the experiment patches and injects activations during inference. The relevant files are [core.py](../src/fcct/core.py), [experiment.py](../src/fcct/experiment.py), [llava.py](../src/fcct/llava.py), and [report.py](../src/fcct/report.py).

The archived run contains 9,000 paired POPE answers over 500 images. There are 1,371 tracing records: 1,300 completed, 70 excluded because the object is outside the processed crop, and one excluded for an undefined or filtered gap. Trace and evaluation image IDs do not overlap. These facts are recorded in [the audit](EXPERIMENT_AND_RESEARCH_REVIEW.md) and [audit metrics](latest_audit_metrics.json). The earlier conversation independently recomputed the answer confusion matrices, gap counts, and the layer-13 influence example below.

| Split | Baseline accuracy | IRI accuracy | Baseline F1 | IRI F1 |
|---|---:|---:|---:|---:|
| Random | 89.03% | 89.33% | 88.96% | 88.94% |
| Popular | 85.60% | 87.07% | 85.99% | 86.90% |
| Adversarial | 79.20% | 81.27% | 80.95% | 82.08% |

Recall falls from 88.4% to 85.8% in each split. On adversarial POPE, false positives fall from 450 to 349 while false negatives rise from 174 to 213. The positive questions are shared across splits; this is not three independent demonstrations of recall loss.

The recovery denominator is negative for 369 of 1,300 completed tracing queries. Thirteen have an absolute denominator below 0.00001. Removing image 39357 changes attention layer 13’s image-mean RR from +6.291 to −2.268. The saved sensitivity analysis changes the attention top-three set from [14, 13, 17] to [19, 30, 24] after removing the largest 1% of absolute image contributions separately for each layer. That removal is a diagnostic, not a validated estimator. Layer indices are zero-based.

These observations motivate research; they do not prove that instability causes the recall loss. They also do not invalidate the original FCCT paper. The original author subset is unavailable in this reconstruction, and the archived source revision is not fully recoverable from hashes alone. Images and original manifests are absent from this workspace. Internal consistency of saved artifacts is weaker evidence than an independently repeated experiment.

**2. Literature-review method and limits**

This is an expanded, targeted review conducted through 13 November 2026, including sources inspected in the preceding project review. Searches covered causal tracing, activation patching, localization versus editing, visual steering, adaptive decoding, intervention selection, uncertainty, recall-preserving evaluation, and statistical risk control. Primary sources were preferred: arXiv, ACL Anthology, PMLR, official ICLR/NeurIPS proceedings, and CVF Open Access.

Representative search strings included “causal tracing localization does not predict model editing effectiveness,” “activation patching restoration amplification steering intervention mismatch,” “vision language hallucination reliable causal layer selection uncertainty recovery ratio,” “vision language intervention selection regret,” “activation steering causal tracing utility,” “visual and textual intervention hallucination,” and “false discovery controlled visual data splitting.” Searches also followed specific neighboring papers and their methodological topics.

This is not an exhaustive systematic review. No complete citation database export, formal screening flow, or citation-count census was produced. Search ranking may miss relevant work. Several sources were inspected at abstract or indexed-excerpt depth, not read completely. A failure to retrieve an exact match does not establish absence. Recent preprints are included because they can defeat a novelty claim even without confirmed proceedings publication.

Evidence labels below mean: **M** = relevant method passages inspected; **A** = primary abstract or proceedings description inspected; **E** = indexed primary-paper excerpt inspected. M does not mean every proof, appendix, experiment, or code path was audited. Each row links directly to its primary source. Venue labels are used only when the retrieved primary record supports them; author-reported acceptance is labeled separately.

**3. Literature map and novelty exclusions**

| Work and primary source | Evidence | Established overlap | Consequence for this proposal |
|---|---|---|---|
| [FCCT / IRI](https://arxiv.org/html/2511.05923v3), AAAI 2026 | M | Restoration-based tracing and recovery-weighted cross-layer injection | Baseline architecture; changing a coefficient is insufficient |
| [Does Localization Inform Editing?](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html), NeurIPS 2023 | A/E | Causal-tracing localization need not predict successful factual weight editing | The general localization/intervention mismatch is already known |
| [Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042), ICLR 2024 | M | Corruption and scoring choices can change localization conclusions | Metric sensitivity or replacing probabilities with logits is not new by itself |
| [AtP*](https://arxiv.org/abs/2403.00745) | E | Efficient approximation of activation patching | Approximate screening needs comparison against established methods |
| [When Attribution Patching Lies](https://arxiv.org/abs/2606.09899), 2026 preprint | A | Reliability and higher-order correction of gradient-based patching approximations | Distinguish approximation error from wrong experimental target |
| [VCD](https://arxiv.org/abs/2311.16922), CVPR 2024 | A/E | Contrast original and distorted visual inputs at decoding | Visual corruption contrasts are established |
| [OPERA](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_OPERA_Alleviating_Hallucination_in_Multi-Modal_Large_Language_Models_via_Over-Trust_CVPR_2024_paper.html), CVPR 2024 | A | Attention-related penalties and retrospection | Include a decoding-based comparison |
| [VTI: Reducing Hallucinations via Latent Space Steering](https://arxiv.org/abs/2410.15778) | A | Visual/textual latent intervention and feature stability | Generic “stable visual steering” is already occupied |
| [Modular Attribution and Intervention](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8001c3568152d134d821cd46d4d84768-Abstract-Conference.html), ICLR 2025 | A | Module/head attribution followed by targeted intervention | Must go beyond finding hallucination heads |
| [The Hidden Life of Tokens / VISTA](https://proceedings.mlr.press/v267/li25ca.html), ICML 2025 | A | Visual information steering | A new steering vector alone faces strong prior art |
| [Activation Steering Decoding](https://aclanthology.org/2025.acl-long.634/), ACL 2025 | A | Calibration-based activation directions and bidirectional steering | Calibration and signed steering are not new |
| [Dynamic Multimodal Activation Steering](https://arxiv.org/abs/2602.21704), 2026 | A/E | Context-dependent steering vectors and influential heads | Input adaptivity is insufficient novelty |
| [CausalLens](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.html), CVPR 2026 | A/E | Sensitivity-guided head modulation and projection alignment | “Reliable heads” and geometry-aware adjustment already have precedent |
| [Causal Route Gating](https://proceedings.mlr.press/v306/cheng26d.html), ICML 2026 | A | Decision-aligned visual/text route effects and selective text-route suppression | Closest challenge to a claim about matching interventions to decisions |
| [HalluTrace](https://aclanthology.org/2026.alvr-main.29/), ALVR workshop 2026 | A | Source attribution followed by source-targeted decoding | Failure-type diagnosis plus correction is established |
| [Vision-Language Introspection](https://aclanthology.org/2026.acl-long.1784/), ACL 2026 | A | Conflict detection, causal visual anchors and adaptive calibration | Do not claim the first causal calibration mechanism |
| [Context-Preference Activation Steering](https://aclanthology.org/2026.findings-emnlp.1156/), Findings 2026 | A/E | Steering context preference while monitoring generation quality | Visual neglect is not an uncontested explanation |
| [Mass-Tail Attribution Entropy](https://aclanthology.org/2026.findings-aacl.174/), Findings 2026 | A | Task-regime differences and recall-preserving conditional decoding | Preserving recall and comparing QA with captions are established concerns |
| [CORAL](https://arxiv.org/html/2609.38979v2), 2026; arXiv reports NeurIPS acceptance | M | Perturbation-based mirror statistics for false-discovery control | Statistical hallucination control is already occupied; inspect assumptions carefully |
| [Robustness of Internal Signals for Hallucination Detection](https://arxiv.org/abs/2610.27041), 2026 preprint | A | Robustness of internal detection signals | Robust internal signals alone are insufficient; focus on action selection |
| [Same Attention, Different Truths](https://arxiv.org/abs/2608.07302), 2026 preprint | A | Attention strength alone can fail to distinguish grounded and hallucinated objects | Attention mass cannot serve as proof of grounding |
| [POPE](https://aclanthology.org/2023.emnlp-main.20/), EMNLP 2023 | A/E | Object-presence evaluation | Necessary continuity benchmark, insufficient alone |
| [CHAIR](https://aclanthology.org/D18-1437/), EMNLP 2018 | A | Object hallucination in image captioning | Caption evaluation must respect object annotations and parsing |
| [AMBER](https://arxiv.org/abs/2311.07397) | A | Generative/discriminative and existence/attribute/relation evaluation | Broaden beyond yes/no object existence |
| [LeHaCE](https://papers.nips.cc/paper_files/paper/2024/hash/c9b551a2e195a209fc0b280de2f7f781-Abstract-Conference.html), NeurIPS 2024 | A/E | Length-sensitive evaluation of hallucination | Length-controlled caption evaluation is not a new contribution |
| [Can They Still See Removed Objects?](https://aclanthology.org/2025.naacl-long.349/), NAACL 2025 | A/E | Object-removal evaluation | Edited-image hallucination tests are established |
| [Conformal Risk Control](https://arxiv.org/abs/2208.02814), ICLR 2024 | A | Calibration of bounded monotone risks | Do not rename standard calibration as new theory |
| [Learn then Test](https://arxiv.org/abs/2110.01052) | A/E | Risk calibration through multiple testing | Relevant for selecting among frozen interventions |
| [Conformal Risk Control for Non-Monotonic Losses](https://arxiv.org/abs/2602.20151), 2026 | A | Stability-dependent control beyond monotone scalar losses | Non-monotonic intervention strength does not create an untouched theory problem |

Additional screened neighbors deserve full-method follow-up: [Prefill-Time Intervention](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Prefill-Time_Intervention_for_Mitigating_Hallucination_in_Large_Vision-Language_Models_CVPR_2026_paper.pdf) (E), [Mask What Matters](https://arxiv.org/abs/2602.11737) (A/E), [Causal Decoding for Hallucination-Resistant MLLMs](https://arxiv.org/abs/2602.21441) (A), and [Who Brings the Frisbee](https://openaccess.thecvf.com/content/WACV2025/papers/Huang_Who_Brings_the_Frisbee_Probing_Hidden_Hallucination_Factors_in_Large_WACV_2025_paper.pdf) (E). Their inclusion is a priority list, not a claim that their full methods were compared.

**4. The precise gap worth testing**

The reviewed sources establish many ways to detect hallucination, identify components, steer activations, adapt to context, or control output risks. This review did not establish that the following combined problem has already been resolved:

> Under a fixed deployment operator and compute budget, how well do restoration scores select useful vision–language interventions, how much selection failure comes from unstable measurement versus operator mismatch, and can direct intervention-effect estimates improve held-out benefit without excessive collateral errors?

This is a candidate gap. An abstract’s silence about this problem is not evidence that the full paper omits it. Before locking a novelty statement, read the full methods and supplements of FCCT, Hase et al., Causal Route Gating, CausalLens, modular attribution, VLI, CORAL and the internal-signal robustness paper. Explicitly search their references and citing papers for intervention utility, intervention selection, calibration, negative controls, and action-transfer studies.

The strongest differentiator would be an experimental decomposition showing that **fixing the estimator does not necessarily fix the intervention choice**, followed by a method that improves the choice under controlled data and compute. If that decomposition fails, revise the paper around the finding rather than forcing the intended story.

**Why this direction is preferable for this repository**

| Candidate | Fit to existing artifacts | Main weakness | Decision |
|---|---|---|---|
| Robust mean/median RR | Excellent | Likely an incremental estimator substitution | Mandatory baseline |
| Adaptive object/head steering | Moderate | Crowded literature; needs more hooks and design choices | Avoid as primary claim |
| Video or medical specialization | Low | New data, adapters and validation; weak connection to observed issue | Defer |
| A new risk-control formula | Moderate | Mature statistics and recent hallucination-specific work | Use established tools |
| Restoration-to-intervention transfer study plus matched selector | Strong | Must distinguish from localization/editing and decision-aligned gating | Recommended conditional on pilot |

**5. Formalize the two experiments before designing the algorithm**

Let x be an image/question input, y its verified label, f the frozen model, c a corruption, and e an intervention candidate. A candidate specifies component, source layer, target layer, strength, normalization, and application schedule. Include the identity candidate e=0.

For a restoration site l, the current score is:

\[
R_l(x,c)=\frac{p_{\mathrm{patch}(l)}(\mathrm{Yes}\mid c(x))-p(\mathrm{Yes}\mid c(x))}{p(\mathrm{Yes}\mid x)-p(\mathrm{Yes}\mid c(x))}.
\]

This measures a particular response to replacing activations in a corrupted computation. It is not automatically the utility of a source-to-target injection in the clean model. Positive-only tracing also provides no direct estimate of the cost of suppressing correct positive answers or amplifying incorrect ones.

For the deployed action define:

\[
G_e(x,y)=L(f(x),y)-L(f_e(x),y),\qquad U_e=\mathbb E[G_e(X,Y)].
\]

With a binary error loss, G is −1, 0 or +1. A positive U means fewer errors on the specified population. Keep separate loss definitions for QA and captioning; do not combine incomparable metrics in one arbitrary scalar.

Define collateral positive harm on an image as the fraction of its verified positive questions that were correct before intervention and wrong afterward. Its expectation H_e upper-bounds the increase in the image-balanced false-negative rate, because any corrections to previously wrong positives can only reduce the net increase. State the denominator explicitly. Images without positive questions do not enter this positive-question estimand.

Research target:

\[
\max_{e\in\mathcal E} U_e\quad\text{subject to}\quad H_e\le\epsilon,\ C_e\le B.
\]

Here epsilon is a predeclared harm tolerance and B a compute budget. These are expected population quantities under a specified distribution; they do not guarantee every image is unharmed or provide unrestricted domain-shift protection.

**6. A mechanism that can sharpen the paper**

IRI preserves the norm of a modified vector. For a nonzero target vector h and a fixed source vector a, consider a single injection:

\[
T_\alpha(h,a)=\|h\|\frac{h+\alpha a}{\|h+\alpha a\|}.
\]

At alpha=0 its derivative is:

\[
\left.\frac{dT_\alpha}{d\alpha}\right|_0
=a-h\frac{h^\top a}{\|h\|^2}.
\]

Thus the first-order effect removes the component of a parallel to h. For a differentiable target score s, the local score change is its gradient dotted with that remaining direction. A large recovery score says little by itself about this alignment at a different target layer on an uncorrupted input.

This is an elementary derivative, not a novel theorem. Its research value would be explaining measured selection failures and predicting when normalized injection helps or hurts. It applies locally, with the source treated as fixed and other interventions disabled. Multiple coupled injections, source activations already altered by upstream interventions, discontinuous decoding and finite strengths require direct experiments.

Test four explanations separately: small-denominator instability, source/target directional compatibility, finite-strength nonlinearity, and changes in the surrounding activation trajectory. Do not attribute all failures to one mechanism after observing a convenient example.

**7. Proposed method: a deliberately small first version**

The working name is **Intervention-Matched Selection (IMS)**. This is a provisional label, not a claim of priority or a new architecture.

1. **Construct a modest candidate set.** Include reachable attention/MLP source-target pairs, a small predeclared strength grid, and identity. Initially keep the original norm preservation and prefill-plus-decoding schedule. Hold the operator fixed while testing selection.
2. **Measure restoration and actual action effects separately.** Collect raw RR, robust RR, unnormalized restoration effects, and logit-margin restoration effects. On distinct development images measure finite-strength injection outcomes and, optionally, small-dose derivatives.
3. **Screen under an explicit budget.** Compare tracing-based, gradient-based, random and layer-depth screening. A tracing-only candidate pool could exclude precisely the actions the study needs to discover. Include candidates from outside the RR top set.
4. **Select using held-out utility.** Prefer the simplest selector that improves paired benefit under the harm budget. Use nested image splits or cross-fitting for development. The final shortlist must be frozen before confirmatory calibration.
5. **Calibrate and freeze.** On independent images assess all shortlisted policies with simultaneous uncertainty control. Return identity if no action clears the prespecified requirements.
6. **Evaluate the selected policy on untouched data.** Measure generalization to new images, corruption conditions and tasks. No labels or correctness indicators are available to the policy at inference.

The first version selects a global configuration per model/task. A learned per-input gate is a later extension only if heterogeneity is demonstrated. Adding a learned selector consumes labeled data and must be reported as such, even when backbone weights remain frozen.

**The decisive competing explanation is ordinary hyperparameter search.** Compare IMS against direct finite-strength grid search using the same images, number of forward/backward evaluations and tuning budget. If IMS does not improve selection quality, transfer or calibration efficiency over that baseline, do not claim a new selection algorithm. The mechanistic measurement study may still be worth pursuing on its own.

Robust RR candidates should include image-mean, median, trimmed mean, unnormalized effects and a pooled slope. For d=clean−corrupted and n=patched−corrupted:

\[
\widehat\beta_l=\frac{\sum_i w_i d_i n_{il}}{\sum_i w_i d_i^2+\lambda}.
\]

This is standard regularized regression, estimates a different quantity from mean RR, and is not the proposed novelty. Equalize total image weight. A positive slope can arise from negative gaps with negative numerators; report sign regimes separately. Setting a positive-only filter changes the estimand and can create selection effects.

**8. Statistical component: defensible guarantees without invented theory**

An initial guarantee can use a finite frozen candidate set and ordinary concentration bounds. Suppose M candidates are fixed without using a calibration sample of n independent, identically distributed image clusters. Let each image’s paired gain be in [−1,1] and its harm be in [0,1]. For a conservative simultaneous construction, use:

\[
r_G=\sqrt{2\log(4M/\delta)/n},\quad
r_H=\sqrt{\log(4M/\delta)/(2n)}.
\]

Select only candidates with empirical gain minus r_G positive and empirical harm plus r_H at most epsilon. A Hoeffding/union-bound argument gives simultaneous coverage of the relevant expected gains and harms with probability at least 1−delta under these assumptions. If gain and harm use different populations or sample sizes, use their respective sample sizes and make the distributions explicit. Identity has exactly zero paired change and can always serve as a fallback.

This is a standard, conservative argument, not a novel theoretical contribution. It does not require monotonicity in injection strength. It does require candidate independence from this calibration data, bounded image-level statistics and the sampling assumptions. Arbitrarily adapting the candidate set to these outcomes invalidates the simple statement.

The bound may be too wide to certify a small improvement. For M=20, delta=.05 and n=500, the gain radius is about .172, or 17.2 percentage points. A one-point improvement cannot be certified by that bound. Use this calculation to prevent inflated “guaranteed safe” claims. Better paired tests or variance-adaptive bounds require their own valid assumptions and multiple-selection treatment. [Learn then Test](https://arxiv.org/abs/2110.01052) is relevant established methodology.

Do not equate false-discovery rate, false-positive rate, net recall loss, and probability of any harmful intervention. [CORAL](https://arxiv.org/html/2609.38979v2) addresses a different output-level statistical construction; its inspected method passages also explicitly discuss assumptions behind the mirror statistics. This proposal’s risk definition concerns paired intervention harm. No unrestricted distribution-shift guarantee is proposed.

**9. Experimental design that could support a strong paper**

**Phase A: artifact and provenance recovery.** Restore images and manifests, verify checksums and crop visibility, preserve the historical archive, record the executed source snapshot, and independently reproduce the supported baseline. The historical POPE set is now exploratory because its outcomes informed this proposal.

**Phase B: falsification pilot.** Use approximately 100–150 development images, with both positive and negative questions, grouped by image. Compare restoration rankings with actual action utility for a modest pair set. Repeat stochastic corruption with at least three seeds and multiple strengths. Include blur or masking as a distinct family. Inspect the source-target geometry on the same cases. These counts are planning values, not a power calculation.

**Phase C: controlled selection benchmark.** On a manageable frozen candidate set, evaluate all actions on an audit subset to obtain an empirical utility table. This exhaustive table is for retrospective selector evaluation, never for choosing the deployed policy using test labels. Quantify rank correlation, top-k overlap, selection regret and cost. An empirical best action is noisy; estimate its uncertainty and avoid calling it the population oracle.

**Phase D: independent calibration and locked evaluation.** Choose sample size from pilot variance, a meaningful effect size, clustering and a prespecified analysis. Freeze prompts, generation length, thresholds, candidate set, selector and loss definitions before final evaluation. Use more calibration data when the risk constraint cannot be estimated precisely.

**Phase E: external replication.** Start with LLaVA-1.5-7B and a distinct family such as a validated Qwen-VL adapter. A stronger generality claim should include a third architecturally distinct family, subject to resources. Use official checkpoint/tokenizer revisions and validate hook semantics separately. Architecture names alone do not ensure compatible injection locations.

The three data roles—candidate development, confirmatory calibration and final evaluation—must remain separate. All questions, captions, crops, edited variants and repeated corruptions of one original image belong to the same split. Cross-benchmark source-image overlap needs checking; dataset names do not prove independence. These splits prevent experimental leakage but cannot establish absence of benchmark examples in pretraining.

**Benchmark roles**

| Evaluation | Role | Required companion measurement |
|---|---|---|
| COCO POPE | Continuity with reconstruction; object presence | Recall, FP/FN, affirmative rate, corrections and harms |
| Fresh object-presence holdout | Confirmatory binary evaluation | Thresholds fixed on development data |
| COCO captioning with CHAIR | Free-form object hallucination | Correct-object coverage, length and descriptive quality |
| AMBER | Broader generative/discriminative coverage | Preserve its official label orientation and metric definitions |
| Verified object-removal pairs | Dependence on changed visual evidence | Editing quality, updated labels and unchanged-object controls |
| Held-out visual domain | Transfer beyond COCO-like images | Explicit domain and annotation quality |
| General capability task set | Collateral capability costs | Task-appropriate metrics, not only hallucination scores |

Use official evaluators where possible. For CHAIR, report both instance-level and caption-level hallucination using the evaluator’s definitions; do not silently average per-caption ratios as a replacement. AMBER may use a different positive-class convention from object-presence recall, so avoid comparing identically named metrics without checking denominators. For judge-based tasks, freeze judge version and prompts, retain raw judgments, and manually audit a blinded sample.

Corruption experiments need two separate interpretations. Label-preserving degradation tests stability while the answer should remain the same. Object removal changes the appropriate label and tests sensitivity to changed evidence. Neither a noisy image nor an edited image is automatically a faithful real-world counterfactual. Annotate ambiguity and do not reuse the original “Yes” as ground truth after removing the object.

**10. Baselines and ablations that reviewers will need**

| Comparison | Question resolved |
|---|---|
| Unmodified model | Is intervention useful at all? |
| Frozen reconstructed IRI | Does the method improve this project’s reference? |
| Reachability-only IRI | Is an unusable source responsible for the gain? |
| Robust RR variants | Is simple aggregation sufficient? |
| Raw restoration effect and logit-margin restoration | Is division or score saturation the main issue? |
| Direct action grid search | Is the method merely tuning on labels? |
| Random and layer-depth selection | Does mechanistic information outperform cheap priors? |
| Gradient/tangent action estimate | Does finite-strength evaluation add value? |
| Uncertainty-free action selection | Does uncertainty handling matter? |
| Threshold/logit-bias baseline | Is binary improvement explained by saying “No” more often? |
| VCD plus at least one recent internal intervention | Is performance competitive beyond reconstructed IRI? |
| Causal Route Gating or CausalLens where reproducible | Does decision-aligned/head-based intervention already solve the practical problem? |
| Recall-aware/statistical neighbor where reproducible | Does the claimed advantage survive contemporary baselines? |

Use a staged design rather than running every possible factorial combination. Begin with selector-only changes at fixed injection scale. Then allow equally budgeted development tuning for all methods. Compare both equal inference cost and total cost including calibration. Record failures to reproduce a baseline instead of quoting its published number as if obtained under the local protocol.

At least one recent 2026 neighbor should be implemented faithfully before claiming contemporary competitiveness. Code availability, license and exact adapter support must be checked at execution time. No availability is assumed from an abstract.

**11. Primary endpoints and analysis**

Predeclare a primary behavioral endpoint and a primary mechanistic endpoint. A practical pair is held-out false-positive reduction under a stated recall-loss tolerance, and intervention-selection regret under a fixed candidate-evaluation budget. Treat the remaining metrics as secondary or exploratory.

For binary outputs, report the full confusion matrix, precision/recall/F1, affirmative rate, corrected and newly wrong answers. For continuous yes/no scores, specify tokenization and answer scoring. A first-token margin is not automatically a calibrated probability of the entire answer. The archive’s answer strings cannot recover missing logits.

For matched recall, choose thresholds on development data and report achieved test recall. Test-set threshold curves can describe operating trade-offs, but the resulting threshold is not an independently selected deployment policy. For captions, show hallucination against correct-object coverage and length; simply generating fewer nouns is an inadequate mitigation claim. LeHaCE is relevant prior work, rather than a newly invented evaluation idea.

Resample paired image clusters. Retain cross-split dependence when pooling POPE. Treat corruption runs as repeated measurements, not new independent images. Separate image-sampling uncertainty from selector randomness, corruption randomness, and model-family variation. Correct for multiplicity in confirmatory comparisons and report effect sizes with intervals.

Correlation between RR and action utility is only one diagnostic. Include residualized or stratified analyses for layer depth, source norm, target norm and action strength, because these can explain apparent correlations. Test whether geometry contributes information beyond those cheap covariates. Stable rankings can be consistently wrong; instability can coexist with useful selection.

**12. Contribution claims, phrased before results exist**

The proposed paper should aim for three connected contributions:

1. **Measurement:** an operator-specific assessment of how restoration scores predict deployed steering utility, decomposing estimator instability and intervention mismatch across models and settings.
2. **Method:** a selector that improves held-out intervention decisions under equal data/compute budgets and explicit collateral-harm constraints, with identity fallback when evidence is insufficient.
3. **Artifact:** a reproducible table of restoration scores, action effects, uncertainty, generation outcomes and compute costs, enabling future methods to evaluate selection rather than only final accuracy.

The first two are research hypotheses; the third becomes a contribution only when the artifact is actually produced and usable. The derivative, ridge estimator, bootstrap and finite-set risk bound are supporting tools. Do not list them as separate novel theoretical inventions.

Avoid the claims “first causal steering method,” “first uncertainty-aware hallucination control,” “first recall-preserving decoder,” “provably eliminates hallucination,” “no training data needed,” or “guaranteed A* paper.” If labeled data select the intervention, disclose that supervision even if backbone weights are frozen.

**13. Falsification and stopping rules**

| Outcome | Honest decision |
|---|---|
| Robust RR performs as well as action-matched selection | Narrow the method; study estimator reliability without claiming a superior operator-aware selector |
| Direct grid search wins at equal total cost | Remove the efficiency/selection superiority claim |
| Threshold adjustment explains the binary gain | Do not infer improved visual grounding |
| Caption gains disappear at matched coverage | Report a conservativeness trade-off |
| Transfer failure appears only on one old checkpoint | Limit generality; add a new-model replication before broad claims |
| Calibration uncertainty is too large | Collect more data or report empirical results without certification |
| Existing full-text work already studies this exact problem | Reframe the contribution or stop the novelty claim |
| Strong negative result replicates broadly | Consider a measurement paper with a well-supported explanation and artifact |

Suggested pilot targets, not guarantees: a meaningful improvement over both raw RR and direct search at matched resource budget, no more than a predeclared small recall loss, and directionally consistent benefit on a second model. Do not choose the numerical tolerance after seeing the test results. A universal one-point target ignores baseline difficulty and uncertainty; determine practical significance before final evaluation.

**14. Feasibility and execution milestones**

The full current trace grid has 32×3×7=672 patch cells per question. Some groups are empty. Restricting a pilot to last-token attention and MLP yields 64 patch cells. Repeating corruption conditions multiplies the decoder work. There are 496 strictly ordered source-target pairs in a 32-layer decoder per component, so exhaustive pair-by-strength search across the full dataset is not a sensible first step.

Use a small exhaustive audit subset and a fixed candidate budget. Report shortlist coverage: a poor shortlist cannot be rescued by a statistically careful selector. Validate source capture, whether upstream injections alter later source activations, KV-cache behavior, and prefill/decode schedules. A source is captured before injection at its own layer in the current code, but that does not make it globally unmodified by earlier layers’ interventions.

| Milestone | Deliverable | Proceed when |
|---|---|---|
| Foundation | Restored data, reproducible baseline, frozen provenance | Input and output checks agree |
| Pilot | Restoration/action transfer table on a small sample | Failure mechanism is measurable or a useful null result emerges |
| Baseline challenge | Equal-budget direct search and robust-RR comparisons | Proposed method has a credible advantage worth testing |
| Protocol lock | Frozen data roles, metrics, method and statistics | Confirmatory evaluation is insulated from development |
| Replication | Multiple models and task settings | Evidence supports the intended scope |
| Paper package | Main results, limitations, executable artifact | Claims match the completed evidence |

Estimate time and GPU requirements from measured pilot throughput. The current machine lacks the full dataset and a verified model execution environment; no reliable completion-time estimate follows from archived hardware records. Quantizing to fit memory changes the experimental model and needs its own comparison.

**15. What would make this competitive at a top-tier venue**

A plausible strong paper would change how researchers interpret and use causal traces: demonstrate a repeatable operator-transfer failure, explain it beyond a single pathological ratio, and show that addressing it improves useful intervention selection. Multi-model evidence, fair modern baselines and a reusable selection benchmark matter more than a collection of unrelated modules.

Likely reviewer objections should be answered experimentally:

| Objection | Required answer |
|---|---|
| “Hase et al. already showed localization need not imply editing.” | New VLM inference operators, measured decomposition, predictive explanation and useful selector; explicitly acknowledge the prior result |
| “Causal Route Gating already aligns interventions with decisions.” | Direct comparison and a distinct selection/transfer question; abandon overlap claims not supported by full-text review |
| “This is validation-set hyperparameter tuning.” | Equal-budget direct search, sample-efficiency curves and mechanistic predictive value |
| “The method is just more conservative.” | Recall, coverage, matched operating points, and paired harm reporting |
| “The guarantee is standard or vacuous.” | Attribute the statistics and report bound widths; make no novel-theory claim without additional substance |
| “The benchmark is circular.” | Frozen candidates, independent development/calibration/test images, blinded annotation and no test-label action selection |
| “The finding concerns only a reproduction artifact.” | Fresh runs, original-baseline fidelity where verifiable, multiple checkpoints and sensitivity to protocol choices |

A suitable paper structure would introduce the transfer question, distinguish restoration from deployment mathematically, establish the empirical failure, explain the geometry and estimation factors, present the selector, and evaluate generalization and costs. Main figures should include a restoration-versus-action utility plot, a controlled estimator/operator ablation, a benefit–harm frontier, and selection performance versus total calibration compute. Avoid an architecture diagram full of optional modules before the core hypothesis is established.

**16. Immediate next research action**

The next action should be a small falsification experiment measuring actual source-target injection outcomes alongside restoration scores, with direct search and robust RR as equal-budget controls. Do this before building a dynamic gate, expanding to video, or writing a result-bearing abstract. If the pilot supports the transfer hypothesis and the literature audit survives the closest full-text comparisons, this repository has a credible path to a substantive research contribution.

**Final evidence boundary:** this document contains 29 core literature entries and four additional screened neighbors. It provides a strong candidate research direction, not an exhaustive proof of originality. No new model experiment, benchmark improvement, implementation, theoretical novelty or acceptance outcome is claimed. The program code was not changed for this deliverable.
