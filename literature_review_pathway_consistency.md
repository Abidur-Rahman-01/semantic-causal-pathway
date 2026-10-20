# Literature for Semantic Causal Pathway Consistency in MLLMs

**Purpose.** This reading list maps the project’s method and goal to peer-reviewed work in strong AI venues and IEEE Transactions journals. It prioritizes ICLR, AAAI, and IEEE TPAMI. Venue labels are stated directly; “A*” is not a universal label, because conference rankings depend on the ranking system and year. I therefore do not assign a ranking unless a specific ranking source has been checked.

## Short answer: which paper is the closest foundation?

There is no single paper that is an exact base for this project. The closest methodological neighbor is **NOTICE**: it combines semantic image corruption with mechanistic analysis and activation patching in vision-language models. The closest top-tier causal-mechanism references are the two **ICLR 2025** papers below, which investigate attention causality and hallucination. The project’s proposed distinction is to measure whether the approximate causal mediator set remains stable across meaning-preserving image variants, and whether that instability predicts failure on a separately held-out transformation family.

Do not describe the project as “the first causal interpretability method for MLLMs.” These papers already establish closely related causal analysis. A narrower potential contribution is the **cross-variant semantic-evidence-to-mediator stability measurement**, subject to a complete literature review and confirmatory evidence.

## Highest-priority papers

| Priority | Paper and venue | Why it matters here | Relationship to this project |
|---|---|---|---|
| 1 | [Mitigating Modality Prior-Induced Hallucinations in Multimodal Large Language Models via Deciphering Attention Causality](https://proceedings.iclr.cc/paper_files/paper/2025/hash/86fe62e3b315d2578721562d9fd1a433-Abstract-Conference.html) — **ICLR 2025** | Applies causal inference, counterfactual reasoning, and adjustment to visual/language attention, targeting modality-prior hallucinations. | Closest match to the project’s causal-attention motivation. It is a mitigation method; it does not test cross-variant mediator-set stability as proposed here. |
| 2 | [Understanding and Mitigating Hallucination in Large Vision-Language Models via Modular Attribution and Intervention](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8001c3568152d134d821cd46d4d84768-Abstract-Conference.html) — **ICLR 2025** | Uses counterfactual module edits and causal mediation to identify hallucination-related attention heads, then intervenes on them. | Important precedent for head-level causal analysis and the eventual model-improvement track. It does not establish the project’s CPS/CECA/SCC measures. |
| 3 | [Causal Tracing of Object Representations in Large Vision Language Models: Mechanistic Interpretability and Hallucination Mitigation](https://ojs.aaai.org/index.php/AAAI/article/view/40431) — **AAAI 2026** | Introduces fine-grained cross-modal causal tracing over visual/text tokens, attention, FFNs, and hidden states; studies representation interventions for hallucination mitigation. | Very close to the mechanism-recovery and intervention components. It emphasizes tracing object representations and mitigation, rather than semantic-equivalence stability and held-out failure prediction. |
| 4 | [Evaluating and Mitigating Relationship Hallucinations in Large Vision-Language Models](https://doi.org/10.1109/TPAMI.2026.3656175) — **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2026** | Studies relationship hallucination with R-Bench and region-level alignment mitigation. | Strong high-impact journal reference for the application problem (visual grounding and hallucination), especially relation questions. Its method is not internal circuit stability measurement. |

## Directly relevant methodological paper

### NOTICE: semantic image corruption and mechanistic VLM analysis

[What Do VLMs NOTICE? A Mechanistic Interpretability Pipeline for Gaussian-Noise-free Text-Image Corruption and Evaluation](https://aclanthology.org/2025.naacl-long.571/) — **NAACL 2025, Long Paper**

NOTICE introduces semantic image-pair corruption and uses causal mediation/activation patching to investigate attention-head roles in BLIP and LLaVA. This is the most direct methodological neighbor for semantic visual perturbations plus internal VLM analysis. NAACL is a selective, peer-reviewed NLP venue, but it is listed here by its actual venue rather than labeled A* or IEEE Transactions. Read it closely even if your bibliography is restricted to a ranking category.

## Supporting high-impact application reference

[Evaluating and Mitigating Relationship Hallucinations in Large Vision-Language Models](https://doi.org/10.1109/TPAMI.2026.3656175) — **IEEE TPAMI, 2026**

The paper frames hallucinated object relationships and weak use of visual content as practical reliability problems. It is especially relevant if the project’s question set includes spatial or relational VQA. Use it to motivate downstream importance, not as evidence that pathway consistency has been validated.

## Methodological caveats to carry into the project

- **Ablation and activation patching are intervention choices, not direct proof of a complete “reasoning circuit.”** Describe recovered units as approximate mediators under the project’s intervention and answer-scoring setup.
- **Input corruption can create artifacts.** Human review, semantic-preservation checks, and area-matched spatial controls are important parts of the protocol, not optional polish.
- **Head ranking and greedy mediator recovery may be sensitive to choices.** Predefine layer/head candidate ranges, top-k, patch locations, target score, and robustness checks; include alternative intervention/control analyses.
- **A stable answer is not the same as a stable mechanism.** This is the project’s central hypothesis, but the pilot subgroup is small and must be confirmed on new images.
- **Do not claim general hallucination or OOD prediction from the current transformation-family target.** State the exact outcome: failure on a held-out, reviewed transformation family, unless a broader external evaluation is run.

## Problem and potential applications

The underlying problem is that a multimodal model may keep giving the same answer after a small, meaning-preserving image change while relying on a different internal route—or on a shortcut unrelated to the question-critical evidence. Ordinary answer accuracy alone cannot detect that behavior. The proposed analysis measures external evidence effects, approximate internal mediator recovery, and mediator consistency across reviewed variants.

If independently validated, this could support:

1. **Model auditing:** flag examples where answer stability hides sensitivity of the evidence-to-answer pathway.
2. **Model comparison:** compare base and fine-tuned models on evidence use and transformation robustness, separately from accuracy.
3. **Human review triage:** prioritize examples with unstable or poorly controlled evidence pathways for inspection.
4. **Targeted mitigation research:** use validated causal interventions to test whether reinforcing relevant visual information improves reliability.

These are potential research applications. The pilot result alone does not establish that the score detects hallucinations, improves accuracy, or transfers to general OOD settings.

## Recommended reading order

1. Read **NOTICE** for semantic image perturbation and mechanistic VLM analysis.
2. Read **CausalMM (ICLR 2025)** for confounding, attention causality, and counterfactual visual/language attention interventions.
3. Read **Modular Attribution and Intervention (ICLR 2025)** for causal mediation and hallucination-head identification.
4. Read **FCCT (AAAI 2026)** for more recent fine-grained multimodal causal tracing.
5. Read **TPAMI R-Bench** for the downstream relationship-hallucination task and region-level alignment application.

## Sources checked

- [ICLR proceedings: CausalMM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/86fe62e3b315d2578721562d9fd1a433-Abstract-Conference.html)
- [ICLR proceedings: Modular Attribution and Intervention](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8001c3568152d134d821cd46d4d84768-Abstract-Conference.html)
- [AAAI proceedings: FCCT](https://ojs.aaai.org/index.php/AAAI/article/view/40431)
- [ACL Anthology: NOTICE](https://aclanthology.org/2025.naacl-long.571/)
- [DOI: TPAMI R-Bench](https://doi.org/10.1109/TPAMI.2026.3656175)
