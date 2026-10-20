# Benchmark Model Card: Vision-Language Model Inference & Causal Pathway Tracking

## 1. Model Details

The benchmark evaluates Vision-Language Models (VLMs) across black-box behavioral stability and white-box causal pathway metrics.

### Primary Evaluated Architecture: Qwen2.5-VL-7B-Instruct
- **Developer:** Alibaba Cloud / Qwen Team
- **Model Checkpoint Revision:** `Qwen/Qwen2.5-VL-7B-Instruct`
- **Architecture:** Multimodal transformer incorporating vision transformer encoder and autoregressive language backbone with rotary position embeddings and cross-attention/patch merging.
- **Precision:** `bfloat16` / `float16`
- **Inference Library:** PyTorch with HuggingFace Transformers / vLLM backend.

---

## 2. Inference & Generation Protocol

Inference parameters are strictly pinned and frozen before evaluation:
- **System Prompt:** Standard conversational VLM instruction or empty string (recorded in `run_config.json`).
- **Prompt Template:** Image token placeholder followed by question and candidate answer prompt format.
- **Maximum Generated Tokens:** 16 (for targeted short-form answer generation) or 64.
- **Decoding Method:** Greedy decoding (`temperature=0.0`, `top_p=1.0`, `do_sample=False`) for deterministic reproducibility.
- **Answer Extraction:** Raw answer string preserved verbatim in `raw_answer`; normalized answers stored separately using official scorers (`vqav2_soft_consensus_v2.0` and `gqa_exact_match_v1.2`).
- **Confidence Scores:** Normalized token probabilities over frozen candidate answer bins where accessible; left undefined/null when token logprobs are unavailable (no invented confidences).

---

## 3. White-Box Track: Activation Patching & Circuit Metrics

For white-box evaluations (Track B), causal pathway mediators are identified and measured:
- **Mediator Search:** Greedy top-k attention head identification based on indirect effect on candidate answer logit margins under visual evidence masking.
- **Candidate Pool:** Attention heads across middle and late transformer layers (e.g. layers 12-27).
- **Intervention Type:** Gaussian blur replacement of evidence region vs control patches (ablations test zero/gray and inpainting).
- **Candidate Pathway Metrics:**
  - **CPS (Causal Pathway Stability):**
    - Hard CPS: Jaccard set overlap of top-k recovered heads between original and probe images.
    - Weighted CPS: Jensen-Shannon similarity of effect distributions across recovered heads.
  - **CECA (Causal Effect Distribution Agreement):** Mean pairwise 1 - normalized JS divergence of mediator effect distributions across transformations.
  - **SCC (Semantic Causal Consistency):** Product of CECA and CPS (`SCC = max(0, min(1, CECA)) * max(0, min(1, CPS))`).
- **Accounting for Undefined Values:** When mediator search fails or effect sum is zero, metrics are preserved as null with explicit reason codes (`metric_undefined_reasons`); they are never silently imputed as zero or perfect.

---

## 4. Evaluated Baselines

All candidate white-box metrics are benchmarked against prespecified black-box baselines:
1. **Probe Behavioral Inconsistency:** `1.0 - probe_answer_consistency`
2. **Output Low Confidence:** `1.0 - max(candidate_probabilities)`
3. **Normalized Candidate Entropy:** Normalized Shannon entropy over candidate answer distribution.
4. **External Evidence Necessity:** Jensen-Shannon divergence induced on candidate distribution by masking the visual evidence.
5. **Control-Adjusted Evidence Necessity:** Evidence necessity minus mean spatial control necessity.

---

## 5. Claims Boundaries & Limitations

1. **Transformation-Specific Scope:** The benchmark measures behavior under the declared photometric and geometric transformations. It does not by itself establish general out-of-distribution robustness, hallucination detection, or human-like reasoning.
2. **Circuit Completeness:** Activation patching estimates effects under specific masking and token alignment assumptions; it does not claim to recover a full or unique reasoning circuit.
3. **Predictive Open Hypothesis:** Empirical evaluations on 500 and 1,000 samples show that SCC and CPS do not significantly outperform black-box probe inconsistency in predicting held-out failures (e.g., test Low-SCC AUROC ~ 0.447 vs Probe Inconsistency AUROC ~ 0.638). SCC must be treated as an open hypothesis rather than an established failure predictor.
