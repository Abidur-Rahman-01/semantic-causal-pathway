# Benchmark Data Card: Image Evidence Stability Benchmark

## 1. Dataset Description & Motivation

The **Image Evidence Stability Benchmark** measures whether a vision-language model (VLM) that produces consistent answers across meaning-preserving image transformations maintains reliance on the question-relevant visual evidence, and whether that reliance persists under a held-out transformation family.

- **Primary question:** When a VLM gives the same answer after a meaning-preserving image change, does it still rely on the visual evidence needed to answer the question, and does that reliance persist under a different, held-out transformation?
- **Primary outcome:** Failure on at least one accepted held-out variant (`heldout_failure = 1` if score < 0.5 among baseline-correct examples).
- **Secondary outcome:** Held-out answer flip (`heldout_answer_flip = 1` if answer text changes, regardless of correctness).
- **Primary Unit:** `image_id`. All questions, masks, controls, probe variants, and held-out variants for an image are grouped into the same split and bootstrap unit.

---

## 2. Dataset Composition & Sources

The benchmark utilizes curated, reviewed examples sourced from:
1. **VQAv2 (Visual Question Answering v2.0):** Natural images from COCO 2014 with 10 human reference answers per question.
2. **GQA (Visual Reasoning in the Real World v1.2):** Balanced scene-graph-grounded questions with single canonical reference answers.

### Curated Cohorts Available in Release
- **`manifests/pilot_200_manifest.jsonl`:** 200 samples (100 VQAv2, 100 GQA). Train: 120, Val: 40, Test: 40.
- **`manifests/confirmatory_500_manifest.jsonl`:** 500 samples (250 VQAv2, 250 GQA). Train: 300, Val: 100, Test: 100.
- **`manifests/confirmatory_1000_manifest.jsonl`:** 1,000 samples (500 VQAv2, 500 GQA). Train: 400, Val: 200, Test: 400.

---

## 3. Human Review & Quality Gates

Every candidate transformation variant must pass **four strict human review gates** before entering an evaluation manifest:
1. **Answerability:** Can the question still be unequivocally answered from the modified image?
2. **Evidence Preservation:** Is the critical visual evidence required for the answer fully visible and uncorrupted?
3. **Relation Preservation:** Are spatial and semantic object relationships preserved?
4. **Artifact Check:** Did the edit introduce visual artifacts or distractors that alter the answer?

All decisions are recorded in `annotations/variant_reviews.jsonl` with reviewer IDs and timestamps. Any candidate variant failing any gate is rejected, logged in `annotations/rejected_variants.jsonl` with an explicit reason, and excluded from inference manifests.

### Rejection Reason Breakdown
- `non_disjoint_or_unassigned_transform_family`: Candidate overlapped with probe/heldout roles.
- `severe_compression_artifacts_degraded_semantic_evidence`: Aggressive compression degraded evidence.
- `edge_blurring_obscures_critical_attributes`: Filtering obscured key visual attributes.
- `critical_evidence_truncated_by_image_boundary`: Geometric transformation cropped evidence.
- `pixel_saturation_clipped_texture_features`: Extreme contrast/brightness clipped textures.

---

## 4. Transformation Families & Disjoint Role Allocation

Transformation families are strictly partitioned to prevent leakage:
- **Probe Variants (In-distribution stability):** Photometric JPEG compression (`jpeg_quality_92`, `jpeg_quality_98`). Used to measure answer consistency and white-box pathway stability.
- **Held-out Variants (Primary evaluation):** Geometric translation (`translate_up_5pct`, `translate_down_5pct`, `translate_left_5pct`, `translate_right_5pct`). Used exclusively to evaluate failure and answer flips.

Probe and held-out families remain disjoint across all samples and splits.

---

## 5. Splits and Leakage Prevention

- Assignments are made deterministically at the `image_id` level using fixed seed 17.
- Analysis splits: Train (60%), Validation (20%), Test (20%).
- Canonical SHA-256 hashes of all source images are verified to ensure zero image content crosses splits.
- Candidate answer bins are frozen before model inference.

---

## 6. Licensing & Attribution

- **VQAv2:** Non-commercial research license; images are from MS-COCO under Creative Commons Attribution 4.0.
- **GQA:** Visual Genome images licensed under Creative Commons Attribution 4.0 International License.
- **Annotations & Manifests:** Released under MIT License.
- Users must obtain source raw COCO and GQA images from official repositories if redistributing derivative weights or evaluations.
