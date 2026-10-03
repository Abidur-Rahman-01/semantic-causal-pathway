# Data preparation for the semantic-circuit experiment

Do not commit benchmark images or masks. Acquire VQA v2, GQA, POPE/POPEv2, Winoground, or UNK-VQA from their official sources and follow their terms. Preserve the official partitions and IDs; make one JSONL per split. Store the image archive and generated annotations outside source control.

## JSONL fields

Required:

```json
{"sample_id":"gqa-001","image":"images/0001.jpg","question":"What color is the umbrella?"}
```

Recommended for the mechanistic pilot:

```json
{"sample_id":"gqa-001","image":"images/0001.jpg","question":"What color is the umbrella?","answers":["red"],"candidate_answers":["red","blue","green","other"],"critical_concepts":["umbrella"],"evidence_mask":"masks/0001_umbrella.png","evidence_mask_reviewed":true,"question_type":"attribute_color","split":"validation","variants":[{"transform":"jpeg_quality_92","image":"variants/gqa-001__jpeg_quality_92.png","answer_preserved":true,"critical_evidence_preserved":true,"relations_preserved":true,"human_audited":true},{"transform":"contrast_1.03","image":"variants/gqa-001__contrast_1.03.png","answer_preserved":true,"critical_evidence_preserved":true,"relations_preserved":true,"human_audited":true}]}
```

Paths can be absolute or relative to `data_root` / the manifest directory. Set `evidence_mask_reviewed` to `true` only after visual inspection. The batch runner requires this review flag.

## Critical evidence masks

Store one single-channel image per critical evidence set, at the source image's dimensions. Pixels greater than zero are treated as evidence. Masks can come from reviewed Grounding DINO + SAM2 proposals or manual annotation. Save annotation provenance and reviewer decisions. For relation questions, annotate subject and object evidence, relation label, and whether the relation is visibly supported; object presence alone is not sufficient evidence for a relation.

## Semantic-equivalence variants

Use `scripts/prepare_variants.py` to create candidates from the supported transformation families: mild photometric edits, whole-frame translations, evidence-preserving background blur/neutral replacement, and optional removal of explicitly annotated noncritical objects. Color questions omit brightness and contrast edits because they can change color evidence. Geometric candidates retain the full frame and translate all objects together; spatial relations should still be checked. Background and inpainting candidates may introduce artifacts. Keep generated variants outside source control. Every candidate requires four explicit booleans before analysis:

Whole-frame translations receive a paired evidence mask with the same affine translation, so the deletion and patching intervention follows the evidence to its new coordinates. Review the candidate mask alongside its transformed image.

- `answer_preserved`
- `critical_evidence_preserved`
- `relations_preserved`
- `human_audited`

The prepared manifest defaults every field to false. Fill the review sheet after inspecting each transformed image, then run `scripts/apply_variant_review.py` to create the approved manifest. Record reviewer and date in the CSV. The runner checks the flags, image dimensions, and minimum of two accepted variants. Use the same source sample's official split for all variants; never distribute related variants across train/test.

## Spatial control masks

`scripts/prepare_controls.py` translates the target evidence mask to candidate same-area, same-shape control locations. These controls estimate generic deletion sensitivity; they do not prove that the translated region is semantically irrelevant. Inspect each proposed mask and reject locations covering another required object or important scene evidence. Apply decisions with `scripts/apply_control_review.py`. The default experiment config requires at least three accepted, reviewed control masks per sample.

## Fixed candidate answers

Intervention experiments normalize likelihoods over a fixed, per-example candidate set. This is a restricted comparison distribution, not the model's full output distribution. Include the baseline model answer, reference answer(s), plausible alternatives, and an `other` choice where appropriate. Use identical candidates for original, counterfactual, and patched runs.
