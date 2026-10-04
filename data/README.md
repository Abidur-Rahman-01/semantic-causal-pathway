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
# Datasets and download links

This project uses two established visual question answering datasets:

1. **VQAv2**: [official download page](https://visualqa.org/download.html). Download the train and validation question/annotation JSON files and the matching COCO 2014 `train2014` and `val2014` image archives. VQAv2 is widely used in MLLM evaluation, but its known language priors mean results should be paired with other benchmarks.
2. **GQA Balanced**: [official download page](https://cs.stanford.edu/people/dorarad/gqa/download.html). Download the balanced train/validation questions and the GQA images archive. GQA emphasizes compositional visual reasoning and gives a complementary evaluation condition.

Dataset owners host these large files and set their terms. This repository intentionally provides download links and conversion code rather than redistributing image archives. Follow the terms on the official pages. Keep the official training, validation, and test partitions intact. Do not tune on test labels.

## Recommended dataset setup for Qwen fine-tuning

For this experiment, train on **both full training splits**: VQAv2 and GQA Balanced. VQAv2 provides broad, human-annotated questions with 10 answers per training example; GQA Balanced adds compositional questions and short exact-match targets. Combining them gives a larger and more varied training set than either alone. Keep their scores separate at evaluation time because their answer distributions and metrics differ.

Use all official train examples for training. The official validation examples are labeled, so this project reserves 90% of their images for validation and 10% as a locked local test set. The holdout is image-disjoint within each dataset and is intended for one final local evaluation. VQAv2's official test questions have no released answers; use them only for a benchmark submission if you need official test results. Never train on the locked test set.

The full archives are large and are hosted by the dataset owners. The automated setup below downloads the official source archives, uses the question files to identify every referenced image, and retains only those images locally. Temporary zip files are removed after extraction unless `--keep-archives` is passed. Budget for the large temporary archives and extracted images at the same time (at least 60 GB free is a practical minimum); GQA images are about 20 GB compressed. This repository does not redistribute dataset files.

## Expected local folders

```text
data/raw/vqav2/
  train2014/COCO_train2014_000000000001.jpg
  val2014/COCO_val2014_000000000001.jpg
  v2_OpenEnded_mscoco_train2014_questions.json
  v2_mscoco_train2014_annotations.json
  v2_OpenEnded_mscoco_val2014_questions.json
  v2_mscoco_val2014_annotations.json
data/raw/gqa/
  images/000001.jpg
  train_balanced_questions.json
  val_balanced_questions.json
```

The GQA question filenames are the names used by the official download page; image IDs in its JSON determine corresponding image filenames. Each converted row contains an absolute image path, question, sample/image IDs, dataset name, split, and reference answers. To download all labeled train/validation sources, retain only their referenced images, convert the four splits, build combined manifests, and create five image-grouped training folds, run this from the repository root:

```bash
python scripts/setup_full_vqa_dataset.py
```

The script downloads VQAv2 train/validation questions and annotations, the COCO train2014 and val2014 image archives, GQA Questions 1.2, and GQA Images from their official hosts. It extracts only images referenced by VQAv2 and GQA Balanced train/validation questions. Downloads resume after interruption. The setup report is written to `data/processed/dataset_setup_report.json` and records missing archive members. Check the retained row counts and report before training. To preserve the full source archives after extraction, add `--keep-archives`.

This setup targets all labeled train and validation questions. It excludes VQAv2 test questions because answers are not publicly released, and excludes GQA unbalanced and test questions by design. Rerun the same command after an interruption to continue setup.

For manual setup, once source files are in the expected layout, convert the four splits:

```bash
python scripts/convert_vqa_datasets.py vqav2 --root data/raw/vqav2 --split train --output data/processed/vqav2_train.jsonl
python scripts/convert_vqa_datasets.py vqav2 --root data/raw/vqav2 --split val --output data/processed/vqav2_val.jsonl
python scripts/convert_vqa_datasets.py gqa --root data/raw/gqa --split train --output data/processed/gqa_train.jsonl
python scripts/convert_vqa_datasets.py gqa --root data/raw/gqa --split val --output data/processed/gqa_val.jsonl
python scripts/build_vqa_manifests.py --data-dir data/processed --test-fraction 0.1 --seed 17
```

The last command creates combined `train.jsonl`, `validation.jsonl`, and `locked_test.jsonl` files and prints per-dataset counts. It shuffles deterministically and reserves test examples by image ID, so questions about the same image stay together. Review the counts and verify the image paths before training. Reuse the same seed and keep `locked_test.jsonl` untouched until final evaluation.

## Five-fold cross-validation

Build five folds from the combined official training examples:

```bash
python scripts/build_vqa_folds.py --input data/processed/train.jsonl --output-dir data/processed/folds --folds 5 --seed 17
```

Each fold uses 4/5 of the training images to fit and the remaining 1/5 to validate. All questions about one image stay in the same fold, and fold assignments are balanced by question count within each dataset. Train a separate 15-epoch adapter for each fold and evaluate that adapter on its fold validation file. Compare the five validation results per dataset. After cross-validation, fit the final adapter on the full `train.jsonl`, select checkpoints using `validation.jsonl`, then evaluate `locked_test.jsonl` once.

```bash
for fold in 0 1 2 3 4; do
  python scripts/train_qwen_vl_lora.py --train "data/processed/folds/fold_${fold}/train.jsonl" --validation "data/processed/folds/fold_${fold}/validation.jsonl" --output-dir "outputs/qwen-vqa-lora/fold_${fold}" --epochs 15 --batch-size 1 --grad-accumulation 8
done
```

The full datasets require substantial storage and compute. Use the combined training file for optimization, validation for model selection, and the locked test file for final local reporting. Keep any limited pilot subsets clearly labeled as pilots.
