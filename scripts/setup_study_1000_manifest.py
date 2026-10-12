"""Setup and lock the 1,000-image study manifest with 400 Train / 200 Val / 400 Locked Test splits.

Enforces:
1. Exactly 1,000 independent images (500 VQAv2, 500 GQA), 1 question per image.
2. Balanced 400 Train (200 VQA, 200 GQA), 200 Validation (100 VQA, 100 GQA), 400 Locked Test (200 VQA, 200 GQA).
3. The 400 Locked Test images are strictly fresh and unpeeked (disjoint from previous 500-study test set).
4. Strictly zero leakage across partitions.
"""
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    print("=" * 70)
    print("Building Fresh 1,000-Image Study Manifest (400 Train / 200 Val / 400 Test)")
    print("=" * 70)

    data_root = ROOT / ".venv" / "data" / "vqa_v2"
    source_manifest = data_root / "approved_confirmatory_1000_manifest.jsonl"
    old_confirmatory = data_root / "approved_confirmatory_manifest.jsonl"

    # Identify old 500-study test IDs
    old_test_ids = set()
    with open(old_confirmatory, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("analysis_split") == "test":
                old_test_ids.add(r["sample_id"])

    print(f"Identified {len(old_test_ids)} previously examined test IDs from 500-image study.")

    all_rows = []
    with open(source_manifest, "r", encoding="utf-8") as f:
        for line in f:
            all_rows.append(json.loads(line))

    vqa_rows = [r for r in all_rows if "vqa" in r["dataset"].lower()]
    gqa_rows = [r for r in all_rows if "gqa" in r["dataset"].lower()]
    print(f"Loaded {len(vqa_rows)} VQAv2 rows, {len(gqa_rows)} GQA rows (Total: {len(all_rows)})")

    rng = random.Random(17)

    # Separate unpeeked rows from old test rows
    vqa_unpeeked = [r for r in vqa_rows if r["sample_id"] not in old_test_ids]
    gqa_unpeeked = [r for r in gqa_rows if r["sample_id"] not in old_test_ids]
    vqa_old_test = [r for r in vqa_rows if r["sample_id"] in old_test_ids]
    gqa_old_test = [r for r in gqa_rows if r["sample_id"] in old_test_ids]

    rng.shuffle(vqa_unpeeked)
    rng.shuffle(gqa_unpeeked)
    rng.shuffle(vqa_old_test)
    rng.shuffle(gqa_old_test)

    # 1. Locked Test: 200 VQA, 200 GQA (strictly drawn from unpeeked pool)
    test_vqa = vqa_unpeeked[:200]
    test_gqa = gqa_unpeeked[:200]
    locked_test_rows = test_vqa + test_gqa
    for r in locked_test_rows:
        r["analysis_split"] = "test"

    # Verify zero overlap between locked_test_rows and old_test_ids
    test_overlap = {r["sample_id"] for r in locked_test_rows} & old_test_ids
    assert len(test_overlap) == 0, f"Error: {len(test_overlap)} overlapping test samples!"
    print("Verification: 400 Locked Test images have ZERO overlap with previous 500-image test set.")

    # 2. Validation: 100 VQA, 100 GQA
    rem_vqa = vqa_unpeeked[200:] + vqa_old_test
    rem_gqa = gqa_unpeeked[200:] + gqa_old_test
    rng.shuffle(rem_vqa)
    rng.shuffle(rem_gqa)

    val_vqa = rem_vqa[:100]
    val_gqa = rem_gqa[:100]
    val_rows = val_vqa + val_gqa
    for r in val_rows:
        r["analysis_split"] = "validation"

    # 3. Train: Remaining 200 VQA, 200 GQA (400 total)
    train_vqa = rem_vqa[100:300]
    train_gqa = rem_gqa[100:300]
    train_rows = train_vqa + train_gqa
    for r in train_rows:
        r["analysis_split"] = "train"

    # Combine all
    final_rows = train_rows + val_rows + locked_test_rows
    rng.shuffle(final_rows)

    print("\nDataset Allocation Summary:")
    print(f"  Train Split      : {len(train_rows)} images ({len(train_vqa)} VQAv2, {len(train_gqa)} GQA)")
    print(f"  Validation Split : {len(val_rows)} images ({len(val_vqa)} VQAv2, {len(val_gqa)} GQA)")
    print(f"  Locked Test Split: {len(locked_test_rows)} images ({len(test_vqa)} VQAv2, {len(test_gqa)} GQA)")
    print(f"  Total Images     : {len(final_rows)}")

    # Check zero leakage by image_id
    train_ids = {r["image_id"] for r in train_rows}
    val_ids = {r["image_id"] for r in val_rows}
    test_ids = {r["image_id"] for r in locked_test_rows}
    assert len(train_ids & val_ids) == 0, "Train-Val leakage!"
    assert len(train_ids & test_ids) == 0, "Train-Test leakage!"
    assert len(val_ids & test_ids) == 0, "Val-Test leakage!"
    print("Zero-leakage verification: PASSED (0 overlapping image IDs across splits).")

    # Save manifest
    dest_path = data_root / "approved_study_1000_manifest.jsonl"
    with open(dest_path, "w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved manifest to: {dest_path}")

    # Copy to data archive
    archive_dir = ROOT / "data" / "study_1000"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "approved_study_1000_manifest.jsonl").write_text(dest_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Archive copy saved to: {archive_dir / 'approved_study_1000_manifest.jsonl'}")

if __name__ == "__main__":
    main()
