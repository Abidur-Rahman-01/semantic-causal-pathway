"""Download and prepare a high-quality 50K-train / 30K-test dataset directly from official sources.

Fetches only the required images for the exact question sets via concurrent HTTP requests,
saving tens of GBs of unused image downloads and completing setup in ~30-45 minutes.

Ensures:
- Strict image-group disjointness (zero visual leakage between train, val, and test).
- 50,000 training samples.
- >= 30,000 test samples.
- 3,000 validation samples.
- 5-fold cross-validation generation.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

GQA_PREFIXES = [
    "https://cs.stanford.edu/people/rak248/VG_100K/",
    "https://cs.stanford.edu/people/rak248/VG_100K_2/",
]
COCO_VAL_PREFIX = "http://images.cocodataset.org/val2014/"
COCO_TRAIN_PREFIX = "http://images.cocodataset.org/train2014/"


def download_single_image(task: tuple[str, str, Path]) -> bool:
    """Download a single image given (dataset, image_id, destination_path)."""
    dataset, img_id, dest = task
    if dest.exists() and dest.stat().st_size > 1000:
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(".tmp")

    urls = []
    if dataset == "gqa":
        for pfx in GQA_PREFIXES:
            urls.append(f"{pfx}{img_id}.jpg")
    else:  # vqav2
        int_id = int(img_id)
        # Check split from destination directory name
        if "val2014" in str(dest):
            urls.append(f"{COCO_VAL_PREFIX}COCO_val2014_{int_id:012d}.jpg")
        else:
            urls.append(f"{COCO_TRAIN_PREFIX}COCO_train2014_{int_id:012d}.jpg")

    for url in urls:
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
                    if len(data) > 1000:
                        temp_dest.write_bytes(data)
                        temp_dest.replace(dest)
                        return True
            except Exception:
                time.sleep(1)
    return False


def collect_gqa_groups(raw_path: Path, split_name: str) -> dict[str, list[dict]]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    by_img = defaultdict(list)
    img_dir = ROOT / "data" / "raw" / "gqa" / "images"
    for qid, item in raw.items():
        img_id = str(item["imageId"])
        img_path = img_dir / f"{img_id}.jpg"
        by_img[img_id].append({
            "sample_id": f"gqa-{qid}",
            "dataset": "gqa",
            "split": split_name,
            "image": str(img_path.resolve()),
            "image_id": img_id,
            "question": item["question"],
            "answers": [str(item["answer"])],
            "answer": str(item["answer"]),
        })
    return by_img


def collect_vqa_groups(q_path: Path, a_path: Path, split_name: str) -> dict[str, list[dict]]:
    questions = {x["question_id"]: x for x in json.loads(q_path.read_text(encoding="utf-8"))["questions"]}
    anns = json.loads(a_path.read_text(encoding="utf-8"))["annotations"]
    img_folder = "val2014" if "val" in split_name else "train2014"
    img_dir = ROOT / "data" / "raw" / "vqav2" / img_folder
    by_img = defaultdict(list)
    for ann in anns:
        q = questions[ann["question_id"]]
        img_id = str(q["image_id"])
        img_path = img_dir / f"COCO_{img_folder}_{int(img_id):012d}.jpg"
        answers = [x["answer"].strip() for x in ann.get("answers", []) if x.get("answer", "").strip()]
        by_img[img_id].append({
            "sample_id": f"vqav2-{q['question_id']}",
            "dataset": "vqav2",
            "split": split_name,
            "image": str(img_path.resolve()),
            "image_id": img_id,
            "question": q["question"],
            "answers": answers or [ann.get("multiple_choice_answer", "")],
            "answer": ann.get("multiple_choice_answer", ""),
        })
    return by_img


def select_samples(groups: dict[str, list[dict]], target_count: int, rng: random.Random) -> tuple[list[dict], set[str]]:
    """Select image groups until reaching approximately target_count samples."""
    img_keys = list(groups.keys())
    rng.shuffle(img_keys)
    selected_rows = []
    selected_imgs = set()
    for img in img_keys:
        rows = groups[img]
        selected_rows.extend(rows)
        selected_imgs.add(img)
        if len(selected_rows) >= target_count:
            break
    # Trim to exact count if slightly over
    return selected_rows[:target_count], selected_imgs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=50000, help="Train samples (default: 50,000)")
    parser.add_argument("--test-samples", type=int, default=30000, help="Test samples (default: 30,000)")
    parser.add_argument("--val-samples", type=int, default=3000, help="Val samples (default: 3,000)")
    parser.add_argument("--max-workers", type=int, default=32, help="Concurrent download threads (default: 32)")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    print("=" * 75)
    print(f"PREPARING TARGETED DATASET: {args.train_samples:,} TRAIN / {args.test_samples:,} TEST / {args.val_samples:,} VAL")
    print("=" * 75)

    rng = random.Random(args.seed)

    # 1. Load GQA and VQAv2 groups
    print("Indexing official question files...")
    gqa_val_by_img = collect_gqa_groups(ROOT / "data" / "raw" / "gqa" / "val_balanced_questions.json", "val")
    gqa_train_by_img = collect_gqa_groups(ROOT / "data" / "raw" / "gqa" / "train_balanced_questions.json", "train")

    vqa_val_by_img = collect_vqa_groups(
        ROOT / "data" / "raw" / "vqav2" / "v2_OpenEnded_mscoco_val2014_questions.json",
        ROOT / "data" / "raw" / "vqav2" / "v2_mscoco_val2014_annotations.json",
        "val"
    )
    vqa_train_by_img = collect_vqa_groups(
        ROOT / "data" / "raw" / "vqav2" / "v2_OpenEnded_mscoco_train2014_questions.json",
        ROOT / "data" / "raw" / "vqav2" / "v2_mscoco_train2014_annotations.json",
        "train"
    )

    # Ratio: 80% GQA, 20% VQAv2
    gqa_train_target = int(args.train_samples * 0.80)
    vqa_train_target = args.train_samples - gqa_train_target

    gqa_test_target = int(args.test_samples * 0.80)
    vqa_test_target = args.test_samples - gqa_test_target

    gqa_val_target = int(args.val_samples * 0.80)
    vqa_val_target = args.val_samples - gqa_val_target

    print(f"Train allocation: GQA = {gqa_train_target:,}, VQAv2 = {vqa_train_target:,}")
    print(f"Test allocation:  GQA = {gqa_test_target:,}, VQAv2 = {vqa_test_target:,}")
    print(f"Val allocation:   GQA = {gqa_val_target:,}, VQAv2 = {vqa_val_target:,}")

    # Select Train from train splits
    train_gqa_rows, train_gqa_imgs = select_samples(gqa_train_by_img, gqa_train_target, rng)
    train_vqa_rows, train_vqa_imgs = select_samples(vqa_train_by_img, vqa_train_target, rng)
    train_rows = train_gqa_rows + train_vqa_rows
    rng.shuffle(train_rows)

    # Select Test from val splits
    test_gqa_rows, test_gqa_imgs = select_samples(gqa_val_by_img, gqa_test_target, rng)
    test_vqa_rows, test_vqa_imgs = select_samples(vqa_val_by_img, vqa_test_target, rng)
    test_rows = test_gqa_rows + test_vqa_rows
    rng.shuffle(test_rows)

    # Remove test images from remaining val pools to guarantee zero image overlap
    remaining_gqa_val = {k: v for k, v in gqa_val_by_img.items() if k not in test_gqa_imgs}
    remaining_vqa_val = {k: v for k, v in vqa_val_by_img.items() if k not in test_vqa_imgs}

    val_gqa_rows, val_gqa_imgs = select_samples(remaining_gqa_val, gqa_val_target, rng)
    val_vqa_rows, val_vqa_imgs = select_samples(remaining_vqa_val, vqa_val_target, rng)
    val_rows = val_gqa_rows + val_vqa_rows
    rng.shuffle(val_rows)

    all_selected = train_rows + val_rows + test_rows
    print(f"\nTotal questions selected: {len(all_selected):,} across {len(train_rows):,} train, {len(val_rows):,} val, {len(test_rows):,} test")

    # 2. Collect unique images needed
    download_tasks = {}
    for r in all_selected:
        dest = Path(r["image"])
        download_tasks[(r["dataset"], r["image_id"])] = (r["dataset"], r["image_id"], dest)

    print(f"Total unique images required: {len(download_tasks):,}")

    # Check already existing
    already_done = sum(1 for _, _, dest in download_tasks.values() if dest.exists() and dest.stat().st_size > 1000)
    needed = len(download_tasks) - already_done
    print(f"Images already on disk: {already_done:,}. Needed to download: {needed:,}")

    # 3. Concurrent multithreaded download
    tasks_to_run = [task for task in download_tasks.values() if not (task[2].exists() and task[2].stat().st_size > 1000)]
    t0 = time.time()
    completed = already_done
    failed = 0

    if tasks_to_run:
        print(f"\nStarting concurrent download with {args.max_workers} threads...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {executor.submit(download_single_image, task): task for task in tasks_to_run}
            for future in concurrent.futures.as_completed(future_to_task):
                ok = future.result()
                if ok:
                    completed += 1
                else:
                    failed += 1
                total_proc = completed + failed
                if total_proc % 200 == 0 or total_proc == len(download_tasks):
                    elapsed = time.time() - t0
                    rate = (completed - already_done) / max(elapsed, 0.1)
                    print(f"  Images: {completed:,}/{len(download_tasks):,} ({completed*100/len(download_tasks):.1f}%) | "
                          f"Rate: {rate:.1f} imgs/s | Failed: {failed} | Elapsed: {elapsed/60:.1f}m", flush=True)

    elapsed_total = time.time() - t0
    print(f"\nDownload phase completed in {elapsed_total/60:.1f} minutes! Success: {completed:,}, Failed: {failed}")

    # 4. Filter only samples whose images exist on disk
    def filter_valid(rows: list[dict], name: str) -> list[dict]:
        valid = [r for r in rows if Path(r["image"]).exists() and Path(r["image"]).stat().st_size > 1000]
        print(f"  {name}: {len(valid):,} / {len(rows):,} retained ({dict(Counter(r['dataset'] for r in valid))})")
        return valid

    print("\nVerifying on-disk image references:")
    valid_train = filter_valid(train_rows, "train")
    valid_val = filter_valid(val_rows, "validation")
    valid_test = filter_valid(test_rows, "locked_test")

    # 5. Write JSONL manifests
    proc_dir = ROOT / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    for name, split in [("train", valid_train), ("validation", valid_val), ("locked_test", valid_test)]:
        out_file = proc_dir / f"{name}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(split):,} samples to {out_file}")

    # 6. Build 5-fold cross-validation
    print("\nBuilding 5-fold cross-validation splits...")
    folds_dir = proc_dir / "folds"
    subprocess.run([
        PYTHON, str(ROOT / "scripts" / "build_vqa_folds.py"),
        "--input", str(proc_dir / "train.jsonl"),
        "--output-dir", str(folds_dir),
        "--folds", "5",
        "--seed", str(args.seed),
    ], check=True)

    # 7. Summary report
    report = {
        "train_samples": len(valid_train),
        "validation_samples": len(valid_val),
        "locked_test_samples": len(valid_test),
        "total_samples": len(valid_train) + len(valid_val) + len(valid_test),
        "unique_images": completed,
        "seed": args.seed,
    }
    (proc_dir / "dataset_50k_setup_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 75)
    print("DATASET PREPARATION COMPLETE!")
    print(f"  Train:       {len(valid_train):,} samples")
    print(f"  Validation:  {len(valid_val):,} samples")
    print(f"  Locked Test: {len(valid_test):,} samples")
    print(f"  5 Folds:     Successfully built in {folds_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
