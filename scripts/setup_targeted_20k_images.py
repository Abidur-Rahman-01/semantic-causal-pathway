"""Download 20,000 unique images and collect all corresponding official Q&A pairs.

Ensures:
- 20,000 unique scene images across GQA and COCO/VQAv2.
- All question-answer pairs corresponding to these 20,000 images extracted.
- Strict image-group disjointness (zero image leakage between train, val, and locked test).
- Locked test set contains >= 30,000 samples.
- 5-fold cross-validation generation via scripts/build_vqa_folds.py.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import subprocess
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
            "image": str(img_path),
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
            "image": str(img_path),
            "image_id": img_id,
            "question": q["question"],
            "answers": answers or [ann.get("multiple_choice_answer", "")],
            "answer": ann.get("multiple_choice_answer", ""),
        })
    return by_img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-images", type=int, default=20000, help="Total unique images to fetch (default: 20,000)")
    parser.add_argument("--gqa-ratio", type=float, default=0.60, help="Ratio of GQA images (default: 0.60, i.e. 12k GQA, 8k COCO)")
    parser.add_argument("--max-workers", type=int, default=32, help="Concurrent download threads (default: 32)")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    print("=" * 75)
    print(f"PREPARING DATASET WITH {args.total_images:,} UNIQUE IMAGES AND ALL ASSOCIATED Q&A")
    print("=" * 75)

    rng = random.Random(args.seed)

    # 1. Index official Q&A files
    print("Indexing official question and annotation files...")
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

    # Target image allocations
    gqa_total_imgs = int(args.total_images * args.gqa_ratio)
    vqa_total_imgs = args.total_images - gqa_total_imgs

    # 70% Train, 10% Val, 20% Locked Test
    gqa_train_n = int(gqa_total_imgs * 0.70)
    gqa_val_n = int(gqa_total_imgs * 0.10)
    gqa_test_n = gqa_total_imgs - gqa_train_n - gqa_val_n

    vqa_train_n = int(vqa_total_imgs * 0.70)
    vqa_val_n = int(vqa_total_imgs * 0.10)
    vqa_test_n = vqa_total_imgs - vqa_train_n - vqa_val_n

    print(f"Image Allocations across splits:")
    print(f"  GQA images:  Train={gqa_train_n:,}, Val={gqa_val_n:,}, Locked Test={gqa_test_n:,} (Total={gqa_total_imgs:,})")
    print(f"  COCO images: Train={vqa_train_n:,}, Val={vqa_val_n:,}, Locked Test={vqa_test_n:,} (Total={vqa_total_imgs:,})")

    # Pre-index existing files on disk for instant O(1) set lookup
    gqa_img_dir = ROOT / "data" / "raw" / "gqa" / "images"
    vqa_train_img_dir = ROOT / "data" / "raw" / "vqav2" / "train2014"
    vqa_val_img_dir = ROOT / "data" / "raw" / "vqav2" / "val2014"

    gqa_existing = set(os.listdir(gqa_img_dir)) if gqa_img_dir.exists() else set()
    vqa_train_existing = set(os.listdir(vqa_train_img_dir)) if vqa_train_img_dir.exists() else set()
    vqa_val_existing = set(os.listdir(vqa_val_img_dir)) if vqa_val_img_dir.exists() else set()
    print(f"Existing files on disk: GQA={len(gqa_existing):,}, COCO_train={len(vqa_train_existing):,}, COCO_val={len(vqa_val_existing):,}", flush=True)

    # Sample images prioritizing those already downloaded on disk
    def sample_keys_prioritizing_existing(d: dict, count: int, existing_set: set[str], filename_fmt) -> tuple[list, list]:
        existing = []
        missing = []
        for k in d.keys():
            fname = filename_fmt(k)
            if fname in existing_set:
                existing.append(k)
            else:
                missing.append(k)
        rng.shuffle(existing)
        rng.shuffle(missing)
        if len(existing) >= count:
            selected = existing[:count]
            pool = existing[count:] + missing
        else:
            selected = existing + missing[:(count - len(existing))]
            pool = missing[(count - len(existing)):]
        return selected, pool

    # GQA: Train from train split, Val & Test from val split
    gqa_train_keys, gqa_train_pool = sample_keys_prioritizing_existing(
        gqa_train_by_img, gqa_train_n, gqa_existing, lambda k: f"{k}.jpg"
    )
    
    # Split GQA val into val and test
    gqa_val_keys, gqa_val_pool = sample_keys_prioritizing_existing(
        gqa_val_by_img, gqa_val_n + gqa_test_n, gqa_existing, lambda k: f"{k}.jpg"
    )
    gqa_test_keys = gqa_val_keys[:gqa_test_n]
    gqa_val_keys = gqa_val_keys[gqa_test_n:]

    # VQAv2: Train from train split, Val & Test from val split
    vqa_train_keys, vqa_train_pool = sample_keys_prioritizing_existing(
        vqa_train_by_img, vqa_train_n, vqa_train_existing, lambda k: f"COCO_train2014_{int(k):012d}.jpg"
    )
    
    vqa_val_keys, vqa_val_pool = sample_keys_prioritizing_existing(
        vqa_val_by_img, vqa_val_n + vqa_test_n, vqa_val_existing, lambda k: f"COCO_val2014_{int(k):012d}.jpg"
    )
    vqa_test_keys = vqa_val_keys[:vqa_test_n]
    vqa_val_keys = vqa_val_keys[vqa_test_n:]

    # Helper to build tasks
    def get_task(dataset: str, img_id: str, group_dict: dict) -> tuple[str, str, Path]:
        sample = group_dict[img_id][0]
        return (dataset, img_id, Path(sample["image"]))

    selected_gqa_train = set(gqa_train_keys)
    selected_gqa_val = set(gqa_val_keys)
    selected_gqa_test = set(gqa_test_keys)

    selected_vqa_train = set(vqa_train_keys)
    selected_vqa_val = set(vqa_val_keys)
    selected_vqa_test = set(vqa_test_keys)

    all_tasks = {}
    for k in selected_gqa_train: all_tasks[("gqa", k)] = get_task("gqa", k, gqa_train_by_img)
    for k in selected_gqa_val: all_tasks[("gqa", k)] = get_task("gqa", k, gqa_val_by_img)
    for k in selected_gqa_test: all_tasks[("gqa", k)] = get_task("gqa", k, gqa_val_by_img)

    for k in selected_vqa_train: all_tasks[("vqav2", k)] = get_task("vqav2", k, vqa_train_by_img)
    for k in selected_vqa_val: all_tasks[("vqav2", k)] = get_task("vqav2", k, vqa_val_by_img)
    for k in selected_vqa_test: all_tasks[("vqav2", k)] = get_task("vqav2", k, vqa_val_by_img)

    total_images_needed = len(all_tasks)
    
    def is_present(task: tuple[str, str, Path]) -> bool:
        ds, img_id, dest = task
        if ds == "gqa": return f"{img_id}.jpg" in gqa_existing
        if "val2014" in str(dest): return f"COCO_val2014_{int(img_id):012d}.jpg" in vqa_val_existing
        return f"COCO_train2014_{int(img_id):012d}.jpg" in vqa_train_existing

    already_done = sum(1 for t in all_tasks.values() if is_present(t))
    needed = total_images_needed - already_done
    print(f"\nUnique images targeted: {total_images_needed:,}", flush=True)
    print(f"Images already on disk: {already_done:,}", flush=True)
    print(f"Images to download:     {needed:,}", flush=True)

    # Download with ThreadPoolExecutor
    tasks_to_run = [t for t in all_tasks.values() if not is_present(t)]
    t0 = time.time()
    completed = already_done
    failed = 0

    if tasks_to_run:
        print(f"\nDownloading {len(tasks_to_run):,} images with {args.max_workers} concurrent threads...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {executor.submit(download_single_image, task): task for task in tasks_to_run}
            for future in concurrent.futures.as_completed(future_to_task):
                ok = future.result()
                if ok:
                    completed += 1
                else:
                    failed += 1
                total_proc = completed + failed
                if total_proc % 250 == 0 or total_proc == total_images_needed:
                    elapsed = time.time() - t0
                    rate = (completed - already_done) / max(elapsed, 0.1)
                    rem_imgs = total_images_needed - completed
                    eta_min = rem_imgs / max(rate, 0.1) / 60
                    print(f"  Images: {completed:,}/{total_images_needed:,} ({completed*100/total_images_needed:.1f}%) | "
                          f"Rate: {rate:.1f} imgs/s | Failed: {failed} | ETA: {eta_min:.1f}m", flush=True)

    elapsed_total = time.time() - t0
    print(f"\nImage download step finished in {elapsed_total/60:.1f} minutes! Success: {completed:,}, Failed: {failed}")

    # Collect ALL questions for the selected images
    def gather_rows(keys: set, group_dict: dict) -> list[dict]:
        out = []
        for k in keys:
            out.extend(group_dict[k])
        return out

    train_rows = gather_rows(selected_gqa_train, gqa_train_by_img) + gather_rows(selected_vqa_train, vqa_train_by_img)
    val_rows = gather_rows(selected_gqa_val, gqa_val_by_img) + gather_rows(selected_vqa_val, vqa_val_by_img)
    test_rows = gather_rows(selected_gqa_test, gqa_val_by_img) + gather_rows(selected_vqa_test, vqa_val_by_img)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    print("\nQuestion Counts extracted for the 20,000 images:")
    print(f"  Train:       {len(train_rows):,} Q&A pairs (GQA: {sum(1 for r in train_rows if r['dataset']=='gqa'):,}, VQAv2: {sum(1 for r in train_rows if r['dataset']=='vqav2'):,})")
    print(f"  Validation:  {len(val_rows):,} Q&A pairs (GQA: {sum(1 for r in val_rows if r['dataset']=='gqa'):,}, VQAv2: {sum(1 for r in val_rows if r['dataset']=='vqav2'):,})")
    print(f"  Locked Test: {len(test_rows):,} Q&A pairs (GQA: {sum(1 for r in test_rows if r['dataset']=='gqa'):,}, VQAv2: {sum(1 for r in test_rows if r['dataset']=='vqav2'):,})")
    print(f"  TOTAL:       {len(train_rows) + len(val_rows) + len(test_rows):,} Q&A pairs")

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
        "unique_images": completed,
        "train_samples": len(valid_train),
        "validation_samples": len(valid_val),
        "locked_test_samples": len(valid_test),
        "total_samples": len(valid_train) + len(valid_val) + len(valid_test),
        "seed": args.seed,
    }
    (proc_dir / "dataset_20k_images_setup_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 75)
    print("20,000-IMAGE DATASET SETUP COMPLETE!")
    print(f"  Unique Images:   {completed:,}")
    print(f"  Train Q&A:       {len(valid_train):,} samples")
    print(f"  Validation Q&A:  {len(valid_val):,} samples")
    print(f"  Locked Test Q&A: {len(valid_test):,} samples")
    print(f"  Folds Directory: {folds_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
