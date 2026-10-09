"""Download targeted COCO images in parallel to accelerate dataset setup."""
from __future__ import annotations

import concurrent.futures
import json
import os
import random
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COCO_VAL_PREFIX = "http://images.cocodataset.org/val2014/"
COCO_TRAIN_PREFIX = "http://images.cocodataset.org/train2014/"


def download_coco_img(task: tuple[str, Path]) -> bool:
    url, dest = task
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".tmp")
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = resp.read()
                if len(data) > 1000:
                    temp.write_bytes(data)
                    temp.replace(dest)
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    rng = random.Random(17)
    vqa_train_q = json.loads((ROOT / "data/raw/vqav2/v2_OpenEnded_mscoco_train2014_questions.json").read_text(encoding="utf-8"))["questions"]
    vqa_val_q = json.loads((ROOT / "data/raw/vqav2/v2_OpenEnded_mscoco_val2014_questions.json").read_text(encoding="utf-8"))["questions"]

    train_ids = list({str(x["image_id"]) for x in vqa_train_q})
    val_ids = list({str(x["image_id"]) for x in vqa_val_q})

    train_dir = ROOT / "data/raw/vqav2/train2014"
    val_dir = ROOT / "data/raw/vqav2/val2014"
    train_existing = set(os.listdir(train_dir)) if train_dir.exists() else set()
    val_existing = set(os.listdir(val_dir)) if val_dir.exists() else set()

    # Prioritize existing as in setup script
    def select_keys(all_keys, target_n, existing_set, fmt):
        exist = [k for k in all_keys if fmt(k) in existing_set]
        missing = [k for k in all_keys if fmt(k) not in existing_set]
        rng.shuffle(exist)
        rng.shuffle(missing)
        if len(exist) >= target_n:
            return exist[:target_n]
        return exist + missing[:target_n - len(exist)]

    train_sel = select_keys(train_ids, 5600, train_existing, lambda k: f"COCO_train2014_{int(k):012d}.jpg")
    val_test_sel = select_keys(val_ids, 2400, val_existing, lambda k: f"COCO_val2014_{int(k):012d}.jpg")

    tasks = []
    for k in train_sel:
        fname = f"COCO_train2014_{int(k):012d}.jpg"
        dest = train_dir / fname
        if not (dest.exists() and dest.stat().st_size > 1000):
            tasks.append((f"{COCO_TRAIN_PREFIX}{fname}", dest))

    for k in val_test_sel:
        fname = f"COCO_val2014_{int(k):012d}.jpg"
        dest = val_dir / fname
        if not (dest.exists() and dest.stat().st_size > 1000):
            tasks.append((f"{COCO_VAL_PREFIX}{fname}", dest))

    print(f"Targeted COCO images to download: {len(tasks):,}", flush=True)
    if not tasks:
        print("All targeted COCO images already downloaded!", flush=True)
        return

    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
        futs = {executor.submit(download_coco_img, t): t for t in tasks}
        for fut in concurrent.futures.as_completed(futs):
            if fut.result():
                done += 1
            if done % 200 == 0 or done == len(tasks):
                el = max(time.time() - t0, 0.1)
                rate = done / el
                eta = (len(tasks) - done) / max(rate, 0.1) / 60
                print(f"COCO: {done:,}/{len(tasks):,} ({done*100/len(tasks):.1f}%) | {rate:.1f} imgs/s | ETA: {eta:.1f}m", flush=True)

    print(f"COCO download completed in {(time.time()-t0)/60:.1f} minutes! Total: {done:,}", flush=True)


if __name__ == "__main__":
    main()
