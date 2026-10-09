import json
import random
from pathlib import Path

random.seed(42)

vqav2_path = Path("data/processed/vqav2_val.jsonl")
gqa_path = Path("data/processed/gqa_val.jsonl")

vqa_rows = [json.loads(line) for line in vqav2_path.read_text(encoding="utf-8").splitlines() if line.strip()]
gqa_rows = [json.loads(line) for line in gqa_path.read_text(encoding="utf-8").splitlines() if line.strip()]

print(f"Loaded {len(vqa_rows)} VQAv2 rows, {len(gqa_rows)} GQA rows")

# Group by image to prevent leakage
def group_by_image(rows):
    by_img = {}
    for r in rows:
        by_img.setdefault(r["image_id"], []).append(r)
    return by_img

vqa_by_img = group_by_image(vqa_rows)
gqa_by_img = group_by_image(gqa_rows)

vqa_img_ids = list(vqa_by_img.keys())
gqa_img_ids = list(gqa_by_img.keys())
random.shuffle(vqa_img_ids)
random.shuffle(gqa_img_ids)

# Partition VQAv2 (10 images total): 6 train, 2 val, 2 test
vqa_train_imgs = vqa_img_ids[:6]
vqa_val_imgs = vqa_img_ids[6:8]
vqa_test_imgs = vqa_img_ids[8:]

# Partition GQA (50 images total): 30 train, 10 val, 10 test
gqa_train_imgs = gqa_img_ids[:30]
gqa_val_imgs = gqa_img_ids[30:40]
gqa_test_imgs = gqa_img_ids[40:]

def collect(img_ids, by_img):
    out = []
    for i in img_ids:
        out.extend(by_img[i])
    return out

train_rows = collect(vqa_train_imgs, vqa_by_img) + collect(gqa_train_imgs, gqa_by_img)
val_rows = collect(vqa_val_imgs, vqa_by_img) + collect(gqa_val_imgs, gqa_by_img)
test_rows = collect(vqa_test_imgs, vqa_by_img) + collect(gqa_test_imgs, gqa_by_img)

random.shuffle(train_rows)
random.shuffle(val_rows)
random.shuffle(test_rows)

for name, rows in [("train", train_rows), ("validation", val_rows), ("locked_test", test_rows)]:
    out_file = Path(f"data/processed/{name}.jsonl")
    with out_file.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} samples to {out_file} (VQAv2: {sum(1 for r in rows if r['dataset']=='vqav2')}, GQA: {sum(1 for r in rows if r['dataset']=='gqa')})")
