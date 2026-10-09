import json
import urllib.request
from pathlib import Path
import time

questions_path = Path("data/raw/gqa/val_balanced_questions.json")
raw = json.loads(questions_path.read_text(encoding="utf-8"))

img_dir = Path("data/raw/gqa/images")
img_dir.mkdir(parents=True, exist_ok=True)

# Select a high-quality set of unique image IDs
needed_images = []
seen = set()
for qid, item in raw.items():
    img_id = str(item["imageId"])
    if img_id not in seen:
        seen.add(img_id)
        needed_images.append(img_id)
    if len(needed_images) >= 50:
        break

print(f"Targeting {len(needed_images)} GQA images...")

prefixes = [
    "https://cs.stanford.edu/people/rak248/VG_100K/",
    "https://cs.stanford.edu/people/rak248/VG_100K_2/"
]

success = 0
for idx, img_id in enumerate(needed_images, 1):
    dest = img_dir / f"{img_id}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        success += 1
        continue
    
    downloaded = False
    for pfx in prefixes:
        url = f"{pfx}{img_id}.jpg"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                if len(data) > 0:
                    dest.write_bytes(data)
                    downloaded = True
                    success += 1
                    break
        except Exception:
            pass
    if idx % 10 == 0:
        print(f"Progress: {idx}/{len(needed_images)} (downloaded {success})")

print(f"Successfully downloaded {success}/{len(needed_images)} images to {img_dir}")
