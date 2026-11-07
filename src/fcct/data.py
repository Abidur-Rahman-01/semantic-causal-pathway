"""COCO-grounded tracing questions and original POPE benchmark manifests."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .io import read_jsonl, sha256, write_json, write_jsonl

QUESTION = "Is there a {object} in the image? Please answer this question with one word (Yes or No)."


def validate_manifest(rows, root: Path, kind: str):
    if not rows:
        raise ValueError("Manifest is empty")
    seen = set()
    for row in rows:
        if row["sample_id"] in seen:
            raise ValueError(f"Duplicate sample ID: {row['sample_id']}")
        seen.add(row["sample_id"])
        path = root / row["image"]
        if not path.is_file() or sha256(path) != row["image_sha256"]:
            raise ValueError(f"Image missing or checksum mismatch: {path}")
        with Image.open(path) as image:
            image.verify()
        if not row["question"].strip():
            raise ValueError("Empty question")
        if kind == "trace":
            if row["object"] not in row["question"] or not row["boxes"]:
                raise ValueError("Tracing needs a queried object and grounded boxes")
        elif kind == "pope" and row["label"] not in ("yes", "no"):
            raise ValueError("POPE label must be yes or no")


def prepare(config: dict, repo: Path, download_images=False):
    root = repo / config["data_root"]
    annotation = root / "sources/instances_val2014.json"
    coco = json.loads(annotation.read_text(encoding="utf-8"))
    categories = {r["id"]: r["name"] for r in coco["categories"]}
    images = {r["id"]: r for r in coco["images"]}
    boxes = defaultdict(lambda: defaultdict(list))
    for row in coco["annotations"]:
        if row.get("iscrowd", 0) or row["bbox"][2] <= 0 or row["bbox"][3] <= 0:
            continue
        boxes[row["image_id"]][row["category_id"]].append(row["bbox"])
    pope, pope_images = [], set()
    for split in ("random", "popular", "adversarial"):
        source = root / "sources" / f"coco_pope_{split}.json"
        items = read_jsonl(source)
        if len(items) != 3000:
            raise ValueError(f"Expected original 3,000-question POPE split: {source}")
        for row in items:
            image_id = int(Path(row["image"]).stem.split("_")[-1])
            if image_id not in images:
                raise ValueError(f"POPE image absent from COCO annotations: {image_id}")
            pope_images.add(image_id)
            pope.append({"sample_id": f"pope-{split}-{row['question_id']}", "image_id": image_id,
                         "image": f"images/{images[image_id]['file_name']}", "question": row["text"],
                         "label": row["label"], "split": split, "question_id": row["question_id"]})
    eligible = sorted(set(boxes) - pope_images)
    random.Random(config["seed"]).shuffle(eligible)
    selected = eligible[:config["trace_images"]]
    if len(selected) != config["trace_images"]:
        raise ValueError("Insufficient disjoint tracing images")
    trace = []
    for image_id in selected:
        category_ids = sorted(boxes[image_id])
        count = config.get("questions_per_image")
        if count is not None:
            category_ids = category_ids[:count]
        for category_id in category_ids:
            name = categories[category_id]
            trace.append({"sample_id": f"trace-{image_id}-{category_id}", "image_id": image_id,
                          "image": f"images/{images[image_id]['file_name']}",
                          "object": name, "boxes": sorted(boxes[image_id][category_id]),
                          "question": QUESTION.format(object=name), "selection": "independent_reconstruction"})
    required_images = sorted(set(selected) | pope_images)
    missing, image_hashes = [], {}
    def prepare_image(image_id):
        filename = images[image_id]["file_name"]
        destination = root / "images" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() and download_images:
            # The official COCO metadata supplies the original URL.
            from scripts.setup_fcct import download
            url = images[image_id]["coco_url"]
            if url.startswith("http://images.cocodataset.org/"):
                url = url.replace("http://images.cocodataset.org/",
                                  "https://s3.amazonaws.com/images.cocodataset.org/", 1)
            download(url, destination)
        if not destination.exists():
            return image_id, filename, None
        with Image.open(destination) as image:
            if image.size != (images[image_id]["width"], images[image_id]["height"]):
                raise ValueError(f"Image dimensions differ from COCO metadata: {destination}")
            image.verify()
        return image_id, filename, sha256(destination)

    # Image fetches are independent. Ordered consumption keeps manifests and
    # selection deterministic regardless of network completion order.
    with ThreadPoolExecutor(max_workers=8 if download_images else 1) as workers:
        for index, (image_id, filename, digest) in enumerate(workers.map(prepare_image, required_images), 1):
            if digest is None:
                missing.append(filename)
            else:
                image_hashes[image_id] = digest
            if index % 100 == 0:
                print(f"Prepared images {index}/{len(required_images)}", flush=True)
    if missing:
        write_json(root / "missing_images.json", missing)
        raise ValueError(f"Missing {len(missing)} images; rerun prepare --download-images. No manifests written.")
    for row in trace + pope:
        row["image_sha256"] = image_hashes[row["image_id"]]
    validate_manifest(trace, root, "trace")
    validate_manifest(pope, root, "pope")
    write_jsonl(root / "trace.jsonl", trace)
    write_jsonl(root / "pope.jsonl", pope)
    card = {"status": "independent_reconstruction_not_author_subset", "seed": config["seed"],
            "tracing_images": len(selected), "tracing_questions": len(trace),
            "pope_images": len(pope_images), "pope_questions": len(pope),
            "overlap_images": len(set(selected) & pope_images),
            "annotation_sha256": sha256(annotation), "trace_sha256": sha256(root / "trace.jsonl"),
            "pope_sha256": sha256(root / "pope.jsonl"),
            "selection": "seeded shuffle of sorted annotated val2014 images excluding all original POPE images; up to first 6 distinct category IDs",
            "licenses": {"annotations": "COCO terms; https://cocodataset.org/#termsofuse",
                         "images": "Original Flickr licenses in COCO images metadata; redistribution requires respecting each license",
                         "pope": "https://github.com/AoiDragon/POPE/blob/main/LICENSE"}}
    write_json(root / "dataset_card.json", card)
    return card
