"""Manifests for R-FCCT: contrastive tracing, held-out calibration, POPE test.

Three disjoint image roles are enforced:

* tracing images  - the 500 FCCT tracing images (``data/fcct/trace.jsonl``),
* calibration     - new COCO val2014 images, used only to choose injection
                    strength and the threshold controls,
* test            - the 500 original COCO POPE images, never used for any choice.

Absent-object negatives follow the POPE sampling strategies (random, popular,
adversarial). Category frequency and co-occurrence statistics are computed on
COCO val2014 images that are *not* POPE test images.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from fcct.io import read_jsonl, sha256, write_json, write_jsonl

POPE_QUESTION = "Is there a {object} in the image?"
TRACE_QUESTION = "Is there a {object} in the image? Please answer this question with one word (Yes or No)."
STRATEGIES = ("adversarial", "popular", "random")


def load_coco(annotation_path: Path):
    coco = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    categories = {r["id"]: r["name"] for r in coco["categories"]}
    images = {r["id"]: r for r in coco["images"]}
    present = defaultdict(set)      # any annotation, including crowd regions
    grounded = defaultdict(set)     # non-crowd, positive-area boxes
    for row in coco["annotations"]:
        present[row["image_id"]].add(row["category_id"])
        if not row.get("iscrowd", 0) and row["bbox"][2] > 0 and row["bbox"][3] > 0:
            grounded[row["image_id"]].add(row["category_id"])
    return categories, images, present, grounded


def label_statistics(present, exclude_images):
    """Image frequency and co-occurrence counts over non-test images."""
    frequency, cooccurrence = Counter(), defaultdict(Counter)
    for image_id, cats in present.items():
        if image_id in exclude_images:
            continue
        for a in cats:
            frequency[a] += 1
            for b in cats:
                if a != b:
                    cooccurrence[a][b] += 1
    return frequency, cooccurrence


class NegativeSampler:
    """POPE-style absent-object sampling with deterministic tie-breaking."""

    def __init__(self, categories, frequency, cooccurrence, seed):
        self.categories = sorted(categories)
        self.frequency, self.cooccurrence = frequency, cooccurrence
        self.rng = random.Random(seed)

    def sample(self, strategy, present, used):
        absent = [c for c in self.categories if c not in present and c not in used]
        if not absent:
            raise ValueError("No absent category available")
        if strategy == "random":
            return self.rng.choice(absent)
        if strategy == "popular":
            return max(absent, key=lambda c: (self.frequency[c], -c))
        if strategy == "adversarial":
            return max(absent, key=lambda c: (sum(self.cooccurrence[g][c] for g in present),
                                              self.frequency[c], -c))
        raise ValueError(f"Unknown strategy: {strategy}")


def _ensure_images(root: Path, images, image_ids, download_images):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    def one(image_id):
        meta = images[image_id]
        destination = root / "images" / meta["file_name"]
        if not destination.exists() and download_images:
            from scripts.setup_fcct import download
            url = meta["coco_url"].replace("http://images.cocodataset.org/",
                                           "https://s3.amazonaws.com/images.cocodataset.org/", 1)
            download(url, destination)
        if not destination.exists():
            return image_id, None
        with Image.open(destination) as image:
            if image.size != (meta["width"], meta["height"]):
                raise ValueError(f"Image dimensions differ from COCO metadata: {destination}")
            image.verify()
        return image_id, sha256(destination)

    hashes, missing = {}, []
    with ThreadPoolExecutor(max_workers=8 if download_images else 1) as pool:
        for image_id, digest in pool.map(one, sorted(image_ids)):
            if digest is None:
                missing.append(images[image_id]["file_name"])
            else:
                hashes[image_id] = digest
    if missing:
        raise ValueError(f"Missing {len(missing)} images; rerun with --download-images")
    return hashes


def prepare(rconfig: dict, base: dict, repo: Path, download_images=False):
    """Write contrastive-trace, calibration and test manifests; return a data card."""
    fcct_root = repo / base["data_root"]
    root = fcct_root  # images are shared with the FCCT reconstruction
    out = repo / rconfig["data_root"]
    categories, images, present, grounded = load_coco(fcct_root / "sources/instances_val2014.json")
    trace_rows = read_jsonl(fcct_root / "trace.jsonl")
    pope_rows = read_jsonl(fcct_root / "pope.jsonl")
    pope_images = {r["image_id"] for r in pope_rows}
    trace_images = {r["image_id"] for r in trace_rows}
    if pope_images & trace_images:
        raise ValueError("FCCT tracing and POPE images overlap")
    frequency, cooccurrence = label_statistics(present, pope_images)
    seed = rconfig["seed"]

    # 1. Contrastive tracing: every FCCT present-object question plus one
    #    absent-object question per present one on the same image.
    sampler = NegativeSampler(categories, frequency, cooccurrence, seed)
    contrastive, by_image = [], defaultdict(list)
    for row in trace_rows:
        by_image[row["image_id"]].append(row)
    for image_id in sorted(by_image):
        rows = by_image[image_id]
        used = set()
        for row in rows:
            contrastive.append({**row, "label": "yes", "strategy": "present"})
        for j in range(len(rows) * rconfig["trace"]["negatives_per_positive"]):
            strategy = STRATEGIES[j % len(STRATEGIES)]
            category = sampler.sample(strategy, present[image_id], used)
            used.add(category)
            name = categories[category]
            contrastive.append({"sample_id": f"rtrace-{image_id}-neg-{category}", "image_id": image_id,
                                "image": rows[0]["image"], "image_sha256": rows[0]["image_sha256"],
                                "object": name, "question": TRACE_QUESTION.format(object=name),
                                "label": "no", "strategy": strategy})

    # 2. Held-out calibration images: not POPE, not tracing.
    calib_cfg = rconfig["calibration_set"]
    eligible = sorted(i for i in grounded if i not in pope_images and i not in trace_images
                      and len(grounded[i]) >= calib_cfg["positives"])
    random.Random(seed + 1).shuffle(eligible)
    selected = eligible[:calib_cfg["images"]]
    if len(selected) != calib_cfg["images"]:
        raise ValueError("Insufficient eligible calibration images")
    hashes = _ensure_images(root, images, selected, download_images)
    calib_sampler = NegativeSampler(categories, frequency, cooccurrence, seed + 2)
    suffix = rconfig["pope_prompt_suffix"]
    calibration = []
    pick = random.Random(seed + 3)
    for image_id in selected:
        file = f"images/{images[image_id]['file_name']}"
        positives = pick.sample(sorted(grounded[image_id]), calib_cfg["positives"])
        used = set()
        questions = [(c, "yes", "present") for c in positives]
        for strategy in calib_cfg["negatives"]:
            category = calib_sampler.sample(strategy, present[image_id], used)
            used.add(category)
            questions.append((category, "no", strategy))
        for category, label, strategy in questions:
            name = categories[category]
            calibration.append({"sample_id": f"calib-{image_id}-{category}", "image_id": image_id,
                                "image": file, "image_sha256": hashes[image_id], "object": name,
                                "question": POPE_QUESTION.format(object=name) + suffix,
                                "label": label, "strategy": strategy})

    # 3. Test: the original POPE questions, with the evaluation prompt suffix.
    test = [{**r, "pope_text": r["question"], "question": r["question"] + suffix} for r in pope_rows]

    roles = [{r["image_id"] for r in rows} for rows in (contrastive, calibration, test)]
    overlaps = {"trace_calibration": len(roles[0] & roles[1]), "trace_test": len(roles[0] & roles[2]),
                "calibration_test": len(roles[1] & roles[2])}
    if any(overlaps.values()):
        raise ValueError(f"Image roles overlap: {overlaps}")
    for name, rows in (("contrastive_trace", contrastive), ("calibration", calibration), ("test_pope", test)):
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate sample IDs in {name}")
        write_jsonl(out / f"{name}.jsonl", rows)
    card = {"seed": seed, "overlaps": overlaps,
            "contrastive_trace": {"questions": len(contrastive), "images": len(roles[0]),
                                  "labels": dict(Counter(r["label"] for r in contrastive)),
                                  "strategies": dict(Counter(r["strategy"] for r in contrastive))},
            "calibration": {"questions": len(calibration), "images": len(roles[1]),
                            "labels": dict(Counter(r["label"] for r in calibration))},
            "test_pope": {"questions": len(test), "images": len(roles[2])},
            "pope_prompt_suffix": suffix,
            "label_statistics": "category frequency/co-occurrence over COCO val2014 excluding POPE images",
            "sha256": {n: sha256(out / f"{n}.jsonl") for n in ("contrastive_trace", "calibration", "test_pope")}}
    write_json(out / "data_card.json", card)
    return card
