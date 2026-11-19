"""Benchmark manifests: POPE, MME, CHAIR images, MMHal-Bench."""
from __future__ import annotations

import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from fcct.io import read_jsonl, sha256, write_json, write_jsonl

MME_HALLUCINATION = ("existence", "count", "position", "color")
MME_PERCEPTION = ("existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark",
                  "artwork", "OCR")
MME_COGNITION = ("commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning")


def prompt_key(image_sha, question):
    return hashlib.sha256(json.dumps([image_sha, question]).encode()).hexdigest()[:24]


def prepare_pope(cfg, base, repo):
    rows = read_jsonl(repo / base["data_root"] / "pope.jsonl")
    if len(rows) != cfg["pope"].get("expected_rows", 9000):
        raise ValueError("Expected the 9,000-question original COCO POPE manifest (run fcct.py prepare)")
    out = []
    for r in rows:
        question = r["question"] + cfg["pope"]["prompt_suffix"]
        out.append({**r, "pope_text": r["question"], "question": question,
                    "key": prompt_key(r["image_sha256"], question)})
    return out


def prepare_mme(cfg, repo, root):
    import pyarrow.parquet as pq
    files = sorted((root / "sources/mme").glob("*.parquet"))
    if len(files) != 4:
        raise ValueError("MME parquet files missing; run scripts/setup_bench.py --mme")
    images_dir = root / "mme/images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows, index = [], 0
    for file in files:
        table = pq.read_table(file, columns=["question_id", "image", "question", "answer", "category"])
        for r in table.to_pylist():
            data = r["image"]["bytes"]
            digest = hashlib.sha256(data).hexdigest()
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
            path = images_dir / f"{digest[:32]}.img"
            if not path.exists():
                path.write_bytes(data)
            answer = r["answer"].strip().lower().replace(".", "")
            if answer not in ("yes", "no"):
                raise ValueError(f"Unexpected MME answer {r['answer']!r}")
            if r["category"] not in MME_PERCEPTION + MME_COGNITION:
                raise ValueError(f"Unknown MME category {r['category']!r}")
            rows.append({"key": f"mme-{index:05d}", "category": r["category"], "question_id": r["question_id"],
                         "image": str(path.relative_to(root)), "image_sha256": digest,
                         "question": r["question"], "label": answer})
            index += 1
    pairs = Counter((r["category"], r["question_id"]) for r in rows)
    if set(pairs.values()) != {2}:
        raise ValueError("MME requires exactly two questions per image")
    return rows


def prepare_chair(cfg, base, repo, root, download_images):
    from rfcct.data import _ensure_images, load_coco
    categories, images, present, grounded = load_coco(repo / base["data_root"] / "sources/instances_val2014.json")
    exclude = {r["image_id"] for r in read_jsonl(repo / base["data_root"] / "trace.jsonl")}
    calibration = repo / "data/rfcct/calibration.jsonl"
    if calibration.is_file():
        exclude |= {r["image_id"] for r in read_jsonl(calibration)}
    eligible = sorted(i for i in images if i not in exclude)
    selected = sorted(random.Random(cfg["chair"]["seed"]).sample(eligible, cfg["chair"]["images"]))
    hashes = _ensure_images(repo / base["data_root"], images, selected, download_images)
    return [{"key": f"chair-{i}", "image_id": i, "image": f"images/{images[i]['file_name']}",
             "image_sha256": hashes[i], "question": cfg["chair"]["prompt"]} for i in selected]


def prepare_mmhal(cfg, root):
    template = json.loads((root / "sources/mmhal/response_template.json").read_text(encoding="utf-8"))
    if len(template) != 96:
        raise ValueError("MMHal-Bench must contain 96 items")
    rows = []
    for i, item in enumerate(template):
        name = item["image_src"].rsplit("/", 1)[-1]
        path = root / "sources/mmhal/images" / name
        if not path.is_file():
            raise ValueError(f"MMHal image missing: {name}; run scripts/setup_bench.py --mmhal")
        rows.append({"key": f"mmhal-{i:02d}", "index": i, "image": str(path.relative_to(root)),
                     "image_sha256": sha256(path), "question": item["question"],
                     "question_type": item["question_type"]})
    return rows


def prepare(cfg, base, repo, benchmarks, download_images=False):
    root = repo / cfg["data_root"]
    card = {}
    for name in benchmarks:
        if name == "pope":
            rows = prepare_pope(cfg, base, repo)
        elif name == "mme":
            rows = prepare_mme(cfg, repo, root)
        elif name == "chair":
            rows = prepare_chair(cfg, base, repo, root, download_images)
        elif name == "mmhal":
            rows = prepare_mmhal(cfg, root)
        else:
            raise ValueError(f"Unknown benchmark {name}")
        write_jsonl(root / f"{name}.jsonl", rows)
        card[name] = {"rows": len(rows), "unique_prompts": len({r["key"] for r in rows}),
                      "sha256": sha256(root / f"{name}.jsonl")}
    write_json(root / "data_card.json", {**(json.loads((root / "data_card.json").read_text())
                                           if (root / "data_card.json").is_file() else {}), **card})
    return card


def image_root(cfg, base, repo, benchmark):
    """POPE/CHAIR images live with the FCCT data; MME/MMHal under data/bench."""
    return repo / (base["data_root"] if benchmark in ("pope", "chair") else cfg["data_root"])


def load_image(path):
    with Image.open(path) as source:
        return source.convert("RGB")
