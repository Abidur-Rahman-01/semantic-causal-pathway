"""Convert locally downloaded VQAv2 and GQA annotations into image-linked JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_vqa(root: Path, out: Path, split: str) -> int:
    qfile = root / f"v2_OpenEnded_mscoco_{split}2014_questions.json"
    afile = root / f"v2_mscoco_{split}2014_annotations.json"
    if not qfile.exists() or not afile.exists():
        raise FileNotFoundError(f"Expected official VQAv2 {split} JSON files in {root}")
    questions = {x["question_id"]: x for x in json.loads(qfile.read_text(encoding="utf-8"))["questions"]}
    anns = json.loads(afile.read_text(encoding="utf-8"))["annotations"]
    rows = []
    for ann in anns:
        q = questions[ann["question_id"]]
        image = root / f"{split}2014" / f"COCO_{split}2014_{q['image_id']:012d}.jpg"
        if not image.is_file():
            continue
        answers = [x["answer"].strip() for x in ann.get("answers", []) if x.get("answer", "").strip()]
        rows.append({"sample_id": f"vqav2-{q['question_id']}", "dataset": "vqav2", "split": split,
                     "image": str(image.resolve()), "image_id": str(q["image_id"]),
                     "question": q["question"], "answers": answers or [ann.get("multiple_choice_answer", "")],
                     "answer": ann.get("multiple_choice_answer", "")})
    dump_jsonl(out, rows)
    return len(rows)


def convert_gqa(root: Path, out: Path, split: str) -> int:
    source = root / f"{split}_balanced_questions.json"
    if not source.exists():
        raise FileNotFoundError(f"Expected official GQA file: {source}")
    raw = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for qid, item in raw.items():
        image_id = str(item["imageId"])
        image = root / "images" / f"{image_id}.jpg"
        if not image.is_file():
            continue
        rows.append({"sample_id": f"gqa-{qid}", "dataset": "gqa", "split": split,
                     "image": str(image.resolve()), "image_id": image_id,
                     "question": item["question"],
                     "answers": [str(item["answer"])], "answer": str(item["answer"])})
    dump_jsonl(out, rows)
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="dataset", required=True)
    v = sub.add_parser("vqav2"); v.add_argument("--root", type=Path, required=True); v.add_argument("--split", choices=("train", "val"), default="train"); v.add_argument("--output", type=Path, required=True)
    g = sub.add_parser("gqa"); g.add_argument("--root", type=Path, required=True); g.add_argument("--split", choices=("train", "val"), default="train"); g.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    n = convert_vqa(a.root, a.output, a.split) if a.dataset == "vqav2" else convert_gqa(a.root, a.output, a.split)
    print(f"Wrote {n:,} image-linked rows to {a.output}")


if __name__ == "__main__":
    main()
