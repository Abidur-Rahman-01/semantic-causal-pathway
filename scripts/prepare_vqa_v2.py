"""Download a reproducible VQA v2 validation pilot and create a raw manifest.

Images are fetched individually from the official COCO 2014 validation image host,
so a pilot does not need the full 6 GB image archive. The resulting manifest is
deliberately raw: evidence masks, variants, and their review flags remain unfilled.
"""
from __future__ import annotations

import argparse
import json
import random
import ssl
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

VQA = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/"
COCO = "https://images.cocodataset.org/val2014/COCO_val2014_{image_id:012d}.jpg"


def fetch(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(5):
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"User-Agent": "semantic-circuits-research/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                append = offset > 0 and response.status == 206
                if offset and not append:
                    offset = 0
                with temporary.open("ab" if append else "wb") as out:
                    while chunk := response.read(256 * 1024):
                        out.write(chunk)
                        out.flush()
            temporary.replace(destination)
            return
        except (OSError, TimeoutError) as exc:
            if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def read_zip_json(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".json")]
        if len(names) != 1:
            raise ValueError(f"Expected one JSON in {path}, found {names}")
        with archive.open(names[0]) as source:
            return json.load(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".venv/data/vqa_v2"))
    parser.add_argument("--limit-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--download-images", action="store_true")
    args = parser.parse_args()
    if args.limit_images < 1:
        parser.error("--limit-images must be positive")
    root = args.output.resolve()
    raw = root / "source"
    qzip, azip = raw / "v2_Questions_Val_mscoco.zip", raw / "v2_Annotations_Val_mscoco.zip"
    fetch(VQA + qzip.name, qzip)
    fetch(VQA + azip.name, azip)
    questions, annotations = read_zip_json(qzip), read_zip_json(azip)
    answer_by_qid = {int(a["question_id"]): a for a in annotations["annotations"]}
    question_by_id = {int(q["question_id"]): q for q in questions["questions"]}
    if question_by_id.keys() != answer_by_qid.keys():
        raise ValueError("Question and annotation question IDs do not match")
    by_image: dict[int, list[dict]] = {}
    for qid, question in question_by_id.items():
        by_image.setdefault(int(question["image_id"]), []).append({"question": question, "annotation": answer_by_qid[qid]})
    image_ids = sorted(by_image)
    random.Random(args.seed).shuffle(image_ids)
    chosen = image_ids[:args.limit_images]
    image_dir = root / "images" / "val2014"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_id in chosen:
        filename = f"COCO_val2014_{image_id:012d}.jpg"
        for item in by_image[image_id]:
            q, ann = item["question"], item["annotation"]
            answers = [entry["answer"].strip().lower() for entry in ann.get("answers", []) if entry.get("answer", "").strip()]
            qid = int(q["question_id"])
            rows.append({
                "sample_id": f"vqa2-val-{qid}", "image_id": image_id,
                "image": f"images/val2014/{filename}", "question": q["question"],
                # Candidate outcome bins are intentionally left for preregistration.
                # Deriving them from these gold answers would leak labels into the intervention.
                "answers": answers, "candidate_answers": [],
                "candidate_answers_protocol": "annotate_before_model_inference_without_consulting_answers",
                "question_type": ann.get("question_type"), "answer_type": ann.get("answer_type"),
                "split": "validation", "evidence_mask": None, "evidence_mask_reviewed": False,
                "variants": [], "dataset": "VQA v2", "source_question_id": qid,
            })
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "raw_validation_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    download_failures = []
    if args.download_images:
        for index, image_id in enumerate(chosen, 1):
            filename = f"COCO_val2014_{image_id:012d}.jpg"
            try:
                fetch(COCO.format(image_id=image_id), image_dir / filename)
            except (OSError, TimeoutError) as exc:
                download_failures.append({"image_id": image_id, "error": f"{type(exc).__name__}: {exc}"})
                download_failures.extend({"image_id": pending, "error": "not attempted after image-host connection failure"}
                                         for pending in chosen[index:])
                break
            if index % 25 == 0:
                print(f"Image download progress: {index}/{len(chosen)}")
    failures_path = root / "image_download_failures.json"
    if download_failures:
        failures_path.write_text(json.dumps(download_failures, indent=2), encoding="utf-8")
    elif args.download_images and failures_path.exists():
        failures_path.unlink()
    downloaded = sum((image_dir / f"COCO_val2014_{image_id:012d}.jpg").exists() for image_id in chosen)
    print(json.dumps({"dataset": "VQA v2 validation", "unique_images": len(chosen),
                      "questions": len(rows), "seed": args.seed, "images_downloaded": downloaded,
                      "image_download_failures": len(download_failures), "manifest": str(manifest)}, indent=2))
    if not args.download_images:
        print("Images were not fetched. Re-run with --download-images to download the selected COCO files.")


if __name__ == "__main__":
    main()
