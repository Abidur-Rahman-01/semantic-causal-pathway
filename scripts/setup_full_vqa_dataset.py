"""Download official VQAv2/GQA sources, retain referenced images, and build manifests.

Archives are temporary: this extracts only images referenced by VQAv2 train/val
and GQA balanced train/val question files, then removes the archives by default.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

VQA_BASE = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/"
COCO_BASE = "https://images.cocodataset.org/zips/"
GQA_BASE = "https://downloads.cs.stanford.edu/nlp/data/gqa/"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(8):
        offset = part.stat().st_size if part.exists() else 0
        print(f"Downloading {dest.name} (resume offset: {offset:,} bytes)", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "vqa-full-dataset-setup/1.0",
                                                    **({"Range": f"bytes={offset}-"} if offset else {})})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                append = offset > 0 and response.status == 206
                if offset and not append:
                    offset = 0
                received = offset if append else 0
                with part.open("ab" if append else "wb") as f:
                    while chunk := response.read(1024 * 1024):
                        f.write(chunk)
                        received += len(chunk)
                        if received and received % (1024 ** 3) < len(chunk):
                            print(f"  {dest.name}: {received / (1024 ** 3):.1f} GiB received", flush=True)
            part.replace(dest)
            return
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            if attempt == 7:
                raise RuntimeError(f"Download failed for {url}: {exc}") from exc
            print(f"Download interrupted; retry {attempt + 1}/7 for {dest.name}", flush=True)
            time.sleep(min(2 ** attempt, 60))


def read_zip_json(archive: Path, suffix: str) -> dict:
    with zipfile.ZipFile(archive) as zf:
        matches = [n for n in zf.namelist() if n.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one {suffix} in {archive}; found {matches[:5]}")
        with zf.open(matches[0]) as f:
            return json.load(f)


def extract_selected(archive: Path, target: Path, wanted: set[str]) -> tuple[int, list[str]]:
    target.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            if name not in wanted:
                continue
            out = target / name
            if not out.exists() or out.stat().st_size != member.file_size:
                with zf.open(member) as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            found.add(name)
    missing = sorted(wanted - found)
    return len(found), missing


def run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(Path(__file__).with_name(script)), *args], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/raw"), help="Raw dataset root (default: data/raw)")
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--keep-archives", action="store_true", help="Keep the full downloaded zip archives after extraction")
    p.add_argument("--skip-folds", action="store_true", help="Do not generate five-fold CV manifests")
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    if not 0 < args.test_fraction < 1:
        p.error("--test-fraction must be between 0 and 1")

    vqa = args.root / "vqav2"
    gqa = args.root / "gqa"
    archives = args.root / "archives"
    vqa.mkdir(parents=True, exist_ok=True)
    gqa.mkdir(parents=True, exist_ok=True)
    archives.mkdir(parents=True, exist_ok=True)

    # Annotation archives are small and provide the authoritative required image IDs.
    vqa_archives = {}
    for split, title in (("train", "Train"), ("val", "Val")):
        for kind, filename in (("questions", f"v2_Questions_{title}_mscoco.zip"),
                               ("annotations", f"v2_Annotations_{title}_mscoco.zip")):
            path = archives / filename
            if not path.exists():
                download(VQA_BASE + filename, path)
            vqa_archives[(split, kind)] = path
            with zipfile.ZipFile(path) as zf:
                zf.extractall(vqa)

    vqa_ids: dict[str, set[int]] = {}
    for split in ("train", "val"):
        questions = read_zip_json(vqa_archives[(split, "questions")], ".json")["questions"]
        vqa_ids[split] = {int(q["image_id"]) for q in questions}
        print(f"VQAv2 {split}: {len(questions):,} questions, {len(vqa_ids[split]):,} referenced images")

    # GQA's official question archive contains the balanced train/validation JSONs.
    gqa_questions_zip = archives / "questions1.2.zip"
    if not gqa_questions_zip.exists():
        download(GQA_BASE + gqa_questions_zip.name, gqa_questions_zip)
    with zipfile.ZipFile(gqa_questions_zip) as zf:
        selected_jsons = [n for n in zf.namelist() if Path(n).name in
                          {"train_balanced_questions.json", "val_balanced_questions.json"}]
        if len(selected_jsons) != 2:
            raise ValueError(f"Expected balanced train and validation question files in GQA archive; found {selected_jsons}")
        zf.extractall(gqa, members=selected_jsons)
    gqa_ids: dict[str, set[str]] = {}
    for split in ("train", "val"):
        qpath = gqa / f"{split}_balanced_questions.json"
        questions = json.loads(qpath.read_text(encoding="utf-8"))
        gqa_ids[split] = {str(row["imageId"]) for row in questions.values()}
        print(f"GQA Balanced {split}: {len(questions):,} questions, {len(gqa_ids[split]):,} referenced images")

    # Full COCO and GQA archives provide reliable source files; only referenced images are retained.
    missing_images: dict[str, list[str]] = {}
    for split in ("train", "val"):
        coco_split = f"{split}2014"
        archive = archives / f"{coco_split}.zip"
        if not archive.exists():
            download(COCO_BASE + archive.name, archive)
        wanted = {f"COCO_{coco_split}_{image_id:012d}.jpg" for image_id in vqa_ids[split]}
        count, missing = extract_selected(archive, vqa / coco_split, wanted)
        missing_images[f"vqav2_{split}"] = missing
        print(f"Extracted {count:,}/{len(wanted):,} required COCO {coco_split} images")

    gqa_archive = archives / "images.zip"
    if not gqa_archive.exists():
        download(GQA_BASE + gqa_archive.name, gqa_archive)
    wanted_gqa = {f"{image_id}.jpg" for ids in gqa_ids.values() for image_id in ids}
    count, missing = extract_selected(gqa_archive, gqa / "images", wanted_gqa)
    missing_images["gqa"] = missing
    print(f"Extracted {count:,}/{len(wanted_gqa):,} required GQA images")

    args.processed.mkdir(parents=True, exist_ok=True)
    report = {"missing_images": {key: len(value) for key, value in missing_images.items()},
              "missing_image_examples": {key: value[:20] for key, value in missing_images.items()},
              "raw_root": str(args.root.resolve()), "processed_root": str(args.processed.resolve()),
              "archives_retained": args.keep_archives}
    report_path = args.processed / "dataset_setup_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if any(missing_images.values()):
        raise SystemExit("Some archive members were missing; partial manifests were not built. See dataset_setup_report.json")

    # Converters retain every question only when its referenced image exists.
    for dataset in ("vqav2", "gqa"):
        for split in ("train", "val"):
            run("convert_vqa_datasets.py", dataset, "--root", str((vqa if dataset == "vqav2" else gqa).resolve()),
                "--split", split, "--output", str((args.processed / f"{dataset}_{split}.jsonl").resolve()))
    run("build_vqa_manifests.py", "--data-dir", str(args.processed.resolve()),
        "--test-fraction", str(args.test_fraction), "--seed", str(args.seed))
    if not args.skip_folds:
        run("build_vqa_folds.py", "--input", str((args.processed / "train.jsonl").resolve()),
            "--output-dir", str((args.processed / "folds").resolve()), "--folds", "5", "--seed", str(args.seed))

    if not args.keep_archives:
        for path in archives.glob("*.zip"):
            path.unlink()
        try:
            archives.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
