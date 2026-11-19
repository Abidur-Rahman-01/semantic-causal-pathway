"""Fetch pinned assets for the paper evaluation matrix (MME, CHAIR, MMHal-Bench).

Every remote file is fetched at an immutable revision and checked against a
SHA-256 (or the ZIP member CRC for COCO annotations). Third-party evaluation
code (OPERA chair.py, MMHal eval_gpt4.py) is fetched at a pinned commit and
verified byte-for-byte before use.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fcct.io import sha256, write_json  # noqa: E402
from setup_fcct import RemoteZip, download  # noqa: E402

MME_REPO, MME_REVISION = "lmms-lab-encoder/MME", "d6c9023f017b564f7b3ccccf5348166bce8fdbcd"
MME_FILES = {  # file -> SHA-256 at the pinned revision (verified 2026-11-20)
    "test-00000-of-00004-a25dbe3b44c4fda6.parquet": "1563fd59c8049dbbabe211a049c38b2e20a6d58da94a75c90cfc937d1f6c52e3",
    "test-00001-of-00004-7d22c7f1aba6fca4.parquet": "3623728c0c98807cc7eef9cbd10dfa18c21654321c4b02b7b19bc8e6fc86d830",
    "test-00002-of-00004-594798fd3f5b029c.parquet": "c169d85bdf84e21b005170698217e96380bef172e8bd7432fc75ddce51dcaabe",
    "test-00003-of-00004-53ae1794f93b1e35.parquet": "aec84886b2bb6dd7bffc94eb3bbca9319e0671f3ee989a08e0758bc2de6c67f6"}
OPERA_COMMIT = "e1bb7632508bdfaa5cdaf5b69b64419688107da8"
MMHAL_REPO, MMHAL_REVISION = "Shengcao1006/MMHal-Bench", "f5f49a938f45ed99e235b8519ba28f76832a2add"
COCO_ZIP = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip"
COCO_MEMBERS = ("captions_train2014.json", "captions_val2014.json", "instances_train2014.json",
                "instances_val2014.json")
NLTK_PACKAGES = ("punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
                 "wordnet", "omw-1.4")
PINNED = {"chair.py": "84ea67bd831f7dc8ed605f5d3428cfb3e8f5dad3f4b73e1d2907a01c66a4af86",
          "eval_gpt4.py": "5ba718be65d03cb9e80321ad9eb08b37a0eb0040372e1fa20f78eaf525fe8f7c"}


def hf_url(repo, revision, path):
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


def checked(path: Path, expected):
    actual = sha256(path)
    if actual != expected:
        path.unlink()
        raise ValueError(f"Checksum mismatch for {path.name}: {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bench_llava15.json")
    parser.add_argument("--mme", action="store_true", help="MME parquet files (about 0.9 GB)")
    parser.add_argument("--chair", action="store_true", help="OPERA chair.py, COCO 2014 captions/instances, NLTK data")
    parser.add_argument("--mmhal", action="store_true", help="MMHal-Bench template, images, judge script")
    args = parser.parse_args()
    if not (args.mme or args.chair or args.mmhal):
        parser.error("Select --mme, --chair and/or --mmhal")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    root = ROOT / cfg["data_root"] / "sources"
    lock_path = root / "bench.lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}

    def record(key, path, **extra):
        entry = {"sha256": sha256(path), "bytes": path.stat().st_size, **extra}
        if key in lock and lock[key]["sha256"] != entry["sha256"]:
            raise ValueError(f"{key} differs from the existing lock")
        lock[key] = entry
        write_json(lock_path, lock)

    if args.mme:
        for name, digest in MME_FILES.items():
            path = root / "mme" / name
            print(f"Fetching MME {name}", flush=True)
            download(hf_url(MME_REPO, MME_REVISION, f"data/{name}"), path)
            checked(path, digest)
            record(f"mme/{name}", path, repo=MME_REPO, revision=MME_REVISION)
    if args.chair:
        path = root / "chair/chair.py"
        download(f"https://raw.githubusercontent.com/shikiw/OPERA/{OPERA_COMMIT}/chair.py", path)
        checked(path, PINNED["chair.py"])
        record("chair/chair.py", path, commit=OPERA_COMMIT)
        print("Fetching COCO 2014 caption/instance annotations (HTTP ranges)", flush=True)
        with RemoteZip(COCO_ZIP) as remote, zipfile.ZipFile(remote) as archive:
            for member in COCO_MEMBERS:
                target = root / "chair/annotations" / member
                if target.exists():
                    continue
                info = archive.getinfo(f"annotations/{member}")
                content = archive.read(info)  # ZIP CRC verified by zipfile
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".part")
                temporary.write_bytes(content)
                temporary.replace(target)
                record(f"chair/annotations/{member}", target, zip_crc32=info.CRC, url=COCO_ZIP)
                print(f"  {member}: {len(content)} bytes", flush=True)
        import nltk
        nltk_dir = ROOT / ".venv/nltk_data"
        for package in NLTK_PACKAGES:
            if not nltk.download(package, download_dir=str(nltk_dir), quiet=True):
                raise RuntimeError(f"NLTK download failed: {package}")
        lock["chair/nltk"] = {"packages": list(NLTK_PACKAGES), "nltk": nltk.__version__, "sha256": "n/a"}
        write_json(lock_path, lock)
    if args.mmhal:
        base = root / "mmhal"
        for name in ("response_template.json", "eval_gpt4.py", "test_data.zip"):
            download(hf_url(MMHAL_REPO, MMHAL_REVISION, name), base / name)
            if name in PINNED:
                checked(base / name, PINNED[name])
            record(f"mmhal/{name}", base / name, repo=MMHAL_REPO, revision=MMHAL_REVISION)
        template = json.loads((base / "response_template.json").read_text(encoding="utf-8"))
        wanted = {item["image_src"].rsplit("/", 1)[-1] for item in template}
        with zipfile.ZipFile(base / "test_data.zip") as archive:
            for name in sorted(wanted):  # extract only the expected flat file names
                if "/" in name or "\\" in name or name.startswith("."):
                    raise ValueError(f"Unsafe MMHal image name: {name}")
                target = base / "images" / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(f"images/{name}"))
        print(f"MMHal images: {len(wanted)}", flush=True)
    print(json.dumps({"lock": str(lock_path), "entries": sorted(lock)}, indent=2))


if __name__ == "__main__":
    main()
