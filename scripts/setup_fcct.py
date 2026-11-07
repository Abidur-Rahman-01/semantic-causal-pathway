"""Fetch pinned public FCCT reconstruction assets; never alter prior study data."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import threading
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fcct.io import sha256, write_json

POPE_COMMIT = "08d957b917e5a378a2f99d35b6293c536a66298b"
POPE_BLOBS = {"random": "cd2b137f786675decb7442675929885125eeb083",
              "popular": "eecb09ee9c68becd95e6b18d48433ec68cc261b1",
              "adversarial": "4eb7fc57596b955756a94bc63d1f967c2afb692a"}
HTTP_SESSIONS = threading.local()


class RemoteZip(io.RawIOBase):
    """Seekable HTTP ranges: fetch one original ZIP member without the train set."""
    def __init__(self, url):
        import requests
        self.session = requests.Session()
        self.url, self.position = url, 0
        with self.session.get(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
                              stream=True, timeout=(30, 60)) as response:
            if response.status_code != 206:
                raise IOError("COCO host does not support HTTP Range; use --annotations")
            self.length = int(response.headers["Content-Range"].split("/")[-1])

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        new = offset + ({0: 0, 1: self.position, 2: self.length}[whence])
        if new < 0:
            raise ValueError("Negative remote ZIP position")
        self.position = new
        return new

    def read(self, size=-1):
        size = min(self.length - self.position, size if size >= 0 else self.length)
        if size <= 0:
            return b""
        end = self.position + size - 1
        with self.session.get(self.url, headers={"Range": f"bytes={self.position}-{end}",
                                                "Accept-Encoding": "identity"},
                              timeout=(30, 120)) as response:
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {self.position}-{end}/{self.length}":
                raise IOError("Invalid byte-range response from COCO host")
            data = response.content
        if len(data) != size:
            raise IOError("Truncated COCO byte range")
        self.position += size
        return data

    def close(self):
        self.session.close()
        super().close()


def download(url, destination: Path, *, git_blob=None):
    import requests
    if not hasattr(HTTP_SESSIONS, "session"):
        HTTP_SESSIONS.session = requests.Session()
        HTTP_SESSIONS.session.headers.update({"User-Agent": "fcct-independent-reproduction/0.1"})
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(total=3, connect=3, read=3, status=3, backoff_factor=0.5,
                      status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
        HTTP_SESSIONS.session.mount("https://", HTTPAdapter(max_retries=retry))
        HTTP_SESSIONS.session.mount("http://", HTTPAdapter(max_retries=retry))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(3):
            try:
                with HTTP_SESSIONS.session.get(url, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    expected = response.headers.get("Content-Length")
                    count = 0
                    with temporary.open("wb") as target:
                        for block in response.iter_content(2 * 1024 * 1024):
                            target.write(block)
                            count += len(block)
                    if expected and not response.headers.get("Content-Encoding") and count != int(expected):
                        raise requests.exceptions.ChunkedEncodingError(f"Truncated download: {destination.name}")
                break
            except requests.RequestException as error:
                if isinstance(error, requests.HTTPError) and error.response is not None and error.response.status_code < 429:
                    raise
                if attempt == 2:
                    raise
                print(f"Retrying interrupted transfer {destination.name} ({attempt + 1}/2)", flush=True)
                time.sleep(2 ** attempt)
        temporary.replace(destination)
    if git_blob:
        content = destination.read_bytes()
        actual = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if actual != git_blob:
            raise ValueError(f"Pinned upstream blob mismatch: {destination}")
    return {"url": url, "sha256": sha256(destination), "bytes": destination.stat().st_size,
            "git_blob": git_blob}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fcct_llava15.json")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--pope", action="store_true")
    parser.add_argument("--annotations", action="store_true")
    parser.add_argument("--annotations-fast", action="store_true", help="Fetch only val instances using HTTP byte ranges")
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--model-metadata", action="store_true", help="Fetch tokenizer/processor files without weights")
    args = parser.parse_args()
    if not any((args.paper, args.pope, args.annotations, args.annotations_fast, args.model, args.model_metadata)):
        parser.error("Select --paper, --pope, --annotations, --model, and/or --model-metadata")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = ROOT / config["data_root"]
    lock_path = root / "sources.lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    tasks = []
    if args.paper:
        tasks.append(("https://arxiv.org/pdf/2511.05923v3", root / "sources/2511.05923v3.pdf", None))
    if args.pope:
        for split, blob in POPE_BLOBS.items():
            filename = f"coco_pope_{split}.json"
            tasks.append((f"https://raw.githubusercontent.com/AoiDragon/POPE/{POPE_COMMIT}/output/coco/{filename}",
                          root / "sources" / filename, blob))
    if args.annotations:
        tasks.append(("http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
                      root / "sources/annotations_trainval2014.zip", None))
    for url, path, blob in tasks:
        print(f"Fetching {path.name}", flush=True)
        metadata = download(url, path, git_blob=blob)
        key = str(path.relative_to(root)).replace("\\", "/")
        if key in lock and lock[key]["sha256"] != metadata["sha256"]:
            raise ValueError(f"Existing source lock differs: {key}")
        lock[key] = metadata
        write_json(lock_path, lock)
    if args.annotations_fast:
        # HTTPS path-style access to the same official public COCO S3 bucket.
        url = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip"
        destination = root / "sources/instances_val2014.json"
        print("Fetching original instances_val2014.json using HTTP ranges", flush=True)
        with RemoteZip(url) as remote, zipfile.ZipFile(remote) as archive:
            member = archive.getinfo("annotations/instances_val2014.json")
            content = archive.read(member)  # zipfile verifies the ZIP CRC
        temporary = destination.with_suffix(".json.part")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(content)
        temporary.replace(destination)
        key = "sources/instances_val2014.json"
        record = {"sha256": sha256(destination), "url": url, "zip_member": member.filename,
                  "zip_crc32": member.CRC, "bytes": len(content)}
        if key in lock and lock[key]["sha256"] != record["sha256"]:
            raise ValueError("COCO annotation differs from existing source lock")
        lock[key] = record
        write_json(lock_path, lock)
    if args.annotations:
        archive = root / "sources/annotations_trainval2014.zip"
        destination = root / "sources/instances_val2014.json"
        with zipfile.ZipFile(archive) as source:
            # Extract exactly one known member; never extract arbitrary paths.
            content = source.read("annotations/instances_val2014.json")
        destination.write_bytes(content)
        lock["sources/instances_val2014.json"] = {"sha256": sha256(destination), "archive": sha256(archive)}
        write_json(lock_path, lock)
    if args.model or args.model_metadata:
        os.environ["HF_HOME"] = str(ROOT / ".venv/hf_home")
        from huggingface_hub import snapshot_download
        print(f"Fetching {config['model_id']} @ {config['revision']}", flush=True)
        patterns = ["*.json", "*.jinja", "*.model"] + (["*.safetensors"] if args.model else [])
        snapshot = Path(snapshot_download(config["model_id"], revision=config["revision"],
                                         allow_patterns=patterns,
                                         max_workers=3))
        if args.model:
            model_lock = {"id": config["model_id"], "revision": config["revision"],
                          "files": {p.name: sha256(p) for p in snapshot.iterdir() if p.is_file()}}
            write_json(root / "model.lock.json", model_lock)
    print(json.dumps({"source_lock": str(lock_path), "assets": list(lock)}, indent=2))


if __name__ == "__main__":
    main()
