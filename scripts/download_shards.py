"""Robust resumable downloader for missing LLaVA safetensor shards."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

try:
    import urllib3.util.connection as urllib3_cn
    urllib3_cn.HAS_IPV6 = False
except ImportError:
    pass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fcct.io import sha256, write_json

SHARDS = [
    {
        "filename": "model-00001-of-00003.safetensors",
        "sha256": "c11dbf016ee7d35ee130c19b67e20eb04873996006f424b7bcbd453b6517ee66",
        "url": "https://huggingface.co/llava-hf/llava-1.5-7b-hf/resolve/b234b804b114d9e37bb655e11cbbb5f5e971b7a9/model-00001-of-00003.safetensors",
    },
    {
        "filename": "model-00002-of-00003.safetensors",
        "sha256": "46df6c6e5fad297fe7fbca4963dcf180cb1d0528cabc884d1f603311fed01328",
        "url": "https://huggingface.co/llava-hf/llava-1.5-7b-hf/resolve/b234b804b114d9e37bb655e11cbbb5f5e971b7a9/model-00002-of-00003.safetensors",
    },
    {
        "filename": "model-00003-of-00003.safetensors",
        "sha256": "4f06177c37ca13944e73b57dd6d051767b950dffc12400c700dfa2509c24c697",
        "url": "https://huggingface.co/llava-hf/llava-1.5-7b-hf/resolve/b234b804b114d9e37bb655e11cbbb5f5e971b7a9/model-00003-of-00003.safetensors",
    },
]


def make_session():
    session = requests.Session()
    session.headers.update({"User-Agent": "fcct-independent-reproduction/0.1"})
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_shard(session: requests.Session, shard_info: dict, dest_dir: Path, blobs_dir: Path):
    target = dest_dir / shard_info["filename"]
    part = dest_dir / (shard_info["filename"] + ".part")
    expected_sha256 = shard_info["sha256"]
    url = shard_info["url"]

    if target.is_file():
        print(f"Checking existing {target.name}...", flush=True)
        h = sha256(target)
        if h == expected_sha256:
            print(f"Shard {target.name} already verified (SHA256 matches).", flush=True)
            ensure_blob(target, blobs_dir / expected_sha256)
            return target
        else:
            print(f"Warning: {target.name} hash mismatch ({h} != {expected_sha256}). Re-downloading.", flush=True)
            target.unlink()

    # Get total file size from server
    resp = session.get(url, stream=True, timeout=(15, 30))
    resp.raise_for_status()
    total_size = int(resp.headers.get("Content-Length", 0))
    resolved_url = resp.url
    resp.close()

    print(f"Downloading {shard_info['filename']} ({total_size / (1024**3):.2f} GB)...", flush=True)

    max_retries = 100
    chunk_size = 4 * 1024 * 1024  # 4 MB chunks

    for attempt in range(max_retries):
        current_size = part.stat().st_size if part.is_file() else 0
        if current_size >= total_size:
            break

        headers = {}
        if current_size > 0:
            headers["Range"] = f"bytes={current_size}-{total_size - 1}"
            print(f"Resuming {shard_info['filename']} from {current_size / (1024*1024):.1f} MB ({current_size/total_size*100:.1f}%)...", flush=True)

        try:
            with session.get(resolved_url, headers=headers, stream=True, timeout=(30, 60)) as r:
                r.raise_for_status()
                with open(part, "ab" if current_size > 0 else "wb") as f:
                    last_log = time.time()
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            current_size += len(chunk)
                            now = time.time()
                            if now - last_log >= 10:
                                print(f"Progress {shard_info['filename']}: {current_size / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({current_size/total_size*100:.1f}%)", flush=True)
                                last_log = now
        except (requests.RequestException, Exception) as e:
            print(f"Connection interrupted ({type(e).__name__}: {e}). Retrying in 3 seconds... (Attempt {attempt + 1}/{max_retries})", flush=True)
            time.sleep(3)

    print(f"Verifying SHA256 for {part.name}...", flush=True)
    actual_hash = sha256(part)
    if actual_hash != expected_sha256:
        raise ValueError(f"Checksum mismatch for {shard_info['filename']}: {actual_hash} != {expected_sha256}")

    print(f"Verification successful: {actual_hash}", flush=True)
    part.replace(target)
    ensure_blob(target, blobs_dir / expected_sha256)
    return target


def ensure_blob(source: Path, blob_target: Path):
    blobs_dir = blob_target.parent
    blobs_dir.mkdir(parents=True, exist_ok=True)
    if not blob_target.exists():
        try:
            os.link(source, blob_target)
            print(f"Hardlinked blob: {blob_target.name}", flush=True)
        except OSError:
            import shutil
            shutil.copyfile(source, blob_target)
            print(f"Copied blob: {blob_target.name}", flush=True)


def main():
    config_path = ROOT / "configs/fcct_llava15.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    snapshot_dir = ROOT / ".venv/hf_home/hub/models--llava-hf--llava-1.5-7b-hf/snapshots" / config["revision"]
    blobs_dir = ROOT / ".venv/hf_home/hub/models--llava-hf--llava-1.5-7b-hf/blobs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    blobs_dir.mkdir(parents=True, exist_ok=True)

    session = make_session()
    for shard in SHARDS:
        download_shard(session, shard, snapshot_dir, blobs_dir)

    # Clean up incomplete files
    for inc in blobs_dir.glob("*.incomplete"):
        try:
            inc.unlink()
        except OSError:
            pass

    # Write model.lock.json
    root = ROOT / config["data_root"]
    lock_path = root / "model.lock.json"
    print("Generating model.lock.json...", flush=True)
    model_lock = {
        "id": config["model_id"],
        "revision": config["revision"],
        "files": {p.name: sha256(p) for p in sorted(snapshot_dir.iterdir()) if p.is_file()}
    }
    write_json(lock_path, model_lock)
    print(f"Wrote {lock_path} with {len(model_lock['files'])} verified files.")


if __name__ == "__main__":
    main()
