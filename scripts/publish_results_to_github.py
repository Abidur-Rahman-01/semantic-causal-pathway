"""Compress large experiment outputs and push results to GitHub.

Enforces:
1. Compresses files larger than 25MB into .gz or .zip bundles before committing.
2. Ensures all reports, diagrams, manifests, and configs are properly staged.
3. Automatically commits and pushes to GitHub repository using authenticated remote.
"""
from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_RAW_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


def compress_file_if_large(file_path: Path) -> Path:
    """Compresses file if larger than threshold, returns path to tracked file."""
    if not file_path.is_file():
        return file_path
    
    size = file_path.stat().st_size
    if size > MAX_RAW_FILE_SIZE_BYTES:
        gz_path = file_path.with_suffix(file_path.suffix + ".gz")
        print(f"File {file_path.name} is {size / (1024*1024):.1f}MB (>25MB). Compressing to {gz_path.name}...")
        with file_path.open("rb") as f_in, gzip.open(gz_path, "wb", compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_size = gz_path.stat().st_size
        print(f"Compressed {file_path.name}: {size / (1024*1024):.1f}MB -> {gz_size / (1024*1024):.1f}MB (ratio: {gz_size/size:.1%})")
        return gz_path
    return file_path


def compress_directory_to_zip(dir_path: Path, zip_path: Path):
    """Compresses entire directory into a zip archive."""
    print(f"Creating zip archive of {dir_path.name} at {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                abs_p = Path(root) / file
                rel_p = abs_p.relative_to(dir_path)
                zipf.write(abs_p, rel_p)
    print(f"Created {zip_path.name} ({zip_path.stat().st_size / (1024*1024):.2f}MB).")


def run_git_command(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True, check=True)


def main():
    print("=" * 70)
    print("Preparing and Publishing Experiment Results to GitHub")
    print("=" * 70)

    # 1. Check outputs to compress
    confirmatory_dir = ROOT / ".venv" / "outputs" / "pathway_confirmatory"
    if confirmatory_dir.exists():
        for jsonl_file in confirmatory_dir.rglob("*.jsonl"):
            compress_file_if_large(jsonl_file)

    pilot_dir = ROOT / ".venv" / "outputs" / "pathway_pilot"
    if pilot_dir.exists():
        for jsonl_file in pilot_dir.rglob("*.jsonl"):
            compress_file_if_large(jsonl_file)

    # 2. Stage tracked files and core artifacts
    files_to_add = [
        "configs/",
        "docs/",
        "scripts/",
        "src/",
        "archives/",
        ".agents/",
        "README.md",
        "outputs/research_diagrams/",
    ]

    for item in files_to_add:
        p = ROOT / item
        if p.exists():
            print(f"Staging: {item}")
            try:
                run_git_command(["add", item])
            except Exception as e:
                print(f"Warning adding {item}: {e}")

    # Check status
    res = run_git_command(["status", "--short"])
    print(f"\nGit status summary:\n{res.stdout}")

    if not res.stdout.strip():
        print("No changes to commit.")
        return

    # Commit
    commit_msg = "feat: update 500-image confirmatory experiment protocol, review audit, and MCP setup"
    print(f"Committing with message: '{commit_msg}'")
    try:
        run_git_command(["commit", "-m", commit_msg])
    except subprocess.CalledProcessError as e:
        print(f"Commit output: {e.stdout}\n{e.stderr}")

    # Push
    print("\nPushing to origin main...")
    try:
        push_res = run_git_command(["push", "origin", "main"])
        print(f"Push successful!\n{push_res.stdout}\n{push_res.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Push failed: {e.stderr}")
        sys.exit(1)

    print("\nGitHub repository successfully updated!")


if __name__ == "__main__":
    main()
