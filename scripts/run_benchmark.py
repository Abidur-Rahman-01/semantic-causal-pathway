"""End-to-end benchmark execution and reproducibility CLI.

Provides unified commands for:
  - validate: Check manifest schema, review gates, paths, and image-level split isolation
  - split:    Generate image-grouped train/val/test splits with fixed seed
  - report:   Generate outcomes.csv, predictors.csv, metrics.json, report.md, figures, and checksums
  - verify:   Verify SHA-256 checksums and provenance of a completed run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_verify(run_dir: Path) -> int:
    checksum_file = run_dir / "checksums.sha256"
    if not checksum_file.is_file():
        print(f"ERROR: {checksum_file} not found.")
        return 1

    print(f"Verifying checksums in {run_dir}...")
    errors = 0
    checked = 0
    with checksum_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2:
                parts = line.split(None, 1)
            expected_hash, rel_path = parts[0], parts[1]
            target_path = run_dir / rel_path
            if not target_path.is_file():
                print(f"  MISSING: {rel_path}")
                errors += 1
                continue
            digest = hashlib.sha256()
            with target_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_hash = digest.hexdigest()
            if actual_hash.lower() != expected_hash.lower():
                print(f"  MISMATCH: {rel_path} (expected {expected_hash[:10]}..., got {actual_hash[:10]}...)")
                errors += 1
            else:
                checked += 1

    if errors == 0:
        print(f"SUCCESS: All {checked} files verified successfully against checksums.sha256.")
        return 0
    else:
        print(f"FAILURE: {errors} file(s) failed verification.")
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # Validate subcommand
    val_p = sub.add_parser("validate", help="Validate a benchmark manifest")
    val_p.add_argument("manifest", help="Path to manifest JSONL")
    val_p.add_argument("--data-root", help="Optional root for asset path resolution")
    val_p.add_argument("--require-schema-v1", action="store_true", default=True)

    # Split subcommand
    split_p = sub.add_parser("split", help="Deterministically split manifest by image_id")
    split_p.add_argument("manifest", help="Input manifest JSONL")
    split_p.add_argument("--output", required=True, help="Output manifest JSONL")
    split_p.add_argument("--seed", type=int, default=17)
    split_p.add_argument("--train", type=float, default=0.6)
    split_p.add_argument("--val", type=float, default=0.2)
    split_p.add_argument("--test", type=float, default=0.2)

    # Report subcommand
    rep_p = sub.add_parser("report", help="Generate full run outputs, metrics, tables, and figures")
    rep_p.add_argument("predictions", help="Prediction JSONL file")
    rep_p.add_argument("--manifest", required=True, help="Frozen manifest JSONL")
    rep_p.add_argument("--config", required=True, help="Frozen config JSON")
    rep_p.add_argument("--output-dir", required=True, help="Run output directory")
    rep_p.add_argument("--run-id", required=True, help="Unique Run ID (e.g. 2026-10-21_qwen25vl7b_v1)")
    rep_p.add_argument("--model-revision", required=True, help="Model revision or commit hash")
    rep_p.add_argument("--scoring-version", default="vqav2_soft_consensus_v2.0_gqa_v1.2")
    rep_p.add_argument("--bootstrap-repeats", type=int, default=2000)
    rep_p.add_argument("--seed", type=int, default=17)

    # Verify subcommand
    ver_p = sub.add_parser("verify", help="Verify checksums of a completed run")
    ver_p.add_argument("run_dir", help="Path to run directory")

    args = parser.parse_args()

    if args.command == "validate":
        cmd = [sys.executable, str(ROOT / "scripts" / "benchmark_manifest.py"), "validate", args.manifest]
        if args.data_root:
            cmd += ["--data-root", args.data_root]
        if args.require_schema_v1:
            cmd += ["--require-schema-v1"]
        sys.exit(subprocess.run(cmd).returncode)

    elif args.command == "split":
        cmd = [
            sys.executable, str(ROOT / "scripts" / "benchmark_manifest.py"), "split",
            args.manifest, "--output", args.output, "--seed", str(args.seed),
            "--train", str(args.train), "--validation", str(args.val), "--test", str(args.test),
        ]
        sys.exit(subprocess.run(cmd).returncode)

    elif args.command == "report":
        cmd = [
            sys.executable, str(ROOT / "scripts" / "report_benchmark.py"), args.predictions,
            "--manifest", args.manifest,
            "--config", args.config,
            "--output-dir", args.output_dir,
            "--run-id", args.run_id,
            "--model-revision", args.model_revision,
            "--scoring-version", args.scoring_version,
            "--bootstrap-repeats", str(args.bootstrap_repeats),
            "--seed", str(args.seed),
        ]
        sys.exit(subprocess.run(cmd).returncode)

    elif args.command == "verify":
        sys.exit(run_verify(Path(args.run_dir).resolve()))


if __name__ == "__main__":
    main()
