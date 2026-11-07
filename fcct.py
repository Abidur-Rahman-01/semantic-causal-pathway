"""Independent FCCT/IRI reproduction CLI. See docs/fcct_reproduction.md."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
# Set before importing torch/CUDA or huggingface_hub.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["HF_HOME"] = str(ROOT / ".venv/hf_home")


def preflight(config, config_path):
    from fcct.io import environment, read_jsonl, sha256
    from fcct.data import validate_manifest
    import torch
    root = ROOT / config["data_root"]
    checks, issues = {}, []
    for kind in ("trace", "pope"):
        path = root / f"{kind}.jsonl"
        try:
            rows = read_jsonl(path)
            validate_manifest(rows, root, kind)
            checks[kind] = {"questions": len(rows), "images": len({r['image_id'] for r in rows}), "sha256": sha256(path)}
        except (OSError, ValueError, KeyError) as error:
            issues.append(f"{kind}: {error}")
    if "trace" in checks and "pope" in checks:
        a = {r["image_id"] for r in read_jsonl(root / "trace.jsonl")}
        b = {r["image_id"] for r in read_jsonl(root / "pope.jsonl")}
        if a & b:
            issues.append("Tracing and POPE image IDs overlap")
        if len(a) != config["trace_images"]:
            issues.append("Tracing image count differs from config")
        if checks["pope"]["questions"] != 9000:
            issues.append("Expected 9000 original COCO POPE questions")
    from huggingface_hub import snapshot_download
    try:
        snapshot = Path(snapshot_download(config["model_id"], revision=config["revision"], local_files_only=True))
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        if any(not (snapshot / f).is_file() for f in set(index["weight_map"].values())):
            issues.append("Model weight shards missing; run setup_fcct.py --model")
        lock_path = root / "model.lock.json"
        if lock_path.exists():
            model_lock = json.loads(lock_path.read_text(encoding="utf-8"))
            if model_lock["id"] != config["model_id"] or model_lock["revision"] != config["revision"]:
                issues.append("Model lock belongs to a different checkpoint")
            elif any(not (snapshot / name).is_file() or sha256(snapshot / name) != digest
                     for name, digest in model_lock["files"].items()):
                issues.append("Model file checksum mismatch")
        else:
            issues.append("Model checksum lock missing; complete setup_fcct.py --model")
        checks["model_snapshot"] = str(snapshot)
    except Exception as error:
        issues.append(f"Model not fully cached: {type(error).__name__}")
    gpu = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = {"name": torch.cuda.get_device_name(0), "free_bytes": free, "total_bytes": total}
        # CUDA's WDDM budget can overstate physical free memory when another
        # application has resident weights. Query physical VRAM as well.
        try:
            physical = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=memory.free",
                                       "--format=csv,noheader,nounits"], capture_output=True,
                                      text=True, timeout=10, check=True)
            gpu["physical_free_bytes"] = int(physical.stdout.strip()) * 1024**2
            free = min(free, gpu["physical_free_bytes"])
        except (OSError, ValueError, subprocess.SubprocessError):
            gpu["physical_free_bytes"] = None
        if config["device"] == "cuda" and free < 17 * 1024**3:
            issues.append("Less than 17 GiB GPU memory free; LLaVA FP16 tracing may not fit")
    elif config["device"] == "cuda":
        issues.append("CUDA requested but unavailable")
    return {"ready": not issues, "issues": issues, "checks": checks, "gpu": gpu,
            "environment": environment(), "config_sha256": sha256(config_path),
            "exact_reproduction_available": False,
            "reason": "Author repository empty; exact data subset, noise and implementation settings unpublished"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fcct_llava15.json")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Check local assets, environment and GPU memory")
    check.add_argument("--output", type=Path, help="Write a machine-readable readiness report")
    prep = commands.add_parser("prepare", help="Build disjoint tracing and original POPE manifests")
    prep.add_argument("--download-images", action="store_true")
    trace = commands.add_parser("trace", help="Run component x layer x token-category restoration")
    trace.add_argument("--limit", type=int, help="Maximum additional questions (engineering check)")
    profile = commands.add_parser("profile", help="Fit image-balanced layer RRs from observed tracing")
    profile.add_argument("--allow-partial", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Generate paired regular/IRI original POPE predictions")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--allow-partial", action="store_true")
    report = commands.add_parser("report", help="Compare measured POPE results to paper Table 2")
    report.add_argument("--tolerance", type=float, default=1.0)
    commands.add_parser("run", help="Run tracing, profile, POPE evaluation and report with prepared assets")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    data, out = ROOT / config["data_root"], ROOT / config["output_root"]
    if args.command == "check":
        result = preflight(config, args.config)
        if args.output:
            from fcct.io import write_json
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0 if result["ready"] else 2
    if args.command == "prepare":
        from fcct.data import prepare
        result = prepare(config, ROOT, args.download_images)
    else:
        from fcct.experiment import deterministic, fit_profile, run_evaluation, run_trace
        from fcct.report import report
        deterministic(config["seed"])
        if args.command in ("trace", "evaluate", "run"):
            checks = preflight(config, args.config)
            if not checks["ready"]:
                print(json.dumps(checks, indent=2))
                return 2
            from fcct.llava import Llava15
            backend = Llava15(config)
        if args.command == "trace":
            result = run_trace(backend, config, ROOT, data / "trace.jsonl", out / "trace.jsonl", args.limit)
        elif args.command == "profile":
            result = fit_profile(out / "trace.jsonl", data / "trace.jsonl", out / "profile.json", args.allow_partial)
        elif args.command == "evaluate":
            result = run_evaluation(backend, config, ROOT, data / "pope.jsonl", out / "pope.jsonl", out / "profile.json",
                                    limit=args.limit, allow_partial=args.allow_partial)
        elif args.command == "report":
            result = report(out / "pope.jsonl", data / "pope.jsonl", out / "profile.json", ROOT, out / "report", args.tolerance)
        else:
            run_trace(backend, config, ROOT, data / "trace.jsonl", out / "trace.jsonl")
            fit_profile(out / "trace.jsonl", data / "trace.jsonl", out / "profile.json")
            run_evaluation(backend, config, ROOT, data / "pope.jsonl", out / "pope.jsonl", out / "profile.json")
            result = report(out / "pope.jsonl", data / "pope.jsonl", out / "profile.json", ROOT, out / "report")
    print(json.dumps(result, indent=2))
    return 3 if result.get("all_pope_metrics_within_tolerance") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
