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
        min_free = int(os.environ.get("MIN_GPU_FREE_BYTES", config.get("min_gpu_free_bytes", 14 * 1024**3)))
        if config["device"] == "cuda" and free < min_free:
            issues.append(f"Less than {min_free // 1024**3} GiB GPU memory free; LLaVA FP16 tracing may not fit")
    elif config["device"] == "cuda":
        issues.append("CUDA requested but unavailable")
    return {"ready": not issues, "issues": issues, "checks": checks, "gpu": gpu,
            "environment": environment(), "config_sha256": sha256(config_path),
            "exact_reproduction_available": False,
            "reason": "Author repository empty; exact data subset, noise and implementation settings unpublished"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fcct_llava15.json")
    parser.add_argument("--require-exact", action="store_true", help="Refuse execution when the complete original paper protocol is unavailable")
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
    audit = commands.add_parser("audit", help="Write the paper-to-code architecture and reproducibility audit")
    audit.add_argument("--output-dir", type=Path, default=ROOT / "docs")
    attention = commands.add_parser("attention", help="Analyze last-token visual/textual object attention (Figure 4)")
    attention.add_argument("--manifest", type=Path, required=True)
    attention.add_argument("--limit", type=int)
    attention.add_argument("--allow-partial", action="store_true")
    ablations = commands.add_parser("ablations", help="List or execute the supported Table 6 controls")
    ablations.add_argument("--plan", type=Path, default=ROOT / "configs/fcct_ablation_plan.json")
    cases = ablations.add_mutually_exclusive_group(required=True)
    cases.add_argument("--list", action="store_true")
    cases.add_argument("--all", action="store_true")
    cases.add_argument("--setting", nargs="+")
    ablations.add_argument("--limit", type=int)
    ablations.add_argument("--allow-partial", action="store_true")
    run = commands.add_parser("run", help="Run supported tracing, IRI, POPE and optional attention/ablation stages")
    run.add_argument("--attention-manifest", type=Path)
    run.add_argument("--include-ablations", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    data, out = ROOT / config["data_root"], ROOT / config["output_root"]
    if args.command == "audit" or args.require_exact:
        from fcct.audit import write_architecture_audit
        audit = write_architecture_audit(config, ROOT, getattr(args, "output_dir", ROOT / "docs"))
        if args.require_exact and not audit["exact_reproduction_ready"]:
            parser.error("Exact reproduction is unavailable: missing author artifacts, other model adapters, benchmark integrations and original judge/human evaluations. See docs/fcct_architecture_audit.md")
        if args.command == "audit":
            print(json.dumps({"status": audit["status"], "exact_reproduction_ready": False,
                              "report": str(args.output_dir / "fcct_architecture_audit.md")}, indent=2))
            return 0
    if args.command == "ablations":
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        if args.list:
            print(json.dumps(plan, indent=2))
            return 0
        selected_cases = [case["id"] for case in plan["settings"] if case["supported"]] if args.all else args.setting
        from fcct.ablations import ablation_settings
        ablation_settings(config, plan, selected_cases)
    attention_manifest = args.manifest if args.command == "attention" else getattr(args, "attention_manifest", None)
    if attention_manifest is not None:
        from fcct.io import read_jsonl
        if not attention_manifest.is_file():
            parser.error("Attention manifest does not exist; the original Figure 4 VQAs are not included in the prepared tracing subset")
        if len(read_jsonl(attention_manifest)) != 3000 and not getattr(args, "allow_partial", False):
            parser.error("Figure 4 requires 3000 VQAs; use attention --allow-partial only for independent engineering analysis")
    if args.command in ("evaluate", "ablations") and not (out / "profile.json").is_file():
        parser.error("Complete tracing and profile fitting first; no RR profile exists")
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
        if args.command in ("trace", "evaluate", "run", "attention", "ablations"):
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
        elif args.command == "attention":
            from fcct.attention import run_attention
            result = run_attention(backend, config, ROOT, args.manifest, out / "attention",
                                   limit=args.limit, allow_partial=args.allow_partial)
        elif args.command == "ablations":
            from fcct.ablations import run_ablations
            result = run_ablations(backend, config, ROOT, args.plan, selected_cases,
                                   limit=args.limit, allow_partial=args.allow_partial)
        else:
            run_trace(backend, config, ROOT, data / "trace.jsonl", out / "trace.jsonl")
            fit_profile(out / "trace.jsonl", data / "trace.jsonl", out / "profile.json")
            run_evaluation(backend, config, ROOT, data / "pope.jsonl", out / "pope.jsonl", out / "profile.json")
            if args.attention_manifest:
                from fcct.attention import run_attention
                run_attention(backend, config, ROOT, args.attention_manifest, out / "attention")
            if args.include_ablations:
                from fcct.ablations import run_ablations
                plan_path = ROOT / "configs/fcct_ablation_plan.json"
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                run_ablations(backend, config, ROOT, plan_path,
                              [case["id"] for case in plan["settings"] if case["supported"]])
            result = report(out / "pope.jsonl", data / "pope.jsonl", out / "profile.json", ROOT, out / "report")
    print(json.dumps(result, indent=2))
    return 3 if result.get("all_pope_metrics_within_tolerance") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
