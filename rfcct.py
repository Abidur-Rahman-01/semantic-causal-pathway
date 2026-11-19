"""R-FCCT command line: reliability extensions to FCCT/IRI. See README.md."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["HF_HOME"] = str(ROOT / ".venv/hf_home")

MODEL_STAGES = ("noise", "trace", "tune", "evaluate", "sample", "run")


def paths_for(rc, base):
    data, out = ROOT / rc["data_root"], ROOT / rc["output_root"]
    return {"contrastive": data / "contrastive_trace.jsonl", "calibration": data / "calibration.jsonl",
            "test": data / "test_pope.jsonl", "noise": out / "noise_calibration.json",
            "trace": out / "contrastive_trace.jsonl", "plans": out / "plans.json",
            "tune": out / "tune_margins.jsonl", "frozen": out / "frozen.json",
            "evaluate": out / "greedy.jsonl", "sample": out / "sampling.jsonl", "report_dir": out / "report"}


def load_backend(rc, base):
    import torch
    from fcct.experiment import deterministic
    from fcct.llava import Llava15
    from rfcct.tracing import resolve_token
    if base["device"] == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the 7B model (see README: hardware).")
    deterministic(rc["seed"])
    backend = Llava15(base)
    ids = {"yes": backend.yes_token_id,
           "no": resolve_token(backend.processor.tokenizer, rc["no_token"], rc["expected_no_token_id"])}
    return backend, ids


def preflight(rc, base, paths):
    """Check every prerequisite of the full run before the model is loaded."""
    problems = [f"missing {paths[k]} (run `python rfcct.py prepare --download-images`)"
                for k in ("contrastive", "calibration", "test") if not paths[k].is_file()]
    profile = ROOT / base["output_root"] / "profile.json"
    if "paper_iri" in rc["variants"] and not profile.is_file():
        problems.append(f"missing {profile} (run Part A: `python -u fcct.py run`)")
    if not rc["trace"]["min_margin_gap"] > 0:
        problems.append("trace.min_margin_gap must be positive")
    unknown = [m for m in rc["evaluation"]["methods"] + rc["sampling"]["methods"]
               if m != "regular" and m.split("@")[0] not in rc["variants"]]
    if unknown:
        problems.append(f"methods use undefined variants: {unknown}")
    if problems:
        raise SystemExit("Cannot start the R-FCCT run:\n- " + "\n- ".join(problems))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rfcct_llava15.json")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Build contrastive-trace, calibration and test manifests")
    prep.add_argument("--download-images", action="store_true")
    sub.add_parser("noise", help="Calibrate the Gaussian corruption level")
    for name, help_text in (("trace", "Contrastive last-token tracing"), ("tune", "Margin scoring on the calibration grid"),
                            ("evaluate", "Greedy POPE generation for all methods"),
                            ("sample", "VCD-protocol sampled POPE generation")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--limit", type=int, help="Process at most this many more items (smoke tests)")
    sub.add_parser("select", help="Fit causal scores and write layer plans")
    sub.add_parser("freeze", help="Choose strengths/thresholds on calibration data")
    sub.add_parser("report", help="Write outputs/rfcct/report")
    run = sub.add_parser("run", help="Every stage in order (resumable)")
    run.add_argument("--skip-sampling", action="store_true")
    args = parser.parse_args(argv)
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    rc = json.loads(args.config.read_text(encoding="utf-8"))
    base = json.loads((ROOT / rc["base_config"]).read_text(encoding="utf-8"))
    paths = paths_for(rc, base)
    from rfcct import pipeline
    from rfcct.report import build_report

    if args.command == "prepare":
        from rfcct.data import prepare
        result = prepare(rc, base, ROOT, args.download_images)
    elif args.command == "select":
        result = pipeline.run_select(rc, base, ROOT, paths)
    elif args.command == "freeze":
        result = pipeline.freeze(rc, base, paths)
    elif args.command == "report":
        result = build_report(rc, base, ROOT, paths)
    else:
        if args.command == "run":
            preflight(rc, base, paths)  # fail before hours of GPU work, not after
        backend, ids = load_backend(rc, base)
        limit = getattr(args, "limit", None)
        if args.command == "noise":
            result = pipeline.run_noise(backend, ids, rc, base, ROOT, paths)
        elif args.command == "trace":
            result = pipeline.run_trace(backend, ids, rc, base, ROOT, paths, limit)
        elif args.command == "tune":
            result = pipeline.run_tune(backend, ids, rc, base, ROOT, paths, limit)
        elif args.command == "evaluate":
            result = pipeline.run_evaluate(backend, ids, rc, base, ROOT, paths, limit)
        elif args.command == "sample":
            result = pipeline.run_evaluate(backend, ids, rc, base, ROOT, paths, limit, sampling=True)
        else:
            if rc["noise"].get("fixed_sigma") is None and not paths["noise"].is_file():
                pipeline.run_noise(backend, ids, rc, base, ROOT, paths)
            pipeline.run_trace(backend, ids, rc, base, ROOT, paths)
            pipeline.run_select(rc, base, ROOT, paths)
            pipeline.run_tune(backend, ids, rc, base, ROOT, paths)
            pipeline.freeze(rc, base, paths)
            pipeline.run_evaluate(backend, ids, rc, base, ROOT, paths)
            if not args.skip_sampling:
                pipeline.run_evaluate(backend, ids, rc, base, ROOT, paths, sampling=True)
            result = build_report(rc, base, ROOT, paths)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
