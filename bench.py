"""Paper evaluation matrix CLI (FCCT/IRI Tables 1-4, 6, 7; LLaVA-1.5-7B). See README.md."""
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
os.environ.setdefault("NLTK_DATA", str(ROOT / ".venv/nltk_data"))

BENCHMARKS = ("pope", "mme", "chair", "mmhal")


def load(args):
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    base = json.loads((ROOT / cfg["base_config"]).read_text(encoding="utf-8"))
    rconfig = json.loads((ROOT / cfg["rfcct_config"]).read_text(encoding="utf-8"))
    return cfg, base, rconfig


def backend_for(base, rconfig):
    import torch
    from fcct.experiment import deterministic
    from fcct.llava import Llava15
    from rfcct.tracing import resolve_token
    if base["device"] == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for LLaVA-1.5-7B (see README: hardware)")
    deterministic(base["seed"])
    backend = Llava15(base)
    no = resolve_token(backend.processor.tokenizer, rconfig["no_token"], rconfig["expected_no_token_id"])
    return backend, (backend.yes_token_id, no)


def preflight(args, cfg, base, rconfig):
    """Resolve every method and manifest the command will use before loading the model."""
    from bench import runner
    problems = []
    benchmarks = list(getattr(args, "benchmarks", None) or BENCHMARKS) if args.command in ("generate", "run") else []
    if args.command in ("ablations", "latency", "run"):
        benchmarks.append("pope")
    for benchmark in dict.fromkeys(benchmarks):
        try:
            runner.manifest(cfg, ROOT, benchmark)
        except ValueError as error:
            problems.append(str(error))
    names = set()
    for benchmark in BENCHMARKS:
        if benchmark in benchmarks:
            names |= {m for m in cfg[benchmark]["methods"] if not getattr(args, "methods", None) or m in args.methods}
    if args.command in ("latency", "run"):
        names |= set(cfg["latency"]["methods"])
    if args.command in ("ablations", "run"):
        names |= {m for m, _, _ in runner.ablation_cells(cfg, ROOT)}
    for name in sorted(names):
        try:
            runner.method_spec(name, cfg, ROOT, base, rconfig)
        except (ValueError, OSError, KeyError) as error:
            problems.append(f"method {name}: {error}")
    if args.command == "run" and "chair" in benchmarks and not (ROOT / cfg["data_root"] / "sources/chair/chair.py").is_file():
        problems.append("CHAIR evaluator missing: run `python scripts/setup_bench.py --chair`")
    if problems:
        raise SystemExit("Cannot start:\n- " + "\n- ".join(problems))


def selected_methods(cfg, benchmark, only):
    methods = cfg[benchmark]["methods"]
    return [m for m in methods if not only or m in only]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bench_llava15.json")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Write benchmark manifests")
    prep.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS)
    prep.add_argument("--download-images", action="store_true", help="Fetch the CHAIR COCO images")
    gen = sub.add_parser("generate", help="Generate answers for benchmark x method x protocol x seed cells")
    gen.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS)
    gen.add_argument("--methods", nargs="+", help="Subset of methods (default: all in the config)")
    gen.add_argument("--protocols", nargs="+", help="Subset of protocols")
    gen.add_argument("--limit", type=int, help="At most this many more items per cell (smoke tests)")
    abl = sub.add_parser("ablations", help="Table 6 controls on POPE adversarial")
    abl.add_argument("--limit", type=int)
    sub.add_parser("latency", help="Table 7 TTFT/TPOT")
    judge = sub.add_parser("judge", help="MMHal-Bench GPT judge (sends answers to the OpenAI API)")
    judge.add_argument("--methods", nargs="+")
    sub.add_parser("report", help="Write outputs/bench/report")
    run = sub.add_parser("run", help="prepare-free: generate all, ablations, latency, report")
    run.add_argument("--with-judge", action="store_true", help="Also call the OpenAI judge for MMHal-Bench")
    args = parser.parse_args(argv)
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    cfg, base, rconfig = load(args)
    from bench import report, runner

    if args.command == "prepare":
        from bench.data import prepare
        result = prepare(cfg, base, ROOT, args.benchmarks, args.download_images)
    elif args.command == "report":
        result = report.build(cfg, ROOT)
    elif args.command == "judge":
        outputs = []
        for method, protocol, seed in runner.cells(cfg, "mmhal"):
            if not args.methods or method in args.methods:
                outputs.append(runner.run_judge(cfg, ROOT, method, protocol, seed))
        result = {"judged": outputs}
    else:
        preflight(args, cfg, base, rconfig)  # fail before any GPU work
        backend, yes_no = backend_for(base, rconfig)
        result = {}
        if args.command in ("generate", "run"):
            for benchmark in getattr(args, "benchmarks", None) or BENCHMARKS:
                for method, protocol, seed in runner.cells(cfg, benchmark):
                    if getattr(args, "methods", None) and method not in args.methods:
                        continue
                    if getattr(args, "protocols", None) and protocol not in args.protocols:
                        continue
                    result[f"{benchmark}/{method}/{protocol}/s{seed}"] = runner.run_cell(
                        backend, cfg, base, rconfig, ROOT, benchmark, method, protocol, seed,
                        yes_no_ids=yes_no, limit=getattr(args, "limit", None))
        if args.command in ("ablations", "run"):
            for method, protocol, seed in runner.ablation_cells(cfg, ROOT):
                result[f"ablation/{method}/{protocol}/s{seed}"] = runner.run_cell(
                    backend, cfg, base, rconfig, ROOT, "pope_adversarial", method, protocol, seed,
                    yes_no_ids=yes_no, limit=getattr(args, "limit", None))
        if args.command in ("latency", "run"):
            result["latency"] = runner.run_latency(backend, cfg, base, rconfig, ROOT)
        if args.command == "run":
            if args.with_judge:
                for method, protocol, seed in runner.cells(cfg, "mmhal"):
                    runner.run_judge(cfg, ROOT, method, protocol, seed)
            result["report"] = report.build(cfg, ROOT)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
