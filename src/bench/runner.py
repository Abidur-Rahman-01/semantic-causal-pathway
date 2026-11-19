"""Resumable generation cells (benchmark x method x protocol x seed), judge, latency."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from fcct.core import IRI
from fcct.io import environment, fingerprint, read_jsonl, sha256, write_json, write_jsonl
from rfcct.plan import PlannedIRI, paper_plan

from . import BENCH_VERSION
from .data import image_root, load_image
from .decoding import Method, timed

BENCHMARKS = ("pope", "mme", "chair", "mmhal")


def code_hashes(repo: Path):
    paths = sorted(p for d in ("src/fcct", "src/rfcct", "src/bench") for p in (repo / d).glob("*.py"))
    paths += [p for p in (repo / "bench.py",) if p.exists()]
    return {str(p.relative_to(repo)).replace("\\", "/"): sha256(p) for p in paths}


def open_cell(output: Path, provenance: dict, keys):
    signature = fingerprint(provenance)
    meta = output.with_suffix(".meta.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not meta.exists():
        raise ValueError(f"{output} has no provenance sidecar")
    if meta.exists():
        if json.loads(meta.read_text(encoding="utf-8"))["signature"] != signature:
            raise ValueError(f"Resume rejected for {output}: code/config/inputs/environment changed; "
                             "delete the file to regenerate it")
    else:
        write_json(meta, {"signature": signature, **provenance})
    existing = read_jsonl(output) if output.exists() else []
    seen = [r["key"] for r in existing]
    if len(set(seen)) != len(seen) or not set(seen) <= set(keys):
        raise ValueError(f"{output} contains duplicate or unexpected records")
    return existing


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def _profile_rr(repo, base):
    profile = repo / base["output_root"] / "profile.json"
    if not profile.is_file():
        raise ValueError("IRI needs the FCCT RR profile: run `python fcct.py run` (or copy the archived profile)")
    data = json.loads(profile.read_text(encoding="utf-8"))
    return data["rr"], sha256(profile)


def method_spec(name, cfg, repo, base, rconfig):
    """Everything that defines a method, for provenance (no model needed)."""
    if name.startswith("abl:"):
        case = ablation_case(cfg, repo, name[4:])
        rr, digest = _profile_rr(repo, base)
        return {"type": "ablation", "case": case, "profile_sha256": digest}
    spec = dict(cfg["methods"][name])
    if spec["type"] == "iri":
        spec["profile_sha256"] = _profile_rr(repo, base)[1]
    elif spec["type"] == "rfcct":
        out = repo / rconfig["output_root"]
        if not (out / "frozen.json").is_file() or not (out / "plans.json").is_file():
            raise ValueError("R-FCCT outputs missing: run Part B (`python -u rfcct.py run`) or drop "
                             f"'{name}' from the methods in configs/bench_llava15.json")
        frozen = json.loads((out / "frozen.json").read_text(encoding="utf-8"))
        plans = json.loads((out / "plans.json").read_text(encoding="utf-8"))
        if frozen["plans_sha256"] != sha256(out / "plans.json"):
            raise ValueError("R-FCCT frozen strengths do not match plans.json; rerun rfcct.py tune/freeze")
        chosen = frozen["methods"][spec["method"]]
        spec.update(plan=plans["plans"][chosen["variant"]], attention_scale=chosen["attention_scale"],
                    mlp_scale=chosen["mlp_scale"], frozen_sha256=sha256(out / "frozen.json"))
    return spec


def ablation_case(cfg, repo, case_id):
    plan = json.loads((repo / cfg["ablations"]["plan"]).read_text(encoding="utf-8"))
    case = next((c for c in plan["settings"] if c["id"] == case_id), None)
    if case is None:
        raise ValueError(f"Unknown ablation {case_id}")
    if not case["supported"]:
        raise ValueError(f"Ablation {case_id} is not implemented: {case['reason']}")
    return case


def build_method(name, spec, backend, repo, base):
    kind = spec["type"]
    if kind == "regular":
        return Method(name)
    if kind == "vcd":
        return Method(name, vcd=spec)
    if kind == "iri":
        rr, _ = _profile_rr(repo, base)
        plan = paper_plan(rr, spec["k_source"], spec["k_target"])
        return Method(name, PlannedIRI(backend.layers, plan, attention_scale=spec["attention_scale"],
                                       mlp_scale=spec["mlp_scale"], normalize=base["normalize"]))
    if kind == "rfcct":
        return Method(name, PlannedIRI(backend.layers, spec["plan"], attention_scale=spec["attention_scale"],
                                       mlp_scale=spec["mlp_scale"], normalize=base["normalize"]))
    if kind == "ablation":
        rr, _ = _profile_rr(repo, base)
        o = {"attention_scale": 0.26, "mlp_scale": 0.16, "k_source": 3, "k_target": 10, "normalize": True,
             "rr_weighting": True, "iri_allowed_layers": None, **spec["case"]["overrides"]}
        if spec["case"]["method"] == "regular":
            return Method(name)
        return Method(name, IRI(backend.layers, rr, k_source=o["k_source"], k_target=o["k_target"],
                                attention_scale=o["attention_scale"], mlp_scale=o["mlp_scale"],
                                normalize=o["normalize"], allowed_layers=o["iri_allowed_layers"],
                                rr_weighting=o["rr_weighting"]))
    raise ValueError(f"Unknown method type {kind}")


# ---------------------------------------------------------------------------
# Generation cells
# ---------------------------------------------------------------------------

def manifest(cfg, repo, benchmark):
    name = "pope" if benchmark == "pope_adversarial" else benchmark
    path = repo / cfg["data_root"] / f"{name}.jsonl"
    if not path.is_file():
        raise ValueError(f"Manifest missing: {path}; run `python bench.py prepare --benchmarks {name}`")
    rows = read_jsonl(path)
    if benchmark == "pope_adversarial":
        rows = [r for r in rows if r["split"] == "adversarial"]
    return path, rows


def cell_path(cfg, repo, benchmark, method, protocol, seed):
    safe = method.replace(":", "-")
    return repo / cfg["output_root"] / "generations" / benchmark / f"{safe}__{protocol}__s{seed}.jsonl"


def run_cell(backend, cfg, base, rconfig, repo, benchmark, method_name, protocol_name, seed, *,
             yes_no_ids=None, limit=None):
    path, rows = manifest(cfg, repo, benchmark)
    unique = {}
    for r in rows:
        unique.setdefault(r["key"], r)
    protocol = cfg["protocols"][protocol_name]
    seed = int(seed) if protocol["do_sample"] else 0
    bench_cfg = cfg["pope" if benchmark == "pope_adversarial" else benchmark]
    spec = method_spec(method_name, cfg, repo, base, rconfig)
    provenance = {"bench_version": BENCH_VERSION, "benchmark": benchmark, "method": spec,
                  "protocol": protocol, "seed": seed, "max_new_tokens": bench_cfg["max_new_tokens"],
                  "manifest_sha256": sha256(path), "model": [base["model_id"], base["revision"], base["dtype"]],
                  "code_sha256": code_hashes(repo), "environment": environment()}
    output = cell_path(cfg, repo, benchmark, method_name, protocol_name, seed)
    results = open_cell(output, provenance, list(unique))
    done = {r["key"] for r in results}
    pending = [k for k in unique if k not in done][:limit]
    if not pending:
        return {"output": str(output), "completed": len(results), "expected": len(unique)}
    method = build_method(method_name, spec, backend, repo, base)
    root = image_root(cfg, base, repo, "pope" if benchmark == "pope_adversarial" else benchmark)
    yes_no = yes_no_ids if benchmark in ("pope", "pope_adversarial") else None
    cache = {}
    for i, key in enumerate(pending, 1):
        row = unique[key]
        if row["image"] not in cache:
            cache = {row["image"]: load_image(root / row["image"])}
        out = method.generate(backend, cache[row["image"]], row["question"], protocol=protocol,
                              max_new_tokens=bench_cfg["max_new_tokens"], seed_key=key, seed=seed,
                              benchmark=benchmark, yes_no_ids=yes_no)
        results.append({"key": key, **out})
        if i % 10 == 0 or i == len(pending):
            write_jsonl(output, results)
        print(f"[{benchmark} {method_name} {protocol_name} s{seed}] {len(results)}/{len(unique)}", flush=True)
    return {"output": str(output), "completed": len(results), "expected": len(unique)}


def cells(cfg, benchmark):
    bench_cfg = cfg[benchmark]
    for method in bench_cfg["methods"]:
        for protocol, seeds in bench_cfg["runs"].items():
            for seed in (seeds if cfg["protocols"][protocol]["do_sample"] else [0]):
                yield method, protocol, seed


def ablation_cells(cfg, repo):
    plan = json.loads((repo / cfg["ablations"]["plan"]).read_text(encoding="utf-8"))
    for case in plan["settings"]:
        if not case["supported"]:
            continue
        for protocol, seeds in cfg["ablations"]["runs"].items():
            for seed in (seeds if cfg["protocols"][protocol]["do_sample"] else [0]):
                yield f"abl:{case['id']}", protocol, seed


# ---------------------------------------------------------------------------
# MMHal-Bench GPT judge (external API: opt-in)
# ---------------------------------------------------------------------------

def run_judge(cfg, repo, method, protocol, seed, *, client=None):
    from .metrics import mmhal_prompt, mmhal_rating, mmhal_template
    root = repo / cfg["data_root"]
    template = mmhal_template(root / "sources/mmhal/eval_gpt4.py")
    items = json.loads((root / "sources/mmhal/response_template.json").read_text(encoding="utf-8"))
    generations = {r["key"]: r["text"] for r in read_jsonl(cell_path(cfg, repo, "mmhal", method, protocol, seed))}
    if len(generations) != 96:
        raise ValueError("Generate all 96 MMHal answers before judging")
    model = cfg["mmhal"]["judge_model"]
    output = repo / cfg["output_root"] / "judge" / f"{method}__{protocol}__s{seed}__{model}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    done = read_jsonl(output) if output.exists() else []
    if client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Set OPENAI_API_KEY to run the MMHal-Bench GPT judge (sends answers to OpenAI).")
        from openai import OpenAI
        client = OpenAI()
    for i in range(len(done), 96):
        prompt = mmhal_prompt(template, items[i], generations[f"mmhal-{i:02d}"])
        while True:
            try:
                response = client.chat.completions.create(model=model, temperature=cfg["mmhal"]["judge_temperature"],
                                                          messages=[{"role": "user", "content": prompt}])
                break
            except Exception as error:  # eval_gpt4.py: print, wait 10 s, retry
                print(error, "retrying...", flush=True)
                time.sleep(10)
        content = response.choices[0].message.content
        done.append({"index": i, "judge_model": model, "response": content, "rating": mmhal_rating(content)})
        write_jsonl(output, done)
        print(f"[judge {method}] {i + 1}/96 rating={done[-1]['rating']}", flush=True)
    return {"output": str(output)}


# ---------------------------------------------------------------------------
# Table 7 latency
# ---------------------------------------------------------------------------

def run_latency(backend, cfg, base, rconfig, repo):
    _, rows = manifest(cfg, repo, "pope_adversarial")
    unique = list({r["key"]: r for r in rows}.values())[: cfg["latency"]["prompts"] + cfg["latency"]["warmup"]]
    root = image_root(cfg, base, repo, "pope")
    n = cfg["latency"]["new_tokens"]
    protocol = cfg["protocols"]["greedy"]
    result = {"device": str(backend.device), "environment": environment(), "new_tokens": n, "methods": {}}
    for name in cfg["latency"]["methods"]:
        method = build_method(name, method_spec(name, cfg, repo, base, rconfig), backend, repo, base)
        ttft, tpot = [], []
        for j, row in enumerate(unique):
            image = load_image(root / row["image"])
            run = lambda tokens: method.generate(backend, image, row["question"], protocol=protocol,
                                                 max_new_tokens=tokens, min_new_tokens=tokens,
                                                 seed_key=row["key"], benchmark="pope")
            first, t1 = timed(lambda: run(1), backend.device)
            full, tn = timed(lambda: run(n), backend.device)
            if full["new_tokens"] != n:
                raise RuntimeError("Latency run did not produce the requested number of tokens")
            if j >= cfg["latency"]["warmup"]:
                ttft.append(1000 * t1)
                tpot.append(1000 * (tn - t1) / (n - 1))
        result["methods"][name] = {"ttft_ms": float(np.mean(ttft)), "tpot_ms": float(np.mean(tpot)),
                                   "ttft_ms_sd": float(np.std(ttft, ddof=1)), "tpot_ms_sd": float(np.std(tpot, ddof=1)),
                                   "prompts": len(ttft)}
        print(f"[latency {name}] TTFT {result['methods'][name]['ttft_ms']:.1f} ms, "
              f"TPOT {result['methods'][name]['tpot_ms']:.1f} ms", flush=True)
    write_json(repo / cfg["output_root"] / "latency.json", result)
    return result
