"""R-FCCT stages: noise calibration, tracing, selection, tuning, evaluation."""
from __future__ import annotations

import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fcct.io import environment, fingerprint, read_jsonl, sha256, write_json, write_jsonl
from fcct.report import pope_answer

from . import EXTENSION_VERSION
from .plan import COMPONENTS, PlannedIRI, effective_edges, paper_plan
from .stats import (bootstrap_scores, best_threshold, confusion, image_tables, inclusion_frequency,
                    paper_rr_profile, ratio_scores, reachable_plan, stable_plan, threshold_for_rate)
from .tracing import calibrate_noise, contrastive_trace_sample, score

NUM_LAYERS = 32


# ---------------------------------------------------------------------------
# Provenance and resumable outputs
# ---------------------------------------------------------------------------

def code_hashes(repo: Path):
    paths = sorted(list((repo / "src/fcct").glob("*.py")) + list((repo / "src/rfcct").glob("*.py"))
                   + [repo / "rfcct.py"])
    return {str(p.relative_to(repo)).replace("\\", "/"): sha256(p) for p in paths if p.exists()}


def open_run(output: Path, inputs: dict, settings: dict, repo: Path, manifest_ids=None):
    """Resume only when code, settings, inputs and environment are unchanged."""
    provenance = {"extension_version": EXTENSION_VERSION, "settings": settings,
                  "inputs_sha256": {k: sha256(v) for k, v in sorted(inputs.items())},
                  "code_sha256": code_hashes(repo), "environment": environment()}
    signature = fingerprint(provenance)
    meta = output.with_suffix(".meta.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not meta.exists():
        raise ValueError(f"{output} has no provenance sidecar; use a new output root")
    if meta.exists():
        if json.loads(meta.read_text(encoding="utf-8"))["signature"] != signature:
            raise ValueError(f"Resume rejected for {output}: code/settings/inputs/environment changed. "
                             "Delete the stage output to recompute it.")
    else:
        write_json(meta, {"signature": signature, **provenance})
    existing = read_jsonl(output) if output.exists() else []
    keys = [r["key"] if "key" in r else r["sample_id"] for r in existing]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate records in {output}")
    if manifest_ids is not None and not set(keys) <= set(manifest_ids):
        raise ValueError(f"{output} contains records outside its manifest")
    return existing


def _image(root: Path, row):
    with Image.open(root / row["image"]) as source:
        return source.convert("RGB")


def _flush(output, results, every, force=False):
    if force or len(results) % every == 0:
        write_jsonl(output, results)


# ---------------------------------------------------------------------------
# Stage 1: corruption calibration
# ---------------------------------------------------------------------------

def run_noise(backend, ids, rc, base, repo, paths):
    rows = read_jsonl(paths["contrastive"])
    cfg = rc["noise"]
    # Balanced, deterministic subset; images are shared with tracing (no test data).
    yes = [r for r in rows if r["label"] == "yes"][: cfg["calibration_questions"] // 2]
    no = [r for r in rows if r["label"] == "no"][: cfg["calibration_questions"] // 2]
    result = calibrate_noise(backend, yes + no, repo / base["data_root"], yes_id=ids["yes"], no_id=ids["no"],
                             sigmas=cfg["sigma_grid"], seed=rc["seed"], min_drop_rate=cfg["min_drop_rate"],
                             min_median_gap=cfg["min_median_gap"])
    result["contrastive_manifest_sha256"] = sha256(paths["contrastive"])
    write_json(paths["noise"], result)
    return result


def chosen_sigma(rc, paths):
    if rc["noise"].get("fixed_sigma") is not None:
        return float(rc["noise"]["fixed_sigma"])
    if not paths["noise"].is_file():
        raise ValueError("Run the noise stage first (or set noise.fixed_sigma)")
    noise = json.loads(paths["noise"].read_text(encoding="utf-8"))
    if noise["contrastive_manifest_sha256"] != sha256(paths["contrastive"]):
        raise ValueError("Noise calibration belongs to a different manifest")
    return float(noise["chosen_sigma"])


# ---------------------------------------------------------------------------
# Stage 2: contrastive tracing
# ---------------------------------------------------------------------------

def run_trace(backend, ids, rc, base, repo, paths, limit=None):
    if not rc["trace"]["min_margin_gap"] > 0:
        raise ValueError("trace.min_margin_gap must be positive: retained traces need a measurable drop")
    rows = read_jsonl(paths["contrastive"])
    sigma = chosen_sigma(rc, paths)
    settings = {"stage": "trace", "sigma": sigma, "seed": rc["seed"], "trace": rc["trace"],
                "model": [base["model_id"], base["revision"]], "dtype": base["dtype"]}
    results = open_run(paths["trace"], {"manifest": paths["contrastive"]}, settings, repo,
                       [r["sample_id"] for r in rows])
    done = {r["sample_id"] for r in results}
    pending = [r for r in rows if r["sample_id"] not in done][:limit]
    root = repo / base["data_root"]
    for i, row in enumerate(pending, 1):
        result = contrastive_trace_sample(backend, _image(root, row), row, yes_id=ids["yes"], no_id=ids["no"],
                                          sigma=sigma, seed=rc["seed"],
                                          min_margin_gap=rc["trace"]["min_margin_gap"])
        results.append(result)
        _flush(paths["trace"], results, 10, force=i == len(pending))
        print(f"Trace {len(results)}/{len(rows)} {row['sample_id']} {result['status']}", flush=True)
    return {"completed": len(results), "expected": len(rows),
            "status": dict(Counter(r["status"] for r in results))}


# ---------------------------------------------------------------------------
# Stage 3: layer selection
# ---------------------------------------------------------------------------

def run_select(rc, base, repo, paths):
    rows = read_jsonl(paths["contrastive"])
    records = read_jsonl(paths["trace"])
    if {r["sample_id"] for r in records} != {r["sample_id"] for r in rows}:
        raise ValueError("Contrastive tracing is incomplete; finish the trace stage first")
    sel = rc["selection"]
    k1, k2 = sel["k_source"], sel["k_target"]
    plans, diagnostics = {}, {"status": dict(Counter(r["status"] for r in records))}
    for component in COMPONENTS:
        image_ids, N, D, C = image_tables(records, component, NUM_LAYERS)
        diag = {"images": len(image_ids),
                "questions_per_class": {c: int(C[:, i].sum()) for i, c in enumerate(("yes", "no"))}}
        for name, classes in (("both", ("yes", "no")), ("present_only", ("yes",))):
            point = ratio_scores(N, D, C, classes=classes, combine=sel["class_combination"])
            boot = bootstrap_scores(N, D, C, draws=sel["bootstrap"], seed=rc["seed"], classes=classes,
                                    combine=sel["class_combination"])
            diag[name] = {"score": point.tolist(), "boot": boot}
        paper_rr, support = paper_rr_profile(records, component, NUM_LAYERS, sel["paper_min_probability_gap"])
        diag["faithful_rr"] = {"score": paper_rr.tolist(), "images": support}
        diagnostics[component] = diag

    def per_component(fn):
        return {c: fn(c) for c in COMPONENTS}

    for variant in rc["variants"]:
        if variant == "paper_iri":
            profile = repo / base["output_root"] / "profile.json"
            if not profile.is_file():
                raise ValueError("paper_iri needs the FCCT reconstruction profile (python fcct.py trace/profile)")
            rr = json.loads(profile.read_text(encoding="utf-8"))["rr"]
            plans[variant] = paper_plan(rr, base["k_source"], base["k_target"])
            plans[variant]["_source"] = {"profile_sha256": sha256(profile)}
        elif variant == "faithful_rr":
            plans[variant] = paper_plan({c: diagnostics[c]["faithful_rr"]["score"] for c in COMPONENTS}, k1, k2)
        elif variant in ("riri", "riri_present_only"):
            key = "both" if variant == "riri" else "present_only"
            plans[variant] = per_component(lambda c: stable_plan(
                np.asarray(diagnostics[c][key]["score"]), diagnostics[c][key]["boot"], k1, k2,
                alpha=sel["alpha"], reachable=True))
        elif variant == "riri_no_stability":
            def plain(c):
                s = np.asarray(diagnostics[c]["both"]["score"])
                return reachable_plan(s, k1, k2, weights=np.clip(s, 0, None),
                                      eligible_sources=[l for l in range(NUM_LAYERS) if s[l] > 0])
            plans[variant] = per_component(plain)
        else:
            raise ValueError(f"Unknown variant: {variant}")
    # Stability of each plan's sources under the R-FCCT bootstrap.
    stability = {}
    for variant, plan in plans.items():
        stability[variant] = {}
        for c in COMPONENTS:
            freq = inclusion_frequency(diagnostics[c]["both"]["boot"], k1)
            stability[variant][c] = {str(s): float(freq[s]) for s in plan[c]["sources"]}
    for c in COMPONENTS:
        for key in ("both", "present_only"):
            boot = diagnostics[c][key].pop("boot")
            diagnostics[c][key]["ci95"] = np.quantile(boot, [0.025, 0.975], axis=0).tolist()
    clean_plans = {v: {c: {k: p[c][k] for k in ("sources", "targets", "weights")} for c in COMPONENTS}
                   for v, p in plans.items()}
    out = {"extension_version": EXTENSION_VERSION, "trace_sha256": sha256(paths["trace"]),
           "plans": clean_plans, "effective_edges": {v: effective_edges(p) for v, p in clean_plans.items()},
           "source_stability_under_rfcct_bootstrap": stability, "diagnostics": diagnostics}
    write_json(paths["plans"], out)
    return {"plans": clean_plans}


def _load_plans(paths):
    data = json.loads(paths["plans"].read_text(encoding="utf-8"))
    if data["trace_sha256"] != sha256(paths["trace"]):
        raise ValueError("Plans do not belong to the current trace; rerun select")
    return data["plans"]


# ---------------------------------------------------------------------------
# Stage 4: held-out tuning of injection strength
# ---------------------------------------------------------------------------

def grid_keys(rc):
    keys = []
    for variant in rc["tuning"]["variants"]:
        for la in rc["tuning"]["attention_grid"]:
            for lm in rc["tuning"]["mlp_grid"]:
                if la == 0 and lm == 0:
                    continue  # identical to regular decoding
                keys.append(f"{variant}|{la}|{lm}")
    return keys


def run_tune(backend, ids, rc, base, repo, paths, limit=None):
    rows = read_jsonl(paths["calibration"])
    plans = _load_plans(paths)
    keys = grid_keys(rc)
    settings = {"stage": "tune", "tuning": rc["tuning"], "model": [base["model_id"], base["revision"]]}
    results = open_run(paths["tune"], {"manifest": paths["calibration"], "plans": paths["plans"]},
                       settings, repo, [r["sample_id"] for r in rows])
    done = {r["sample_id"] for r in results}
    pending = [r for r in rows if r["sample_id"] not in done][:limit]
    operators = {}
    for key in keys:
        variant, la, lm = key.split("|")
        operators[key] = PlannedIRI(backend.layers, plans[variant], attention_scale=float(la),
                                    mlp_scale=float(lm), normalize=base["normalize"])
    root = repo / base["data_root"]
    for i, row in enumerate(pending, 1):
        batch, _ = backend.batch(_image(root, row), row["question"])
        embeds = backend.trace_batch(batch)  # vision tower is untouched by IRI hooks
        margins = {"regular": score(backend.model, embeds, ids["yes"], ids["no"])["margin"]}
        for key, operator in operators.items():
            with operator.apply():
                margins[key] = score(backend.model, embeds, ids["yes"], ids["no"])["margin"]
        results.append({"sample_id": row["sample_id"], "image_id": row["image_id"], "label": row["label"],
                        "margins": margins})
        _flush(paths["tune"], results, 10, force=i == len(pending))
        print(f"Tune {len(results)}/{len(rows)} {row['sample_id']}", flush=True)
    return {"completed": len(results), "expected": len(rows)}


def freeze(rc, base, paths):
    """Choose strengths and threshold controls on calibration data only."""
    rows = read_jsonl(paths["calibration"])
    results = read_jsonl(paths["tune"])
    if {r["sample_id"] for r in results} != {r["sample_id"] for r in rows}:
        raise ValueError("Tuning is incomplete; finish the tune stage first")
    labels = np.array([r["label"] == "yes" for r in results], int)
    regular_margin = np.array([r["margins"]["regular"] for r in results])
    regular = confusion(labels, regular_margin > 0)
    floor = regular["recall"] - rc["tuning"]["max_recall_drop_pp"]
    frozen = {"calibration_sha256": sha256(paths["calibration"]), "tune_sha256": sha256(paths["tune"]),
              "plans_sha256": sha256(paths["plans"]), "regular": regular, "recall_floor": floor,
              "methods": {}, "grid": {}}
    for variant in rc["tuning"]["variants"]:
        candidates = [(0.0, 0.0, regular)]
        for key in grid_keys(rc):
            v, la, lm = key.split("|")
            if v != variant:
                continue
            stats = confusion(labels, np.array([r["margins"][key] for r in results]) > 0)
            frozen["grid"][key] = stats
            candidates.append((float(la), float(lm), stats))
        feasible = [c for c in candidates if c[2]["recall"] >= floor - 1e-9]
        la, lm, stats = max(feasible, key=lambda c: (c[2]["accuracy"], -(c[0] + c[1]), -c[0]))
        tau_rate = threshold_for_rate(regular_margin, stats["yes_rate"] / 100)
        tau_best = best_threshold(regular_margin, labels, min_recall=floor)
        frozen["methods"][f"{variant}@tuned"] = {
            "variant": variant, "attention_scale": la, "mlp_scale": lm, "calibration": stats,
            "is_identity": la == 0 and lm == 0,
            "threshold_rate": tau_rate, "threshold_best": tau_best}
    tau_best = best_threshold(regular_margin, labels, min_recall=floor)
    frozen["threshold_control"] = {"tau": tau_best,
                                   "calibration": confusion(labels, regular_margin > tau_best)}
    write_json(paths["frozen"], frozen)
    return {k: {"attention_scale": v["attention_scale"], "mlp_scale": v["mlp_scale"],
                "calibration_accuracy": v["calibration"]["accuracy"]} for k, v in frozen["methods"].items()}


# ---------------------------------------------------------------------------
# Stage 5/6: test evaluation (greedy) and the VCD sampling protocol
# ---------------------------------------------------------------------------

def _methods(rc, base, paths, names):
    plans = _load_plans(paths)
    frozen = json.loads(paths["frozen"].read_text(encoding="utf-8")) if paths["frozen"].is_file() else None
    if frozen and (frozen["plans_sha256"] != sha256(paths["plans"])):
        raise ValueError("Frozen strengths belong to different plans; rerun tune/freeze")
    methods = {}
    for name in names:
        if name == "regular":
            methods[name] = None
        elif name.endswith("@paper"):
            variant = name[:-len("@paper")]
            methods[name] = (variant, base["attention_scale"], base["mlp_scale"])
        elif name.endswith("@tuned"):
            if frozen is None:
                raise ValueError("Run tune/freeze before evaluating tuned methods")
            m = frozen["methods"][name]
            methods[name] = (m["variant"], m["attention_scale"], m["mlp_scale"])
        else:
            raise ValueError(f"Unknown method: {name}")
    return {name: None if spec is None else (spec, plans[spec[0]]) for name, spec in methods.items()}


def prompt_key(row):
    return fingerprint([row["image_sha256"], row["question"]])[:24]


def unique_prompts(rows):
    seen = {}
    for row in rows:
        seen.setdefault(prompt_key(row), row)
    return seen


@torch.inference_mode()
def generate(backend, image, question, *, max_new_tokens, yes_id, no_id, sampling=None, seed=None):
    batch, _ = backend.batch(image, question)
    length = batch["input_ids"].shape[1]
    if sampling is None:
        options = {"do_sample": False, "num_beams": 1}
    else:
        torch.manual_seed(seed)
        options = {"do_sample": True, "num_beams": 1, "temperature": sampling["temperature"],
                   "top_p": sampling["top_p"], "top_k": sampling["top_k"]}
    output = backend.model.generate(**batch, max_new_tokens=max_new_tokens, use_cache=True,
                                    return_dict_in_generate=True, output_logits=True, **options)
    first = output.logits[0][0].float()  # raw (unprocessed) first-step logits
    text = backend.processor.tokenizer.decode(output.sequences[0, length:], skip_special_tokens=True).strip()
    return {"text": text, "margin": float(first[yes_id] - first[no_id])}


def _operators(backend, base, methods):
    ops = {}
    for name, spec in methods.items():
        if spec is None:
            ops[name] = None
        else:
            (variant, la, lm), plan = spec
            ops[name] = PlannedIRI(backend.layers, plan, attention_scale=la, mlp_scale=lm,
                                   normalize=base["normalize"])
    return ops


def run_evaluate(backend, ids, rc, base, repo, paths, limit=None, *, sampling=False):
    rows = read_jsonl(paths["test"])
    prompts = unique_prompts(rows)
    cfg = rc["sampling"] if sampling else rc["evaluation"]
    names = cfg["methods"]
    methods = _methods(rc, base, paths, names)
    seeds = cfg["seeds"] if sampling else [None]
    output = paths["sample"] if sampling else paths["evaluate"]
    inputs = {"manifest": paths["test"], "plans": paths["plans"]}
    if paths["frozen"].is_file():
        inputs["frozen"] = paths["frozen"]
    settings = {"stage": "sample" if sampling else "evaluate", "config": cfg,
                "max_new_tokens": rc["max_new_tokens"], "model": [base["model_id"], base["revision"]],
                "methods": {n: None if s is None else s[0] for n, s in methods.items()}}
    keys = [f"{k}|{s}" for k in prompts for s in seeds]
    results = open_run(output, inputs, settings, repo, keys)
    done = {r["key"] for r in results}
    pending = [k for k in keys if k not in done][:limit]
    ops = _operators(backend, base, methods)
    root = repo / base["data_root"]
    images = {}
    for i, key in enumerate(pending, 1):
        prompt, seed_text = key.split("|")
        row = prompts[prompt]
        if row["image"] not in images:
            images.clear()
            images[row["image"]] = _image(root, row)
        answers = {}
        for name, op in ops.items():
            seed = None if seed_text == "None" else int(fingerprint([int(seed_text), prompt])[:8], 16)
            with op.apply() if op is not None else nullcontext():
                answers[name] = generate(backend, images[row["image"]], row["question"],
                                         max_new_tokens=rc["max_new_tokens"], yes_id=ids["yes"],
                                         no_id=ids["no"], sampling=cfg if sampling else None, seed=seed)
        results.append({"key": key, "prompt": prompt, "seed": seed_text, "answers": answers})
        _flush(output, results, 10, force=i == len(pending))
        print(f"{'Sample' if sampling else 'Evaluate'} {len(results)}/{len(keys)}", flush=True)
    return {"completed": len(results), "expected": len(keys), "unique_prompts": len(prompts)}


def expand(rows, results):
    """Map unique-prompt generations back onto every POPE row."""
    by_key = {}
    for r in results:
        by_key.setdefault(r["seed"], {})[r["prompt"]] = r["answers"]
    out = {}
    for seed, answers in by_key.items():
        expanded = []
        for row in rows:
            a = answers.get(prompt_key(row))
            if a is None:
                raise ValueError("Evaluation is incomplete; finish the stage before reporting")
            expanded.append({**row, "answers": a})
        out[seed] = expanded
    return out


def predictions(rows, method, *, threshold=None):
    """Binary predictions: POPE text parser, or a margin threshold on regular decoding."""
    if threshold is None:
        return np.array([pope_answer(r["answers"][method]["text"]) == "yes" for r in rows], int)
    return np.array([r["answers"]["regular"]["margin"] > threshold for r in rows], int)
