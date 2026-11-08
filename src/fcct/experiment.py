"""Resumable tracing, RR profile fitting, and paired POPE generation."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import IMPLEMENTATION_VERSION
from .core import COMPONENTS, TOKEN_GROUPS, IRI, capture, patch_component, recovery_rate
from .data import validate_manifest
from .io import environment, fingerprint, read_jsonl, sha256, write_json, write_jsonl
from .llava import gaussian_corruption, token_coverage


def deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def code_hashes(repo):
    paths = list((repo / "src/fcct").glob("*.py")) + [repo / "fcct.py", repo / "scripts/setup_fcct.py"]
    return {str(p.relative_to(repo)).replace("\\", "/"): sha256(p) for p in sorted(paths)}


def open_run(output: Path, manifest: Path, config, repo, extra=None):
    provenance = {"implementation_version": IMPLEMENTATION_VERSION, "config": config,
                  "manifest_sha256": sha256(manifest), "code_sha256": code_hashes(repo),
                  "environment": environment(), "extra": extra or {}}
    signature = fingerprint(provenance)
    meta = output.with_suffix(".meta.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not meta.exists():
        raise ValueError("Results have no provenance sidecar; use a new output path")
    if meta.exists():
        previous = json.loads(meta.read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise ValueError("Resume rejected: code/config/manifest/environment/profile changed; use a new output path")
    else:
        write_json(meta, {"signature": signature, **provenance})
    existing = read_jsonl(output) if output.exists() else []
    if len({r["sample_id"] for r in existing}) != len(existing):
        raise ValueError("Duplicate sample IDs in prior results")
    allowed_ids = {r["sample_id"] for r in read_jsonl(manifest)}
    if any(r["sample_id"] not in allowed_ids for r in existing):
        raise ValueError("Results contain unexpected sample IDs")
    return existing


def trace_sample(backend, image, row, config):
    clean_batch, spans = backend.batch(image, row["question"], row["object"])
    try:
        groups = backend.groups(image, row, clean_batch, spans)
    except ValueError as error:
        if "Object must have visible" not in str(error):
            raise
        return {"sample_id": row["sample_id"], "image_id": row["image_id"],
                "status": "excluded_object_outside_crop", "reason": str(error), "patches": []}
    # Stable per-sample RNG is independent of execution/resume order.
    noise_seed = int(fingerprint([config["seed"], row["sample_id"]])[:16], 16)
    noisy_image = gaussian_corruption(image, config["noise_sigma"], noise_seed)
    corrupted_batch, _ = backend.batch(noisy_image, row["question"])
    if not torch.equal(clean_batch["input_ids"], corrupted_batch["input_ids"]):
        raise ValueError("Clean/corrupted token alignment differs")
    clean_batch = backend.trace_batch(clean_batch)
    corrupted_batch = backend.trace_batch(corrupted_batch)
    with capture(backend.layers, groups) as (cache, positions):
        clean = backend.probability(clean_batch)
    corrupted = backend.probability(corrupted_batch)
    result = {"sample_id": row["sample_id"], "image_id": row["image_id"],
              "noise_seed": noise_seed, "p_clean": clean, "p_corrupted": corrupted,
              "probability_gap": clean - corrupted, "yes_token_id": backend.yes_token_id,
              "rr_gap_policy": config.get("rr_gap_policy", "nonzero"),
              "token_coverage": token_coverage(groups, spans["image_positions"], len(clean_batch["attention_mask"][0])),
              "token_groups": groups, "status": "complete", "patches": []}
    policy = config.get("rr_gap_policy", "nonzero")
    if recovery_rate(clean, corrupted, corrupted, config["min_probability_gap"], policy) is None:
        result["status"] = "excluded_undefined_or_filtered_gap"
        return result
    lookup = {p: i for i, p in enumerate(positions)}
    for layer_index, layer in enumerate(backend.layers):
        for component in COMPONENTS:
            for group in TOKEN_GROUPS:
                indices = groups[group]
                if indices:
                    values = cache[layer_index, component][:, [lookup[p] for p in indices], :]
                    with patch_component(layer, component, indices, values):
                        patched = backend.probability(corrupted_batch)
                    rr = recovery_rate(clean, corrupted, patched, config["min_probability_gap"], policy)
                else:
                    patched, rr = None, None
                result["patches"].append({"layer": layer_index, "component": component,
                                           "group": group, "p_patched": patched, "rr": rr})
    return result


def run_trace(backend, config, repo, manifest: Path, output: Path, limit=None):
    root = repo / config["data_root"]
    rows = read_jsonl(manifest)
    validate_manifest(rows, root, "trace")
    if len({r["image_id"] for r in rows}) != config["trace_images"]:
        raise ValueError("Manifest image count differs from config.trace_images")
    results = open_run(output, manifest, config, repo, {"stage": "trace"})
    done = {r["sample_id"] for r in results}
    pending = [r for r in rows if r["sample_id"] not in done]
    if limit is not None:
        pending = pending[:limit]
    for row in pending:
        with Image.open(root / row["image"]) as source:
            result = trace_sample(backend, source.convert("RGB"), row, config)
        results.append(result)
        # Commit whole sample atomically. A failed patch never becomes a result.
        write_jsonl(output, results)
        print(f"Trace {len(results)}/{len(rows)}: {row['sample_id']} {result['status']}", flush=True)
    return {"completed_questions": len(results), "expected_questions": len(rows),
            "all_questions_processed": len(results) == len(rows), "output": str(output)}


def fit_profile(trace: Path, manifest: Path, output: Path, allow_partial=False):
    rows, samples = read_jsonl(trace), read_jsonl(manifest)
    by_id = {r["sample_id"]: r for r in samples}
    if len(by_id) != len(samples):
        raise ValueError("Duplicate sample IDs in tracing manifest")
    expected = {r["sample_id"] for r in samples}
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate trace sample IDs")
    if not {r["sample_id"] for r in rows} <= expected:
        raise ValueError("Trace includes unknown samples")
    complete = {r["sample_id"] for r in rows} == expected
    if not complete and not allow_partial:
        raise ValueError("Tracing is incomplete; --allow-partial is for engineering checks only")
    meta_path = trace.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("implementation_version", IMPLEMENTATION_VERSION) != IMPLEMENTATION_VERSION:
        raise ValueError("Trace implementation version changed; collect fresh traces with the current RR definition")
    if meta["manifest_sha256"] != sha256(manifest):
        raise ValueError("Trace manifest differs from supplied manifest")
    aggregation = meta["config"].get("rr_aggregation", "image_mean")
    if aggregation not in ("image_mean", "question_mean"):
        raise ValueError("rr_aggregation must be image_mean or question_mean")
    grouped = defaultdict(lambda: defaultdict(list))
    statuses = defaultdict(int)
    gap_counts = defaultdict(int)
    for row in rows:
        if row["image_id"] != by_id[row["sample_id"]]["image_id"]:
            raise ValueError("Trace image IDs differ from manifest")
        if row["status"] not in ("complete", "excluded_object_outside_crop", "excluded_undefined_or_filtered_gap"):
            raise ValueError("Unknown tracing status")
        if "p_clean" in row:
            gap = row["p_clean"] - row["p_corrupted"]
            gap_counts["positive" if gap > 0 else "negative" if gap < 0 else "zero"] += 1
        if row["status"] == "complete":
            expected_cells = {(c, g, l) for c in COMPONENTS for g in TOKEN_GROUPS for l in range(32)}
            cells = {(p["component"], p["group"], p["layer"]) for p in row["patches"]}
            if len(row["patches"]) != len(expected_cells) or cells != expected_cells:
                raise ValueError("A completed trace must contain all component/group/layer cells")
        statuses[row["status"]] += 1
        for cell in row["patches"]:
            if cell["rr"] is not None:
                grouped[cell["component"], cell["group"], cell["layer"]][row["image_id"]].append(cell["rr"])
    curves, counts, question_counts, alternatives = {}, {}, {}, {}
    for component in COMPONENTS:
        curves[component], counts[component], question_counts[component], alternatives[component] = {}, {}, {}, {}
        for group in TOKEN_GROUPS:
            values, support, question_support, image_means, question_means = [], [], [], [], []
            for layer in range(32):
                by_image = grouped[component, group, layer]
                image_mean = float(np.mean([np.mean(v) for v in by_image.values()])) if by_image else None
                question_values = [v for group_values in by_image.values() for v in group_values]
                question_mean = float(np.mean(question_values)) if question_values else None
                image_means.append(image_mean)
                question_means.append(question_mean)
                values.append(image_mean if aggregation == "image_mean" else question_mean)
                support.append(len(by_image))
                question_support.append(len(question_values))
            curves[component][group], counts[component][group] = values, support
            question_counts[component][group] = question_support
            alternatives[component][group] = {"image_mean": image_means, "question_mean": question_means}
    rr = {c: curves[c]["last"] for c in ("attention", "mlp")}
    if any(v is None for values in rr.values() for v in values):
        raise ValueError("No eligible last-token RRs for one or more layers; cannot invent IRI weights")
    profile = {"status": "independent_reconstruction", "complete": complete,
               "implementation_version": IMPLEMENTATION_VERSION,
               "trace_image_ids": sorted({r["image_id"] for r in samples}),
               "source_trace_sha256": sha256(trace), "source_meta_sha256": sha256(meta_path),
               "model_id": meta["config"]["model_id"], "revision": meta["config"]["revision"],
               "trace_config": meta["config"], "curves": curves, "image_support": counts,
               "question_support": question_counts, "aggregation_sensitivity": alternatives,
               "probability_gap_counts": dict(gap_counts),
               "rr": rr, "sample_status_counts": dict(statuses),
               "processed_questions": len(rows), "expected_questions": len(samples),
               "aggregation": aggregation, "rr_clipping": False}
    write_json(output, profile)
    return {k: profile[k] for k in ("status", "complete", "processed_questions", "sample_status_counts")}


def load_iri(backend, profile_path: Path, config, allow_partial=False):
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile["model_id"] != config["model_id"] or profile["revision"] != config["revision"]:
        raise ValueError("RR profile belongs to a different model/revision")
    if profile.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise ValueError("RR profile predates the current tracing semantics; regenerate traces/profile")
    trace_keys = ("seed", "yes_token", "expected_yes_token_id", "noise_sigma", "min_probability_gap",
                  "rr_gap_policy", "rr_aggregation")
    if any(profile["trace_config"].get(key) != config.get(key) for key in trace_keys):
        raise ValueError("RR profile uses different tracing settings; regenerate or select the matching configuration")
    if not profile["complete"] and not allow_partial:
        raise ValueError("Incomplete RR profile; --allow-partial is only for engineering checks")
    return IRI(backend.layers, profile["rr"], k_source=config["k_source"], k_target=config["k_target"],
               attention_scale=config["attention_scale"], mlp_scale=config["mlp_scale"],
               normalize=config["normalize"], allowed_layers=config.get("iri_allowed_layers"),
               rr_weighting=config.get("rr_weighting", True))


def run_evaluation(backend, config, repo, manifest, output, profile_path, kind="pope", limit=None,
                   allow_partial=False):
    root = repo / config["data_root"]
    rows = read_jsonl(manifest)
    validate_manifest(rows, root, kind)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if "trace_image_ids" not in profile:
        raise ValueError("RR profile lacks tracing image IDs")
    trace_images = set(profile["trace_image_ids"])
    if any(r["image_id"] in trace_images for r in rows):
        raise ValueError("Evaluation images overlap profile tracing images")
    # Normal profiles also retain the full tracing manifest's IDs below.
    iri = load_iri(backend, profile_path, config, allow_partial)
    results = open_run(output, manifest, config, repo, {"stage": "evaluate", "kind": kind,
                       "profile_sha256": sha256(profile_path), "allow_partial": allow_partial})
    done = {r["sample_id"] for r in results}
    pending = [r for r in rows if r["sample_id"] not in done]
    if limit is not None:
        pending = pending[:limit]
    tokens = config["caption_max_new_tokens"] if kind == "caption" else config["binary_max_new_tokens"]
    for row in pending:
        answers = {}
        with Image.open(root / row["image"]) as source:
            image = source.convert("RGB")
        for method in ("regular", "iri"):
            with iri.apply() if method == "iri" else nullcontext():
                answers[method] = backend.generate(image, row["question"], tokens)
        results.append({**row, "answers": answers})
        write_jsonl(output, results)
        print(f"Evaluate {len(results)}/{len(rows)}: {row['sample_id']}", flush=True)
    return {"completed_questions": len(results), "expected_questions": len(rows),
            "all_questions_processed": len(results) == len(rows), "output": str(output)}
