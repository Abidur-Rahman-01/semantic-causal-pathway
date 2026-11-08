"""Figure 4 attention analysis with explicit, separately saved normalizations."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import validate_manifest
from .experiment import open_run
from .io import read_jsonl, sha256, write_json, write_jsonl


def attention_statistics(attentions, visual_positions, text_positions):
    if attentions is None or not attentions:
        raise ValueError("Attention weights unavailable; an eager decoder with output_attentions=True is required")
    if (not visual_positions or not text_positions
            or len(set(visual_positions)) != len(visual_positions)
            or len(set(text_positions)) != len(text_positions)
            or set(visual_positions) & set(text_positions)):
        raise ValueError("Visual/textual object positions must be nonempty and disjoint")
    layers = []
    for layer, attention in enumerate(attentions):
        if attention is None or attention.ndim != 4 or attention.shape[0] != 1:
            raise ValueError("Expected batch-one [batch, heads, query, key] attention tensors")
        positions = visual_positions + text_positions
        if min(positions) < 0 or max(positions) >= attention.shape[-1]:
            raise ValueError("Object token position outside the attention key sequence")
        last = attention[0, :, -1, :].float()
        if (not torch.isfinite(last).all() or (last < 0).any()
                or not torch.allclose(last.sum(dim=-1), torch.ones(last.shape[0], device=last.device), atol=0.01)):
            raise ValueError("Expected normalized, finite attention probabilities in evaluation mode")
        # Sum within each token group, then average across heads.
        visual = float(last[:, visual_positions].sum(dim=-1).mean())
        text = float(last[:, text_positions].sum(dim=-1).mean())
        denominator = visual + text
        layers.append({"layer": layer, "visual_mass": visual, "text_mass": text,
                       "visual_per_token": visual / len(visual_positions),
                       "text_per_token": text / len(text_positions),
                       "visual_object_relative": visual / denominator if denominator > 0 else None,
                       "text_object_relative": text / denominator if denominator > 0 else None,
                       "visual_token_count": len(visual_positions), "text_token_count": len(text_positions)})
    return layers


def summarize_attention(rows, expected_questions):
    valid = [r for r in rows if r["status"] == "complete"]
    if not valid:
        raise ValueError("No valid attention samples to summarize")
    n_layers = len(valid[0]["layers"])
    if any(len(r["layers"]) != n_layers for r in valid):
        raise ValueError("Attention records have inconsistent decoder layer counts")
    metrics = ("visual_mass", "text_mass", "visual_per_token", "text_per_token",
               "visual_object_relative", "text_object_relative")
    curves = {}
    for metric in metrics:
        curves[metric] = []
        for layer in range(n_layers):
            values = [r["layers"][layer][metric] for r in valid if r["layers"][layer][metric] is not None]
            curves[metric].append(float(np.mean(values)) if values else None)
    complete = len(rows) == expected_questions
    return {"status": "independent_attention_analysis", "exact_reproduction": False,
            "all_questions_processed": complete, "paper_sample_count_met": expected_questions == 3000 and complete,
            "processed_questions": len(rows), "expected_questions": expected_questions,
            "valid_questions": len(valid), "sample_status_counts": dict(Counter(r["status"] for r in rows)),
            "aggregation": "question mean; attention heads averaged within each question",
            "normalizations": {"mass": "full-softmax attention mass summed over object tokens",
                               "per_token": "object-group mass divided by number of object tokens",
                               "object_relative": "group mass divided by visual-plus-textual object mass"},
            "author_normalization_verified": False, "curves": curves}


def run_attention(backend, config, repo, manifest: Path, output_dir: Path, *, limit=None, allow_partial=False):
    samples = read_jsonl(manifest)
    if len(samples) != 3000 and not allow_partial:
        raise ValueError("Figure 4 reports 3000 VQAs. Supply that manifest, or use --allow-partial for independent analysis")
    validate_manifest(samples, repo / config["data_root"], "trace")
    output = output_dir / "attention.jsonl"
    rows = open_run(output, manifest, config, repo, {"stage": "attention", "allow_partial": allow_partial})
    done = {r["sample_id"] for r in rows}
    pending = [r for r in samples if r["sample_id"] not in done]
    if limit is not None:
        pending = pending[:limit]
    for row in pending:
        with Image.open(repo / config["data_root"] / row["image"]) as image:
            try:
                values = backend.attention_statistics(image.convert("RGB"), row)
                record = {"sample_id": row["sample_id"], "image_id": row["image_id"], "status": "complete", **values}
            except ValueError as error:
                if "Object must have visible" not in str(error):
                    raise
                record = {"sample_id": row["sample_id"], "image_id": row["image_id"],
                          "status": "excluded_object_outside_crop", "reason": str(error)}
        rows.append(record)
        write_jsonl(output, rows)
        print(f"Attention {len(rows)}/{len(samples)}: {row['sample_id']}", flush=True)
    summary = summarize_attention(rows, len(samples))
    summary["manifest_sha256"] = sha256(manifest)
    summary["records_sha256"] = sha256(output)
    write_json(output_dir / "attention_summary.json", summary)
    lines = ["# Last-token attention to visual and textual objects", "",
             "Independent Figure 4 analysis. The original 3,000 questions and plotted normalization are unverified.", "",
             f"Processed {len(rows)}/{len(samples)} questions; {summary['valid_questions']} valid.", "",
             "| Layer | Visual mass | Textual mass | Visual per token | Textual per token |",
             "|---|---:|---:|---:|---:|"]
    for layer in range(len(summary["curves"]["visual_mass"])):
        values = [summary["curves"][key][layer] for key in ("visual_mass", "text_mass", "visual_per_token", "text_per_token")]
        lines.append(f"| {layer} | " + " | ".join(f"{v:.6g}" for v in values) + " |")
    (output_dir / "attention_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(summary["curves"]["visual_mass"], label="Visual object tokens")
    ax.plot(summary["curves"]["text_mass"], label="Textual object tokens")
    ax.set(xlabel="Decoder layer (zero based)", ylabel="Mean full-softmax attention mass",
           title="Independent attention analysis; author normalization unverified")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "object_attention.png", dpi=180)
    plt.close(fig)
    return {key: summary[key] for key in ("status", "exact_reproduction", "processed_questions", "valid_questions")}
