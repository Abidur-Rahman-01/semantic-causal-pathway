"""Paper-to-code coverage audit; implemented equations are not reproduced results."""
from __future__ import annotations

from pathlib import Path

from . import IMPLEMENTATION_VERSION
from .experiment import code_hashes
from .io import sha256, write_json


def architecture_audit(config, repo: Path):
    items = [
        ("Equation 1: residual stream", "implemented", "Uses the pretrained LLaVA decoder; component hooks observe projected MHSA, MLP, and decoder-layer outputs.", "src/fcct/core.py"),
        ("Equation 2: recovery rate", "implemented", "Full-vocabulary affirmative next-token probability; signed nonzero denominators by default; no clipping. Zero denominators are undefined.", "src/fcct/core.py"),
        ("Equations 3-5: IRI", "implemented", "Independent component rankings, strict source < target ordering, last-token source outputs, RR weighting and per-vector L2 norm preservation.", "src/fcct/core.py"),
        ("LLaVA main hyperparameters", "implemented", "k1=3, k2=10, attention scale=0.26, MLP scale=0.16.", "configs/fcct_llava15.json"),
        ("Figure 3: component/layer/category tracing", "implemented_with_unverified_inputs", "Three components, 32 layers, seven categories; original sample list, grounding and aggregation are not available.", "src/fcct/experiment.py"),
        ("Seven category assignments", "independent_interpretation", "COCO boxes mapped through the CLIP crop; unassigned visual/text positions are recorded. Exact assignments can be supplied with an encoded-token checksum.", "src/fcct/llava.py"),
        ("Figure 4: last-token object attention", "implemented_with_unverified_inputs", "Saves attention mass, per-token and object-relative values. Requires a separate 3000-question manifest for the reported sample count; author normalization is unspecified.", "src/fcct/attention.py"),
        ("Table 2: LLaVA regular and IRI POPE", "implemented_with_unverified_protocol", "Original 9000-question COCO POPE. The paper's dataset/run aggregation, prompts and decoding snapshot remain unverified.", "src/fcct/report.py"),
        ("Table 6: named controls", "partially_implemented", "12 supported settings with the printed coefficients; first/last-ten and equal-weight controls state their interpretation. Hidden-state IRI is not invented.", "configs/fcct_ablation_plan.json"),
        ("Qwen-VL-Chat, LLaVA-NeXT, Qwen2-VL-7B, InternVL2-8B", "not_implemented", "Dedicated checkpoint-specific adapters, tracing and evaluation are still required. Qwen2.5-VL is not a replacement.", None),
        ("Table 1: MME", "not_implemented", "Official assets, paired-question evaluator and all model/method results are required.", None),
        ("Table 3: CHAIR", "not_implemented", "Original image subset and caption prompts plus original CHAIR evaluator are required; max_new_tokens=512 is reported.", None),
        ("Table 4: MMHal-Bench", "blocked_external_and_not_implemented", "96 questions plus the original GPT-4 judge version, instructions and outputs are required.", None),
        ("Table 4: MHumanEval", "blocked_external_and_not_implemented", "146 examples and three actual human annotators; automated scores cannot replace this experiment.", None),
        ("VCD, OPERA, PAI and VTI baselines", "not_implemented", "Original baseline implementations and matched decoding/evaluation are required.", None),
        ("Table 7: TTFT/TPOT", "not_implemented", "Matched measurement protocol and original 8 x A100 80GB hardware experiment are unavailable.", None),
    ]
    missing = [
        "Original 500-image tracing subset, question list, object regions and exact token-category assignments",
        "Gaussian noise domain, standard deviation, clipping, RNG and corruption-selection protocol",
        "Original checkpoint/backend revisions, prompt templates, affirmative token ID and decoding settings",
        "RR aggregation, numerical filtering and original per-layer RR profiles",
        "Original separate 3000-VQA attention manifest and attention normalization",
        "IRI source-cache and prefill/decoding intervention details; hidden-state ablation semantics",
        "Full benchmark subsets, baseline configurations, judge snapshot and human annotations",
    ]
    return {"status": "partial_independent_reconstruction", "exact_reproduction_ready": False,
            "numerical_reproduction_verified": False, "implementation_version": IMPLEMENTATION_VERSION,
            "paper_source": "User-supplied AAAI-26 text and arXiv:2511.05923v3",
            "author_repository": "https://github.com/QimingLi-scir/FCCT-IRI",
            "author_repository_observation": "Empty on public web inspection 2026-11-08",
            "items": [{"paper_item": item, "status": status, "finding": finding, "file": file}
                      for item, status, finding, file in items],
            "required_original_artifacts": missing,
            "current_config": {k: config[k] for k in ("model_id", "revision", "noise_sigma", "yes_token",
                                   "min_probability_gap", "k_source", "k_target", "attention_scale", "mlp_scale")},
            "code_sha256": code_hashes(repo),
            "ablation_plan_sha256": sha256(repo / "configs/fcct_ablation_plan.json")}


def write_architecture_audit(config, repo: Path, output_dir: Path):
    audit = architecture_audit(config, repo)
    write_json(output_dir / "fcct_architecture_audit.json", audit)
    lines = ["# FCCT/IRI paper-to-code architecture audit", "",
             "**Exact numerical reproduction has not been achieved.** This audit checks implementation coverage; it does not substitute tests for paper experiments.", "",
             "Source: the supplied AAAI-26 conference text and https://arxiv.org/abs/2511.05923v3.", "",
             "| Paper item | Coverage | Finding | Code |", "|---|---|---|---|"]
    for row in audit["items"]:
        file = f"`{row['file']}`" if row["file"] else "-"
        lines.append(f"| {row['paper_item']} | {row['status']} | {row['finding']} | {file} |")
    lines += ["", "## Original artifacts still required", ""]
    lines += [f"- {item}" for item in audit["required_original_artifacts"]]
    lines += ["", "## Verified corrections in version 0.2", "",
              "- Equation 2 now retains negative denominators. Only a zero denominator is undefined by default; positive-only filtering is explicit and optional.",
              "- Profile files expose image-weighted and question-weighted RR averages and their supports, because the paper does not state its aggregation.",
              "- Token coverage records unassigned positions; externally supplied category assignments require an exact encoded-token checksum.",
              "- Figure 4 analysis saves multiple explicit attention normalizations instead of claiming an undocumented one is the authors' choice.",
              "- Table 6 controls freeze the printed coefficients; the ambiguous hidden-state variant is explicitly unsupported.",
              "- Profiles from old tracing semantics or incompatible noise/token/aggregation settings are rejected.", "",
              "The original author repository is empty as checked on 2026-11-08. Numerical targets must never be copied into measured outputs."]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fcct_architecture_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit
