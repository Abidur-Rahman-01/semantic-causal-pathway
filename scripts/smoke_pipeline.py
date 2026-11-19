"""Real-model integration check for Parts B and C (minutes, not days).

Run on the GPU machine after `fcct.py prepare` and before the long runs:

    python scripts/smoke_pipeline.py

It loads LLaVA-1.5-7B once and checks every new code path on real inputs:
vision-cache equivalence, PlannedIRI == original IRI (bit-identical), hook
cleanup, VCD(alpha=0) == greedy, model files vs pinned checksums, parser/margin agreement, seeded sampling
reproducibility, contrastive tracing, and latency timing. It writes
outputs/smoke_pipeline.json and exits non-zero if any check fails.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["HF_HOME"] = str(ROOT / ".venv/hf_home")

import torch  # noqa: E402

from fcct.core import IRI  # noqa: E402
from fcct.report import pope_answer  # noqa: E402
from bench.decoding import Method, hf_generate, timed, vcd_generate  # noqa: E402
from rfcct.plan import PlannedIRI, paper_plan  # noqa: E402
from rfcct.tracing import contrastive_trace_sample, score  # noqa: E402

GREEDY = {"do_sample": False}
SAMPLING = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 50}


def run_checks(backend, ids, image, row, rr, *, sigma=0.5, tokens=8):
    """Return {check: {"pass": bool, ...}} for one image/question on a loaded backend."""
    yes, no = ids
    checks = {}
    batch, _ = backend.batch(image, row["question"])
    direct = score(backend.model, batch, yes, no)["margin"]
    cached = score(backend.model, backend.trace_batch(batch), yes, no)["margin"]
    checks["vision_cache_equivalence"] = {"pass": abs(direct - cached) < 1e-2, "direct": direct, "cached": cached}

    with torch.inference_mode():
        regular = backend.model(**batch, use_cache=False, logits_to_keep=1).logits
        reference = IRI(backend.layers, rr, k_source=3, k_target=min(10, len(rr["attention"])),
                        attention_scale=0.26, mlp_scale=0.16)
        planned = PlannedIRI(backend.layers, paper_plan(rr, 3, min(10, len(rr["attention"]))),
                             attention_scale=0.26, mlp_scale=0.16)
        with reference.apply():
            expected = backend.model(**batch, use_cache=False, logits_to_keep=1).logits
        with planned.apply():
            actual = backend.model(**batch, use_cache=False, logits_to_keep=1).logits
        after = backend.model(**batch, use_cache=False, logits_to_keep=1).logits
    checks["planned_iri_equals_original_iri"] = {"pass": bool(torch.equal(actual, expected))}
    checks["iri_changes_logits"] = {"pass": not bool(torch.equal(actual, regular))}
    checks["hooks_removed_after_iri"] = {"pass": bool(torch.equal(after, regular))}

    hf_ids, hf_first = hf_generate(backend, batch, protocol=GREEDY, max_new_tokens=tokens)
    vcd_ids, _ = vcd_generate(backend, batch, protocol=GREEDY, max_new_tokens=tokens, alpha=0.0, beta=0.1,
                              noise_step=999, noise_seed=1)
    checks["vcd_alpha0_equals_greedy"] = {"pass": hf_ids.tolist() == vcd_ids.tolist()}

    method = Method("regular")
    out = method.generate(backend, image, row["question"], protocol=GREEDY, max_new_tokens=tokens,
                          seed_key="smoke", yes_no_ids=(yes, no))
    margin_yes, text_yes = out["margin"] > 0, pope_answer(out["text"]) == "yes"
    checks["pope_parser_agrees_with_first_token_margin"] = {"pass": margin_yes == text_yes, "text": out["text"],
                                                           "margin": out["margin"]}

    for name, m in (("regular", method), ("vcd", Method("vcd", vcd={"alpha": 1.0, "beta": 0.1,
                                                                   "noise_step": {"default": 500}}))):
        a = m.generate(backend, image, row["question"], protocol=SAMPLING, max_new_tokens=tokens, seed_key="s", seed=3)
        b = m.generate(backend, image, row["question"], protocol=SAMPLING, max_new_tokens=tokens, seed_key="s", seed=3)
        checks[f"sampling_reproducible_{name}"] = {"pass": a == b, "text": a["text"]}

    trace = contrastive_trace_sample(backend, image, {**row, "label": "yes"}, yes_id=yes, no_id=no,
                                     sigma=sigma, seed=17, min_margin_gap=1e-9,
                                     check_visibility="boxes" in row)
    finite = trace["status"] != "complete" or all(
        math.isfinite(v) and len(trace["s_patched"][c]) == len(backend.layers)
        for c in ("attention", "mlp") for v in trace["s_patched"][c])
    checks["contrastive_trace"] = {"pass": finite and trace["status"] in (
        "complete", "excluded_clean_incorrect", "excluded_no_measurable_drop", "excluded_object_outside_crop"),
        "status": trace["status"], "gap": trace.get("gap")}

    device = getattr(backend, "device", torch.device("cpu"))
    _, seconds = timed(lambda: method.generate(backend, image, row["question"], protocol=GREEDY,
                                               max_new_tokens=4, min_new_tokens=4, seed_key="t"), device)
    checks["latency_timing"] = {"pass": seconds > 0, "seconds_4_tokens": seconds}
    return checks


def verify_model_files(base):
    """Compare the local LLaVA snapshot with the checksums pinned in docs/fcct_model.lock.json."""
    from huggingface_hub import snapshot_download
    from fcct.io import sha256
    lock = json.loads((ROOT / "docs/fcct_model.lock.json").read_text(encoding="utf-8"))
    if (lock["id"], lock["revision"]) != (base["model_id"], base["revision"]):
        return {"pass": False, "reason": "config model differs from the pinned lock"}
    snapshot = Path(snapshot_download(base["model_id"], revision=base["revision"], local_files_only=True))
    bad = [name for name, digest in lock["files"].items()
           if not (snapshot / name).is_file() or sha256(snapshot / name) != digest]
    return {"pass": not bad, "mismatched_or_missing": bad}


def main():
    from fcct.experiment import deterministic
    from fcct.io import read_jsonl, write_json
    from fcct.llava import Llava15
    from rfcct.tracing import resolve_token
    base = json.loads((ROOT / "configs/fcct_llava15.json").read_text(encoding="utf-8"))
    rc = json.loads((ROOT / "configs/rfcct_llava15.json").read_text(encoding="utf-8"))
    trace = ROOT / base["data_root"] / "trace.jsonl"
    if not trace.is_file():
        raise SystemExit("Run `python fcct.py prepare --download-images` first")
    if base["device"] == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for LLaVA-1.5-7B")
    deterministic(base["seed"])
    backend = Llava15(base)
    ids = (backend.yes_token_id, resolve_token(backend.processor.tokenizer, rc["no_token"], rc["expected_no_token_id"]))
    row = read_jsonl(trace)[0]
    profile = ROOT / base["output_root"] / "profile.json"
    rr = (json.loads(profile.read_text())["rr"] if profile.is_file()
          else {c: [0.1 + 0.02 * i for i in range(len(backend.layers))] for c in ("attention", "mlp")})
    from PIL import Image
    with Image.open(ROOT / base["data_root"] / row["image"]) as source:
        image = source.convert("RGB")
    checks = run_checks(backend, ids, image, row, rr)
    checks["model_files_match_pinned_lock"] = verify_model_files(base)
    result = {"all_pass": all(c["pass"] for c in checks.values()), "checks": checks,
              "rr_source": "profile" if profile.is_file() else "synthetic"}
    write_json(ROOT / "outputs/smoke_pipeline.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
