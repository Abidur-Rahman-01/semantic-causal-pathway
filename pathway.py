"""Single command line entry point for the semantic causal pathway architecture."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Listing commands does not import GPU dependencies.
COMMANDS = {
    "data": ("prepare_vqa_v2.py", (), "Prepare VQAv2 study data"),
    "models": ("download_models.py", (), "Cache configured model checkpoints"),
    "sample": ("sample_pathway_images.py", (), "Sample images and assign splits"),
    "review-answers": ("apply_candidate_answer_review.py", (), "Apply human candidate-answer review"),
    "propose": ("propose_evidence.py", (), "Propose Grounding DINO/SAM2 evidence masks"),
    "review-evidence": ("apply_evidence_review.py", (), "Apply human evidence-mask review"),
    "controls": ("prepare_controls.py", (), "Generate matched spatial controls"),
    "review-controls": ("apply_control_review.py", (), "Apply human control-mask review"),
    "variants": ("prepare_variants.py", (), "Generate transformation proposals"),
    "review-variants": ("apply_variant_review.py", (), "Apply human variant review and roles"),
    "run": ("run_pilot.py", (), "Validate, trace mediators, and summarize models"),
    "validate": ("run_benchmark.py", ("validate",), "Validate a versioned benchmark manifest"),
    "split": ("run_benchmark.py", ("split",), "Assign image-grouped benchmark splits"),
    "report": ("run_benchmark.py", ("report",), "Generate statistics, figures, and checksums"),
    "seal": ("benchmark_manifest.py", ("seal",), "Freeze asset checksums in a versioned manifest"),
    "verify": ("run_benchmark.py", ("verify",), "Verify archived run checksums"),
    "annotations": ("build_variant_annotations.py", (), "Export review annotations"),
    "failure-data": ("prepare_failure_benchmark.py", (), "Prepare observed failure labels"),
    "train-failure": ("train_failure_predictor.py", (), "Fit train-only failure predictor"),
    "ollama": ("run_ollama_vision_benchmark.py", (), "Evaluate local Ollama vision models"),
    "training-data": ("setup_full_vqa_dataset.py", (), "Prepare full VQAv2/GQA training datasets"),
    "convert": ("convert_vqa_datasets.py", (), "Convert VQAv2 or GQA source data"),
    "training-splits": ("build_vqa_manifests.py", (), "Build image-grouped training splits"),
    "folds": ("build_vqa_folds.py", (), "Build cross-validation folds"),
    "train": ("train_qwen_vl_lora.py", (), "Train a Qwen LoRA adapter"),
    "evaluate": ("evaluate_qwen_vl.py", (), "Evaluate a base model or LoRA adapter"),
}


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, epilog="Use: python pathway.py COMMAND --help")
    parser.add_argument("command", choices=COMMANDS, help="Workflow stage")
    if not args or args[0] in {"-h", "--help"}:
        parser.print_help()
        print("\nWorkflow commands:")
        for name, (_, _, description) in COMMANDS.items():
            print(f"  {name:18} {description}")
        if not args:
            parser.exit(2)
        return
    command = parser.parse_args(args[:1]).command
    script, prefix, _ = COMMANDS[command]
    for path in (ROOT, ROOT / "src", ROOT / "scripts"):
        sys.path.insert(0, str(path))
    sys.argv = [str(ROOT / "scripts" / script), *prefix, *args[1:]]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
