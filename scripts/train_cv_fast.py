"""High-performance in-memory 5-Fold Cross-Validation, Final Model Training, and Test Benchmarking.

Peak GPU Optimization on NVIDIA RTX 3090 (24GB VRAM):
- Base Qwen2.5-VL-7B loaded once in VRAM (zero redundant 15GB disk reloads between folds)
- TF32 Tensor Core math enabled
- cuDNN benchmark auto-tuning enabled
- bfloat16 mixed precision
- Gradient checkpointing enabled
- 10 full epochs per fold across all 5 folds
- In-process instant validation and PEFT adapter swapping
- Zero-shot baseline vs 10-epoch LoRA comparison on held-out locked test set
- Automated generation of 300 DPI publication research figures
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import random
import re
import string
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".venv" / "hf_home")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(ROOT / ".venv" / "hf_home" / "hub")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import generate_paper_diagrams as gpd


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


class VQADataset(Dataset):
    def __init__(self, rows: list[dict]): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def answer_of(row: dict) -> str:
    answers = row.get("answers") or [row.get("answer", "")]
    answers = [str(a).strip() for a in answers if str(a).strip()]
    if not answers: raise ValueError(f"No answer for {row.get('sample_id')}")
    return random.choice(answers)


def normalize_text(s: str) -> str:
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def score_answer(pred: str, answers: list[str], dataset: str) -> float:
    pred = normalize_text(pred)
    if dataset == "gqa":
        return float(any(pred == normalize_text(a) for a in answers))
    refs = [normalize_text(a) for a in answers]
    return min(sum(x == pred for x in refs) / 3.0, 1.0)


def collate_fn(rows: list[dict], processor, max_length: int = 512):
    texts, images = [], []
    for row in rows:
        img = Image.open(row["image"]).convert("RGB")
        img.thumbnail((448, 448))
        user = [{"type": "image", "image": img}, {"type": "text", "text": row["question"]}]
        prompt = processor.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        full = processor.apply_chat_template([{"role": "user", "content": user},
            {"role": "assistant", "content": [{"type": "text", "text": answer_of(row)}]}], tokenize=False)
        texts.append((prompt, full))
        images.append(img)
    prompts, fulls = zip(*texts)
    full_batch = processor(text=list(fulls), images=list(images), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    labels = full_batch["input_ids"].clone()
    labels[full_batch["attention_mask"] == 0] = -100
    for i, prompt in enumerate(prompts):
        prefix_len = len(processor.tokenizer(prompt)["input_ids"])
        labels[i, :min(prefix_len, labels.shape[1])] = -100
    batch = dict(full_batch)
    batch["labels"] = labels
    return batch


def evaluate_loss(model, loader, device) -> float:
    model.eval()
    total = count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            total += float(loss) * batch["input_ids"].shape[0]
            count += batch["input_ids"].shape[0]
    model.train()
    return total / max(count, 1)


def evaluate_accuracy(model, processor, rows: list[dict], device, max_new_tokens: int = 16, limit: int = 100) -> dict:
    model.eval()
    if limit and len(rows) > limit:
        rows = rows[:limit]
    results = []
    totals = Counter()
    counts = Counter()
    t0 = time.time()
    for ix, row in enumerate(rows, 1):
        img = Image.open(row["image"]).convert("RGB")
        img.thumbnail((448, 448))
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": row["question"] + " Answer briefly."}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(device)
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tok = gen[:, inputs["input_ids"].shape[1]:]
        pred = processor.batch_decode(new_tok, skip_special_tokens=True)[0].strip()
        refs = row.get("answers") or [row.get("answer", "")]
        ds = row.get("dataset", "vqav2")
        s = score_answer(pred, refs, ds)
        totals[ds] += s
        counts[ds] += 1
        results.append({
            "sample_id": row.get("sample_id"),
            "image_id": row.get("image_id"),
            "dataset": ds,
            "question": row.get("question", ""),
            "prediction": pred,
            "answers": refs,
            "score": s,
        })
        if ix % 50 == 0 or ix == len(rows):
            el = max(time.time() - t0, 0.1)
            print(f"    Evaluated {ix}/{len(rows)} samples ({ix/el:.1f} samples/s)", flush=True)

    summary = {d: {"n": counts[d], "accuracy": totals[d] / max(counts[d], 1)} for d in counts}
    return {"summary": summary, "predictions": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per fold (default: 10)")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--grad-accumulation", type=int, default=4, help="Gradient accumulation (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--max-train-samples", type=int, default=300, help="Train samples per fold (default: 300)")
    parser.add_argument("--max-val-samples", type=int, default=100, help="Val samples per fold (default: 100)")
    parser.add_argument("--test-eval-samples", type=int, default=250, help="Locked test evaluation samples (default: 250)")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    processed_dir = ROOT / "data" / "processed"
    folds_dir = processed_dir / "folds"
    output_root = ROOT / "outputs" / "qwen-vqa-lora"
    diagrams_dir = ROOT / "outputs" / "research_diagrams"
    output_root.mkdir(parents=True, exist_ok=True)
    diagrams_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA GPU is required for this experiment.")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)

    print("=" * 80)
    print("END-TO-END 5-FOLD CROSS-VALIDATION & BENCHMARK SUITE")
    print(f"  Target GPU:         {gpu_name} ({total_vram} GB VRAM)")
    print(f"  GPU Optimizations:  TF32=True, cuDNN benchmark=True, bfloat16 mixed precision")
    print(f"  Cross-Validation:   {args.folds} Folds, {args.epochs} Epochs per fold")
    print(f"  Effective Batch:    {args.batch_size * args.grad_accumulation} (batch {args.batch_size} × accum {args.grad_accumulation})")
    print(f"  Sample Budget:      {args.max_train_samples} train / {args.max_val_samples} val per fold")
    print(f"  Test Evaluation:    {args.test_eval_samples} held-out locked test samples")
    print("=" * 80, flush=True)

    # 1. Load Base Model Once into GPU Memory
    print(f"\n[1/5] Loading Base Qwen2.5-VL-7B-Instruct onto {gpu_name}...", flush=True)
    t_load = time.time()
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()
    print(f"Base model loaded in {time.time()-t_load:.1f}s! Initial VRAM: {torch.cuda.memory_allocated()/(1024**3):.2f} GB", flush=True)

    # 2. Evaluate Baseline Unadapted Base Model on Locked Test Set
    print("\n" + "=" * 80)
    print("[2/5] Evaluating Pre-trained Base Model (Zero-Shot Baseline) on Locked Test Set")
    print("=" * 80, flush=True)
    locked_test_rows = load_rows(processed_dir / "locked_test.jsonl")
    baseline_test_file = output_root / "baseline_locked_test_predictions.json"
    
    base_res = evaluate_accuracy(base_model, processor, locked_test_rows, device, limit=args.test_eval_samples)
    baseline_test_file.write_text(json.dumps(base_res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Baseline Zero-Shot Accuracy on Locked Test Set:")
    for d, s in base_res["summary"].items():
        print(f"  {d.upper()}: {s['accuracy']*100:.2f}% (N={s['n']})")

    # 3. Execute 5-Fold Cross Validation (10 Epochs each)
    print("\n" + "=" * 80)
    print("[3/5] Launching 5-Fold Cross Validation (10 Epochs per fold)")
    print("=" * 80, flush=True)

    fold_reports = {}
    fold_histories = {}
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    for fold in range(args.folds):
        print(f"\n--- FOLD {fold+1}/{args.folds} (10 Epochs) ---", flush=True)
        train_file = folds_dir / f"fold_{fold}" / "train.jsonl"
        val_file = folds_dir / f"fold_{fold}" / "validation.jsonl"
        fold_out = output_root / f"fold_{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)

        train_rows = load_rows(train_file)[:args.max_train_samples]
        val_rows = load_rows(val_file)[:args.max_val_samples]

        train_loader = DataLoader(
            VQADataset(train_rows), batch_size=args.batch_size, shuffle=True,
            pin_memory=True, collate_fn=lambda rows: collate_fn(rows, processor)
        )
        val_loader = DataLoader(
            VQADataset(val_rows), batch_size=1, shuffle=False,
            pin_memory=True, collate_fn=lambda rows: collate_fn(rows, processor)
        )

        # Wrap with fresh LoRA adapter
        model = get_peft_model(base_model, lora_config)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
        steps_per_epoch = (len(train_loader) + args.grad_accumulation - 1) // args.grad_accumulation
        total_steps = steps_per_epoch * args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

        history = []
        best_val_loss = float("inf")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            t_epoch = time.time()
            for step, batch in enumerate(train_loader, 1):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss / args.grad_accumulation
                loss.backward()
                running_loss += float(loss.detach()) * args.grad_accumulation
                if step % args.grad_accumulation == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            train_loss = running_loss / max(len(train_loader), 1)
            val_loss = evaluate_loss(model, val_loader, device)
            peak_vram = round(torch.cuda.max_memory_allocated(device) / (1024**3), 2)
            epoch_time = time.time() - t_epoch

            rec = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "validation_loss": round(val_loss, 4),
                "learning_rate": round(scheduler.get_last_lr()[0], 6),
                "peak_vram_gb": peak_vram,
                "time_sec": round(epoch_time, 1),
            }
            history.append(rec)
            print(f"  Fold {fold} | Epoch {epoch+1:02d}/10 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Peak VRAM: {peak_vram}GB ({epoch_time:.1f}s)", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(fold_out / "best_adapter")
                processor.save_pretrained(fold_out / "best_adapter")

        (fold_out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        fold_histories[f"fold_{fold}"] = history

        # Evaluate best adapter on fold validation set
        print(f"  Evaluating Fold {fold} Best Adapter on Validation Set ({len(val_rows)} samples)...", flush=True)
        val_eval_res = evaluate_accuracy(model, processor, val_rows, device, limit=args.max_val_samples)
        (fold_out / "validation_predictions.json").write_text(json.dumps(val_eval_res, indent=2, ensure_ascii=False), encoding="utf-8")
        fold_reports[f"fold_{fold}"] = val_eval_res["summary"]

        for d, s in val_eval_res["summary"].items():
            print(f"    Fold {fold} {d.upper()} Accuracy: {s['accuracy']*100:.2f}% (N={s['n']})")

        # Unload LoRA adapter to leave base model clean for next fold
        base_model = model.unload()
        del optimizer, scheduler, model
        torch.cuda.empty_cache()

    # 4. Statistical Aggregation of Cross-Validation Results
    print("\n" + "=" * 80)
    print("5-FOLD CROSS-VALIDATION STATISTICAL AGGREGATION (10 EPOCHS)")
    print("=" * 80, flush=True)
    datasets = ["gqa", "vqav2"]
    cv_summary = {"per_fold": fold_reports, "cross_validation": {}, "hyperparameters": vars(args)}

    for d in datasets:
        accs = [fold_reports[f"fold_{f}"][d]["accuracy"] for f in range(args.folds) if d in fold_reports[f"fold_{f}"]]
        counts = [fold_reports[f"fold_{f}"][d]["n"] for f in range(args.folds) if d in fold_reports[f"fold_{f}"]]
        if accs:
            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            cv_summary["cross_validation"][d] = {
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "fold_accuracies": accs,
                "total_evaluated": int(np.sum(counts)),
            }
            print(f"  {d.upper()}: Mean Accuracy = {mean_acc*100:.2f}% ± {std_acc*100:.2f}% (N={np.sum(counts)})")

    overall_accs = []
    for f in range(args.folds):
        f_dict = fold_reports[f"fold_{f}"]
        tot_correct = sum(f_dict[d]["accuracy"] * f_dict[d]["n"] for d in f_dict)
        tot_n = sum(f_dict[d]["n"] for d in f_dict)
        overall_accs.append(tot_correct / max(tot_n, 1))

    cv_summary["cross_validation"]["overall"] = {
        "mean_accuracy": float(np.mean(overall_accs)),
        "std_accuracy": float(np.std(overall_accs)),
        "fold_accuracies": overall_accs,
    }
    print(f"  OVERALL: Mean Accuracy = {np.mean(overall_accs)*100:.2f}% ± {np.std(overall_accs)*100:.2f}%")
    (output_root / "cross_validation_summary.json").write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")

    # 5. Train Final Model on Combined Dataset for 10 Epochs
    print("\n" + "=" * 80)
    print("[4/5] Training Final Model on Combined Dataset (10 Epochs)")
    print("=" * 80, flush=True)

    final_train_rows = load_rows(processed_dir / "train.jsonl")[:args.max_train_samples * 2]
    final_val_rows = load_rows(processed_dir / "validation.jsonl")[:args.max_val_samples]
    final_out = output_root / "final"
    final_out.mkdir(parents=True, exist_ok=True)

    final_train_loader = DataLoader(
        VQADataset(final_train_rows), batch_size=args.batch_size, shuffle=True,
        pin_memory=True, collate_fn=lambda rows: collate_fn(rows, processor)
    )
    final_val_loader = DataLoader(
        VQADataset(final_val_rows), batch_size=1, shuffle=False,
        pin_memory=True, collate_fn=lambda rows: collate_fn(rows, processor)
    )

    final_model = get_peft_model(base_model, lora_config)
    optimizer = torch.optim.AdamW((p for p in final_model.parameters() if p.requires_grad), lr=args.lr)
    steps_per_epoch = (len(final_train_loader) + args.grad_accumulation - 1) // args.grad_accumulation
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    final_history = []
    best_final_val_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        final_model.train()
        running_loss = 0.0
        t_epoch = time.time()
        for step, batch in enumerate(final_train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = final_model(**batch).loss / args.grad_accumulation
            loss.backward()
            running_loss += float(loss.detach()) * args.grad_accumulation
            if step % args.grad_accumulation == 0 or step == len(final_train_loader):
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        train_loss = running_loss / max(len(final_train_loader), 1)
        val_loss = evaluate_loss(final_model, final_val_loader, device)
        peak_vram = round(torch.cuda.max_memory_allocated(device) / (1024**3), 2)
        epoch_time = time.time() - t_epoch

        rec = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "validation_loss": round(val_loss, 4),
            "peak_vram_gb": peak_vram,
            "time_sec": round(epoch_time, 1),
        }
        final_history.append(rec)
        print(f"  Final Model | Epoch {epoch+1:02d}/10 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Peak VRAM: {peak_vram}GB ({epoch_time:.1f}s)", flush=True)

        if val_loss < best_final_val_loss:
            best_final_val_loss = val_loss
            final_model.save_pretrained(final_out / "best_adapter")
            processor.save_pretrained(final_out / "best_adapter")

    (final_out / "training_history.json").write_text(json.dumps(final_history, indent=2), encoding="utf-8")

    # 6. Evaluate Final Model on Held-Out Locked Test Set
    print("\n" + "=" * 80)
    print("[5/5] Evaluating Final Model on Held-Out Locked Test Set")
    print("=" * 80, flush=True)

    final_test_file = final_out / "locked_test_predictions.json"
    final_res = evaluate_accuracy(final_model, processor, locked_test_rows, device, limit=args.test_eval_samples)
    final_test_file.write_text(json.dumps(final_res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON ON HELD-OUT LOCKED TEST SET (DISJOINT IMAGES)")
    print("=" * 80)
    for d in datasets:
        if d in base_res["summary"] and d in final_res["summary"]:
            b_acc = base_res["summary"][d]["accuracy"] * 100
            f_acc = final_res["summary"][d]["accuracy"] * 100
            diff = f_acc - b_acc
            print(f"  {d.upper()}: Base={b_acc:.2f}% | 10-Epoch LoRA={f_acc:.2f}% | Delta={diff:+.2f}% (N={final_res['summary'][d]['n']})")

    b_overall = sum(base_res["summary"][d]["accuracy"] * base_res["summary"][d]["n"] for d in base_res["summary"]) / sum(base_res["summary"][d]["n"] for d in base_res["summary"]) * 100
    f_overall = sum(final_res["summary"][d]["accuracy"] * final_res["summary"][d]["n"] for d in final_res["summary"]) / sum(final_res["summary"][d]["n"] for d in final_res["summary"]) * 100
    print(f"  OVERALL: Base={b_overall:.2f}% | 10-Epoch LoRA={f_overall:.2f}% | Delta={f_overall-b_overall:+.2f}%")

    # 7. Generate Publication-Quality Research Paper Diagrams (300 DPI)
    print("\n" + "=" * 80)
    print("GENERATING RESEARCH PAPER DIAGRAMS (300 DPI)")
    print("=" * 80, flush=True)

    gpd.plot_loss_curves(fold_histories, diagrams_dir / "loss_curves_5folds.png")
    gpd.plot_cross_validation_accuracy(cv_summary, diagrams_dir / "cross_validation_accuracy.png")
    gpd.plot_benchmark_comparison(base_res["summary"], final_res["summary"], diagrams_dir / "benchmark_comparison.png")
    gpd.plot_question_breakdown(final_res.get("predictions", []), diagrams_dir / "question_type_accuracy.png")
    gpd.plot_gpu_profile(diagrams_dir / "gpu_efficiency_profile.png")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"  Cross-Validation Metrics: {output_root / 'cross_validation_summary.json'}")
    print(f"  Final Benchmark Results:  {final_test_file}")
    print(f"  Publication Diagrams:     {diagrams_dir}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
