"""LoRA fine-tune Qwen2.5-VL on converted VQAv2/GQA JSONL files.

Example: python scripts/train_qwen_vl_lora.py --train data/processed/train.jsonl \
  --validation data/processed/validation.jsonl --output-dir outputs/qwen-vqa-lora --epochs 10
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


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
    # Randomly sample one human answer each epoch for VQAv2; singleton GQA remains fixed.
    return random.choice(answers)


def collate(rows: list[dict], processor, max_length: int):
    texts, images = [], []
    for row in rows:
        image = Image.open(row["image"]).convert("RGB")
        user = [{"type": "image", "image": image}, {"type": "text", "text": row["question"]}]
        prompt = processor.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        full = processor.apply_chat_template([{"role": "user", "content": user},
            {"role": "assistant", "content": answer_of(row)}], tokenize=False)
        texts.append((prompt, full)); images.append(image)
    prompts, fulls = zip(*texts)
    prompt_batch = processor(text=list(prompts), images=images, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    full_batch = processor(text=list(fulls), images=images, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    labels = full_batch["input_ids"].clone()
    labels[full_batch["attention_mask"] == 0] = -100
    # Mask user/prompt tokens; supervise only assistant answer tokens.
    for i, prompt in enumerate(prompts):
        prefix = processor(text=[prompt], images=[images[i]], return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].shape[1]
        labels[i, :min(prefix, labels.shape[1])] = -100
    batch = dict(full_batch)
    batch["labels"] = labels
    return batch


def evaluate_loss(model, loader, device) -> float:
    model.eval(); total = count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            total += float(loss) * batch["input_ids"].shape[0]
            count += batch["input_ids"].shape[0]
    model.train()
    return total / max(count, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True); p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--epochs", type=int, default=10); p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accumulation", type=int, default=8); p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=1024); p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-validation-samples", type=int, default=2000); p.add_argument("--seed", type=int, default=17)
    p.add_argument("--num-workers", type=int, default=0)
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.grad_accumulation < 1: p.error("epochs, batch-size and grad-accumulation must be positive")
    random.seed(a.seed); torch.manual_seed(a.seed)
    train_rows = load_rows(a.train); val_rows = load_rows(a.validation)
    if a.max_train_samples: train_rows = train_rows[:a.max_train_samples]
    val_rows = val_rows[:a.max_validation_samples]
    if not torch.cuda.is_available(): raise RuntimeError("This 7B vision-language fine-tune requires CUDA; use a smaller model or add a CPU/offload setup.")
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(a.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    train_loader = DataLoader(VQADataset(train_rows), batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
        collate_fn=lambda rows: collate(rows, processor, a.max_length))
    val_loader = DataLoader(VQADataset(val_rows), batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda rows: collate(rows, processor, a.max_length))
    optimizer = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad), lr=a.lr)
    steps_per_epoch = (len(train_loader) + a.grad_accumulation - 1) // a.grad_accumulation
    total_steps = steps_per_epoch * a.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    history = []; best = float("inf"); optimizer.zero_grad(set_to_none=True)
    for epoch in range(a.epochs):
        model.train(); running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / a.grad_accumulation
            loss.backward(); running += float(loss.detach()) * a.grad_accumulation
            if step % a.grad_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
        train_loss = running / max(len(train_loader), 1)
        val_loss = evaluate_loss(model, val_loader, device)
        record = {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss,
                  "learning_rate": scheduler.get_last_lr()[0], "train_rows": len(train_rows), "validation_rows": len(val_rows)}
        history.append(record); print(json.dumps(record), flush=True)
        if val_loss < best:
            best = val_loss; model.save_pretrained(a.output_dir / "best_adapter"); processor.save_pretrained(a.output_dir / "best_adapter")
        with (a.output_dir / "training_history.json").open("w", encoding="utf-8") as f: json.dump(history, f, indent=2)
    model.save_pretrained(a.output_dir / "final_adapter"); processor.save_pretrained(a.output_dir / "final_adapter")
    metadata = {"base_model": a.model, "epochs_requested": a.epochs, "train_rows": len(train_rows),
        "validation_rows": len(val_rows), "best_validation_loss": best, "seed": a.seed,
        "note": "Training loss is answer-token cross entropy; report held-out benchmark accuracy separately."}
    (a.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved best adapter and metadata under {a.output_dir}")


if __name__ == "__main__": main()
