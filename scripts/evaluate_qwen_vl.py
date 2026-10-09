"""Generate answers from a Qwen2.5-VL LoRA adapter and score VQA consensus accuracy."""
from __future__ import annotations
import argparse, json, re, string
from collections import Counter
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".venv" / "hf_home")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(ROOT / ".venv" / "hf_home" / "hub")

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def normalize(s: str) -> str:
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def score(pred: str, answers: list[str], dataset: str) -> float:
    pred = normalize(pred)
    if dataset == "gqa": return float(any(pred == normalize(a) for a in answers))
    refs = [normalize(a) for a in answers]
    # Official VQA consensus score: min(matching_answers / 3.0, 1.0)
    return min(sum(x == pred for x in refs) / 3.0, 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True); p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct"); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--max-new-tokens", type=int, default=16)
    a = p.parse_args()
    rows = [json.loads(x) for x in a.data.read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.limit: rows = rows[:a.limit]
    if a.adapter:
        processor = AutoProcessor.from_pretrained(a.adapter)
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.base_model, torch_dtype="auto", device_map="auto")
        model = PeftModel.from_pretrained(base, a.adapter).eval()
    else:
        processor = AutoProcessor.from_pretrained(a.base_model)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.base_model, torch_dtype="auto", device_map="auto").eval()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    results = []; totals = Counter(); counts = Counter()
    for ix, row in enumerate(rows, 1):
        image = Image.open(row["image"]).convert("RGB")
        image.thumbnail((512, 512))
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": row["question"] + " Answer briefly."}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
        with torch.inference_mode(): generated = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        prediction = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        refs = row.get("answers") or [row.get("answer", "")]
        dataset = row.get("dataset", "vqav2")
        metric = score(prediction, refs, dataset)
        totals[dataset] += metric; counts[dataset] += 1
        results.append({"sample_id": row.get("sample_id"), "image_id": row.get("image_id"), "dataset": dataset,
                        "question": row.get("question", ""), "prediction": prediction, "answers": refs, "score": metric})
        if ix % 100 == 0: print(f"Evaluated {ix}/{len(rows)}", flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {d: {"n": counts[d], "accuracy": totals[d] / max(counts[d], 1)} for d in counts}
    with a.output.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "predictions": results}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
