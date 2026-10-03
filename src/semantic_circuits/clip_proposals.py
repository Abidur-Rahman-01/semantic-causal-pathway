"""Optional CLIP semantic relevance baseline for evidence proposals."""
from __future__ import annotations

import torch
from .runtime import configure_local_runtime, resolve_cached_revision, resolve_device


class ClipSemanticScorer:
    def __init__(self, model_id="openai/clip-vit-large-patch14", device="auto", revision=None):
        from pathlib import Path
        configure_local_runtime(Path(__file__).resolve().parents[2])
        from transformers import CLIPModel, CLIPProcessor
        self.device = resolve_device(device)
        self.model_id = model_id
        self.revision = revision
        self.processor = CLIPProcessor.from_pretrained(model_id,revision=revision)
        self.model = CLIPModel.from_pretrained(model_id,revision=revision).to(self.device).eval()
        self.resolved_revision = resolve_cached_revision(model_id, revision)

    @torch.inference_mode()
    def __call__(self, image, text: str) -> float:
        batch = self.processor(images=image.convert("RGB"), text=[text], return_tensors="pt", padding=True).to(self.device)
        output = self.model(**batch)
        visual = output.image_embeds / output.image_embeds.norm(dim=-1, keepdim=True)
        semantic = output.text_embeds / output.text_embeds.norm(dim=-1, keepdim=True)
        return float((visual @ semantic.T).item())


def rank_grounded_candidates(image, candidates: list[dict], scorer: ClipSemanticScorer) -> list[dict]:
    """Rank detected regions by CLIP cosine; a proposal baseline, not verification truth."""
    from PIL import Image
    ranked = []
    for item in candidates:
        x1,y1,x2,y2 = [int(round(x)) for x in item["box_xyxy_pixels"]]
        crop = image.convert("RGB").crop((max(0,x1), max(0,y1), min(image.width,x2), min(image.height,y2)))
        row = {k:v for k,v in item.items() if k != "mask"}
        row["clip_region_cosine"] = scorer(crop, item["concept"])
        ranked.append(row)
    return sorted(ranked,key=lambda x:x["clip_region_cosine"],reverse=True)
