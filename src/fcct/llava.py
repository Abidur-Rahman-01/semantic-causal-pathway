"""LLaVA-1.5 adapter using pinned Hugging Face converted weights.

This is an independent HF implementation, not the unpublished author backend.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image, ImageDraw

from .core import TOKEN_GROUPS
from .io import fingerprint


def validate_token_groups(groups, image_positions, sequence_length):
    """Validate externally supplied assignments without inferring their meaning."""
    if set(groups) != set(TOKEN_GROUPS):
        raise ValueError("Token assignments must contain exactly the seven FCCT categories")
    positions = [p for name in TOKEN_GROUPS for p in groups[name]]
    if (any(type(p) is not int or not 0 <= p < sequence_length for p in positions)
            or len(set(positions)) != len(positions)):
        raise ValueError("Token groups must contain disjoint, in-range integer positions")
    if groups["last"] != [sequence_length - 1]:
        raise ValueError("Last-token group must contain exactly the final prompt position")
    visual = set(image_positions)
    for name in TOKEN_GROUPS[:3]:
        if not set(groups[name]) <= visual:
            raise ValueError("Visual group contains a nonvisual token")
    for name in TOKEN_GROUPS[3:]:
        if set(groups[name]) & visual:
            raise ValueError("Text group contains a visual token")
    if not groups["object_visual"] or not groups["object_text"]:
        raise ValueError("Object must have visible visual patches and textual tokens")
    return {name: list(groups[name]) for name in TOKEN_GROUPS}


def token_coverage(groups, image_positions, sequence_length):
    covered = {p for indices in groups.values() for p in indices}
    visual = set(image_positions)
    return {"sequence_length": sequence_length, "visual_token_count": len(visual),
            "assigned_token_count": len(covered),
            "unassigned_visual_positions": sorted(visual - covered),
            "unassigned_text_positions": sorted(set(range(sequence_length)) - visual - covered)}

VICUNA_SYSTEM = ("A chat between a curious human and an artificial intelligence assistant. "
                 "The assistant gives helpful, detailed, and polite answers to the human's questions.")


def prompt_for(question: str) -> str:
    return f"{VICUNA_SYSTEM} USER: <image>\n{question} ASSISTANT:"


def gaussian_corruption(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    """RGB-space Gaussian noise, sigma in [0,1] units, followed by clipping.

    Noise domain, sigma, clipping, and rounding are explicit reconstruction
    choices because the paper does not report them.
    """
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Noise sigma must be positive and finite")
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noise = np.random.default_rng(seed).normal(0, sigma, array.shape)
    return Image.fromarray(np.rint(np.clip(array + noise, 0, 1) * 255).astype(np.uint8))


def object_patch_indices(image: Image.Image, boxes: list[list[float]], processor,
                         patch_size: int, grid: tuple[int, int]) -> list[int]:
    """Apply the vision processor's exact resize/crop geometry to a box mask.

    Include any patch overlapping any instance of the queried category. Boxes
    are COCO x,y,width,height in original-image coordinates. Nearest-neighbor
    mask interpolation is an explicit approximation to segmentation tokens.
    """
    mask = Image.new("RGB", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for x, y, width, height in boxes:
        if not all(math.isfinite(v) for v in (x, y, width, height)) or width <= 0 or height <= 0:
            raise ValueError("Invalid object bounding box")
        # PIL rectangle has inclusive right/bottom coordinates.
        draw.rectangle((math.floor(x), math.floor(y), math.ceil(x + width) - 1,
                        math.ceil(y + height) - 1), fill="white")
    array = np.asarray(mask)
    if processor.do_resize:
        array = processor.resize(array, size=processor.size, resample=Image.Resampling.NEAREST)
    if processor.do_center_crop:
        array = processor.center_crop(array, size=processor.crop_size)
    rows, cols = grid
    if array.shape[:2] != (rows * patch_size, cols * patch_size):
        raise ValueError(f"Unexpected image geometry: {array.shape} vs {grid} * {patch_size}")
    occupied = array[..., 0].reshape(rows, patch_size, cols, patch_size).max(axis=(1, 3)) > 0
    return np.flatnonzero(occupied.ravel()).tolist()


def token_groups(image_positions: list[int], object_patches: list[int],
                 object_text: list[int], sequence_length: int) -> dict[str, list[int]]:
    if not image_positions or not object_patches or not object_text:
        raise ValueError("Object must have visible visual patches and textual tokens")
    if image_positions != list(range(min(image_positions), max(image_positions) + 1)):
        raise ValueError("Expected exactly one contiguous image-token sequence")
    if min(object_patches) < 0 or max(object_patches) >= len(image_positions):
        raise ValueError("Object patch index is out of range")
    if min(object_text) <= max(image_positions) or max(object_text) >= sequence_length - 1:
        raise ValueError("Object span must precede last token and follow image")
    objects = [image_positions[p] for p in object_patches]
    groups = {
        "early_visual": [p for p in image_positions if p < min(objects)],
        "object_visual": objects,
        "late_visual": [p for p in image_positions if p > max(objects)],
        "early_text": list(range(max(image_positions) + 1, min(object_text))),
        "object_text": object_text,
        "late_text": list(range(max(object_text) + 1, sequence_length - 1)),
        "last": [sequence_length - 1],
    }
    assert tuple(groups) == TOKEN_GROUPS
    return validate_token_groups(groups, image_positions, sequence_length)


class Llava15:
    def __init__(self, config: dict, *, local_files_only=True):
        from transformers import AutoTokenizer, CLIPImageProcessor, LlavaProcessor, LlavaForConditionalGeneration

        model_id, revision = config["model_id"], config["revision"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Model revision must be an immutable 40-character commit hash")
        # Pin the PIL/NumPy image transform while retaining tokenizer offsets.
        # A global use_fast=False would also request a slow SentencePiece
        # tokenizer, which cannot provide the offsets needed for object spans.
        options = {"revision": revision, "local_files_only": local_files_only}
        processor_config, _ = LlavaProcessor.get_processor_dict(model_id, **options)
        saved_options = {k: processor_config[k] for k in
                         ("patch_size", "vision_feature_select_strategy", "num_additional_image_tokens", "image_token")
                         if k in processor_config}
        self.processor = LlavaProcessor(
            image_processor=CLIPImageProcessor.from_pretrained(model_id, **options),
            tokenizer=AutoTokenizer.from_pretrained(model_id, use_fast=True, **options), **saved_options)
        dtype = {"float16": torch.float16, "float32": torch.float32,
                 "bfloat16": torch.bfloat16}[config["dtype"]]
        device = config["device"]
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; choose a documented CPU configuration explicitly")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, revision=revision, dtype=dtype, attn_implementation="eager",
            local_files_only=local_files_only, low_cpu_mem_usage=True).to(device).eval()
        self.device, self.dtype = torch.device(device), dtype
        self.layers = self.model.model.language_model.layers
        if len(self.layers) != 32:
            raise ValueError("Expected LLaVA-1.5-7B's 32 decoder layers")
        self.patch_size = self.model.config.vision_config.patch_size
        if self.processor.patch_size != self.patch_size:
            raise ValueError("Processor/model patch sizes differ")
        self.image_token_id = self.model.config.image_token_index
        self.yes_text = config["yes_token"]
        ids = self.processor.tokenizer.encode(self.yes_text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"yes_token must encode to exactly one token, got {ids}")
        self.yes_token_id = ids[0]
        if self.yes_token_id != config["expected_yes_token_id"]:
            raise ValueError("Affirmative token ID differs from the frozen configuration")

    def batch(self, image: Image.Image, question: str, object_name: str | None = None):
        prompt = prompt_for(question)
        batch = self.processor(images=image.convert("RGB"), text=prompt, return_tensors="pt")
        groups = None
        if object_name is not None:
            image_positions = (batch["input_ids"][0] == self.image_token_id).nonzero().flatten().tolist()
            expanded = prompt.replace("<image>", "<image>" * len(image_positions))
            encoded = self.processor.tokenizer(expanded, return_offsets_mapping=True, return_tensors="pt")
            if not torch.equal(encoded["input_ids"], batch["input_ids"]):
                raise ValueError("Offset tokenizer does not align with processor image expansion")
            start = expanded.index(question) + question.index(object_name)
            end = start + len(object_name)
            groups = {"image_positions": image_positions,
                      "object_text": [i for i, (a, b) in enumerate(encoded["offset_mapping"][0].tolist())
                                      if a < end and b > start]}
        batch = {key: value.to(self.device) for key, value in batch.items()}
        batch["pixel_values"] = batch["pixel_values"].to(self.dtype)
        return batch, groups

    def groups(self, image, row, batch, spans):
        if "token_groups" in row:
            actual = fingerprint(batch["input_ids"][0].tolist())
            if row.get("token_ids_sha256") != actual:
                raise ValueError("Precomputed token groups require a matching token_ids_sha256 for this exact prompt")
            return validate_token_groups(row["token_groups"], spans["image_positions"], batch["input_ids"].shape[1])
        height, width = batch["pixel_values"].shape[-2:]
        grid = height // self.patch_size, width // self.patch_size
        if len(spans["image_positions"]) != grid[0] * grid[1]:
            raise ValueError("Visual tokens do not match the patch grid; CLS or tiling needs another adapter")
        patches = object_patch_indices(image, row["boxes"], self.processor.image_processor, self.patch_size, grid)
        return token_groups(spans["image_positions"], patches, spans["object_text"], batch["input_ids"].shape[1])

    @torch.inference_mode()
    def attention_statistics(self, image, row):
        """Figure-4 ingredients; retain raw mass and per-token statistics.

        The paper does not specify its plotted normalization. No undocumented
        normalization is substituted for the saved attention probabilities.
        """
        from .attention import attention_statistics
        batch, spans = self.batch(image, row["question"], row["object"])
        groups = self.groups(image, row, batch, spans)
        output = self.model(**batch, use_cache=False, output_attentions=True, logits_to_keep=1)
        values = attention_statistics(output.attentions, groups["object_visual"], groups["object_text"])
        return {"layers": values, "token_coverage": token_coverage(groups, spans["image_positions"],
                                                                  batch["input_ids"].shape[1])}

    @torch.inference_mode()
    def trace_batch(self, batch):
        """Cache the fixed vision projection once for all decoder patch runs."""
        backbone = self.model.model
        embeddings = backbone.get_input_embeddings()(batch["input_ids"])
        features = torch.cat(backbone.get_image_features(batch["pixel_values"]), dim=0)
        features = features.to(device=embeddings.device, dtype=embeddings.dtype)
        mask = backbone.get_placeholder_mask(batch["input_ids"], inputs_embeds=embeddings,
                                              image_features=features)
        return {"inputs_embeds": embeddings.masked_scatter(mask, features),
                "attention_mask": batch["attention_mask"]}

    @torch.inference_mode()
    def probability(self, batch):
        # Compute the distribution at the next-token position only. Never
        # teacher-force the target or renormalize into a yes/no candidate set.
        output = self.model(**batch, use_cache=False, logits_to_keep=1)
        return float(torch.softmax(output.logits[0, -1].float(), dim=-1)[self.yes_token_id])

    @torch.inference_mode()
    def generate(self, image, question, max_new_tokens):
        batch, _ = self.batch(image, question)
        length = batch["input_ids"].shape[1]
        output = self.model.generate(**batch, do_sample=False, num_beams=1,
                                     max_new_tokens=max_new_tokens, use_cache=True)
        return self.processor.tokenizer.decode(output[0, length:], skip_special_tokens=True).strip()
