"""Hookable Qwen2.5-VL backend using Hugging Face Transformers."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from PIL import Image
from .runtime import configure_local_runtime, resolve_cached_revision, resolve_device


@dataclass
class QwenConfig:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    revision: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 48
    max_image_pixels: int | None = 1_003_520


class Qwen25VL:
    """Minimal reproducible interface for baseline answers, forced-answer scores and head hooks."""
    def __init__(self, config: QwenConfig):
        configure_local_runtime(__import__("pathlib").Path(__file__).resolve().parents[2])
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.config = config
        self.resolved_revision = resolve_cached_revision(config.model_id, config.revision)
        processor_kwargs = {"revision": config.revision}
        if config.max_image_pixels is not None:
            processor_kwargs["max_pixels"] = int(config.max_image_pixels)
        self.processor = AutoProcessor.from_pretrained(config.model_id, **processor_kwargs)
        dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config.dtype]
        selected_device = resolve_device(config.device)
        device_map = "auto" if selected_device == "cuda" and config.device == "auto" else None
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_id, revision=config.revision, torch_dtype=dtype, device_map=device_map, attn_implementation="eager")
        if device_map is None:
            self.model.to(selected_device)
        self.model.eval()
        self.resolved_revision = resolve_cached_revision(config.model_id, config.revision)
        self.device = next(self.model.parameters()).device
        self.layers = self._find_attention_output_projections()
        self.forward_calls = 0
        self.forward_seconds = 0.0
        self._timers = []
        self._cost_handles = [
            self.model.register_forward_pre_hook(self._count_forward_start),
            self.model.register_forward_hook(self._count_forward_end),
        ]

    def _count_forward_start(self, module, inputs):
        self.forward_calls += 1
        self._timers.append(perf_counter())

    def _count_forward_end(self, module, inputs, output):
        if self._timers:
            self.forward_seconds += perf_counter() - self._timers.pop()

    def _find_attention_output_projections(self):
        result = {}
        for name, module in self.model.named_modules():
            if name.endswith(".self_attn.o_proj"):
                result[name] = module
        if not result:
            raise RuntimeError("No self_attn.o_proj modules found; inspect model module names for this Transformers revision.")
        return result

    def _messages(self, image: Image.Image, question: str, answer: str | None = None):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image.convert("RGB")},
            {"type": "text", "text": question},
        ]}]
        if answer is not None:
            messages.append({"role": "assistant", "content": answer})
        return messages

    def _batch(self, image: Image.Image, question: str, answer: str | None = None):
        batch = self.processor.apply_chat_template(
            self._messages(image, question, answer), tokenize=True,
            add_generation_prompt=answer is None, return_dict=True, return_tensors="pt")
        return batch.to(self.device)

    @torch.inference_mode()
    def answer(self, image: Image.Image, question: str) -> str:
        batch = self._batch(image, question)
        generated = self.model.generate(**batch, max_new_tokens=self.config.max_new_tokens,
                                        do_sample=False, use_cache=False)
        prompt_len = batch["input_ids"].shape[1]
        return self.processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0].strip()

    @torch.inference_mode()
    def answer_logprobs(self, image: Image.Image, question: str, candidates: list[str]) -> dict[str, float]:
        """Teacher-force each answer continuation and sum conditional token log-probabilities."""
        scores = {}
        for answer in candidates:
            full = self._batch(image, question, answer)
            prompt = self._batch(image, question)
            ids = full["input_ids"]
            prefix = prompt["input_ids"]
            prefix_len = prefix.shape[1]
            if ids.shape[1] <= prefix_len or not torch.equal(ids[:, :prefix_len], prefix):
                raise RuntimeError("Chat-template prefix mismatch; cannot safely compute answer continuation likelihood.")
            out = self.model(**full, use_cache=False)
            logits = out.logits[:, prefix_len-1:-1, :].float()
            target = ids[:, prefix_len:]
            token_log_probs = logits.log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
            scores[answer] = float(token_log_probs.sum().item())
        return scores

    @contextmanager
    def _hooks(self, capture: dict | None = None, patch: dict | None = None):
        handles = []
        for name, module in self.layers.items():
            def hook_factory(layer_name):
                def hook(mod, inputs):
                    activation = inputs[0]
                    if capture is not None:
                        capture[layer_name] = activation.detach().float().cpu()
                    if patch and layer_name in patch:
                        reference = patch[layer_name]["activation"].to(activation.device, activation.dtype)
                        heads = patch[layer_name]["heads"]
                        if (reference.shape[0] != activation.shape[0] or reference.shape[-1] != activation.shape[-1]
                                or reference.shape[1] > activation.shape[1]):
                            raise ValueError(f"Activation shape changed in {layer_name}: {tuple(reference.shape)} vs {tuple(activation.shape)}")
                        head_dim = activation.shape[-1] // self.model.config.text_config.num_attention_heads
                        edited = activation.clone()
                        for head in heads:
                            start, end = head * head_dim, (head + 1) * head_dim
                            edited[:, :reference.shape[1], start:end] = reference[..., start:end]
                        return (edited, *inputs[1:])
                return hook
            handles.append(module.register_forward_pre_hook(hook_factory(name)))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def capture_answer_activations(self, image: Image.Image, question: str, answer: str) -> dict:
        cache = {}
        batch = self._batch(image, question, answer)
        prompt_len = self._batch(image, question)["input_ids"].shape[1]
        with torch.inference_mode(), self._hooks(capture=cache):
            self.model(**batch, use_cache=False)
        # Exclude forced-answer token positions to prevent target leakage in mediator search.
        return {layer: activation[:, :prompt_len].clone() for layer, activation in cache.items()}

    def patched_answer_logprob(self, image: Image.Image, question: str, answer: str,
                               patch: dict[str, dict[str, Any]]) -> float:
        # Rescore with cached activations patched at attention output projection inputs.
        batch = self._batch(image, question, answer)
        prompt = self._batch(image, question)
        prefix_len = prompt["input_ids"].shape[1]
        with torch.inference_mode(), self._hooks(patch=patch):
            out = self.model(**batch, use_cache=False)
        logits = out.logits[:, prefix_len-1:-1, :].float()
        target = batch["input_ids"][:, prefix_len:]
        return float(logits.log_softmax(-1).gather(-1, target.unsqueeze(-1)).sum().item())

    def patched_answer_logprobs(self, image: Image.Image, question: str, candidates: list[str],
                                patch: dict[str, dict[str, Any]]) -> dict[str, float]:
        return {candidate: self.patched_answer_logprob(image, question, candidate, patch)
                for candidate in candidates}


def rank_attention_heads(factual: dict, counterfactual: dict, num_heads: int, top_k: int = 40) -> list[dict]:
    """Rank layer/head units by factual-counterfactual attention input delta norm."""
    import torch
    scores = []
    missing = set(factual) ^ set(counterfactual)
    if missing:
        raise ValueError(f"Layer activation mismatch prevents aligned tracing: {sorted(missing)[:3]}")
    for layer, fact in factual.items():
        if fact.shape != counterfactual[layer].shape:
            raise ValueError(f"Sequence/activation shape changed for {layer}; reject or re-align this variant.")
        delta = (fact - counterfactual[layer]).float()
        head_dim = delta.shape[-1] // num_heads
        for head in range(num_heads):
            score = float(delta[..., head*head_dim:(head+1)*head_dim].norm().item())
            scores.append({"layer": layer, "head": head, "attribution_norm": score})
    return sorted(scores, key=lambda row: row["attribution_norm"], reverse=True)[:top_k]
