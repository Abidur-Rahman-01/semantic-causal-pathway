"""Decoding for every compared method under named protocols.

* ``regular`` / IRI-type methods use Hugging Face ``generate`` (IRI hooks active
  for IRI-type methods).
* ``vcd`` reimplements VCD (Leng et al., CVPR 2024, ``vcd_utils/vcd_sample.py``):
  ``cd = (1+alpha)*logits - alpha*logits_noised`` masked where
  ``logits < log(beta) + max(logits)``, then the protocol's warpers. The noised
  image is the closed-form diffusion forward process of ``vcd_add_noise.py``
  applied to the normalized pixel tensor.

Protocols: ``greedy`` (argmax) and ``vcd_sampling`` (direct sampling,
temperature 1, top-p 1, top-k 50: the "Regular" decoding of VCD Table 1/2,
which FCCT reuses as its baseline).
"""
from __future__ import annotations

import math
import time
from contextlib import nullcontext

import torch

from fcct.io import fingerprint


def add_diffusion_noise(image_tensor, noise_step, generator=None):
    """VCD ``add_diffusion_noise``: x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps."""
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas_prod = torch.cumprod(1 - betas, dim=0)
    t = int(noise_step)
    if not 0 <= t < num_steps:
        raise ValueError("noise_step must be in [0, 999]")
    x0 = image_tensor.float()
    noise = torch.randn(x0.shape, generator=generator, dtype=torch.float32).to(x0.device)  # CPU RNG: device-independent
    noisy = math.sqrt(float(alphas_prod[t])) * x0 + math.sqrt(1 - float(alphas_prod[t])) * noise
    return noisy.to(image_tensor.dtype)


def _warp(logits, protocol):
    """Temperature, top-k, top-p in Hugging Face order (processors are no-ops here)."""
    logits = logits / protocol["temperature"]
    top_k = protocol.get("top_k") or 0
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, -float("inf"))
    top_p = protocol.get("top_p", 1.0)
    if top_p < 1.0:
        ordered, index = torch.sort(logits, descending=False, dim=-1)
        cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative <= (1 - top_p)
        remove[..., -1:] = False
        logits = logits.masked_fill(remove.scatter(-1, index, remove), -float("inf"))
    return logits


def _select(logits, protocol):
    if not protocol["do_sample"]:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(_warp(logits, protocol), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def sample_seed(seed, key):
    return int(fingerprint([int(seed), key])[:8], 16)


@torch.inference_mode()
def hf_generate(backend, batch, *, protocol, max_new_tokens, seed=None, min_new_tokens=None):
    length = batch["input_ids"].shape[1]
    options = {"do_sample": protocol["do_sample"], "num_beams": 1}
    if protocol["do_sample"]:
        torch.manual_seed(seed)
        options.update(temperature=protocol["temperature"], top_p=protocol["top_p"], top_k=protocol["top_k"])
    if min_new_tokens:
        options["min_new_tokens"] = min_new_tokens
    output = backend.model.generate(**batch, max_new_tokens=max_new_tokens, use_cache=True,
                                    return_dict_in_generate=True, output_logits=True, **options)
    return output.sequences[0, length:], output.logits[0][0].float()


@torch.inference_mode()
def vcd_generate(backend, batch, *, protocol, max_new_tokens, alpha, beta, noise_step, noise_seed,
                 seed=None, eos_token_id=None, min_new_tokens=0):
    """Two-stream VCD decoding with separate KV caches for clean and noised images."""
    model = backend.model
    device = batch["pixel_values"].device
    generator = torch.Generator(device="cpu").manual_seed(noise_seed)
    noised = {**batch, "pixel_values": add_diffusion_noise(batch["pixel_values"].cpu(), noise_step,
                                                           generator).to(device)}
    if protocol["do_sample"]:
        torch.manual_seed(seed)
    eos = model.generation_config.eos_token_id if eos_token_id is None else eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    clean_out = model(**batch, use_cache=True)
    noised_out = model(**noised, use_cache=True)
    clean_cache, noised_cache = clean_out.past_key_values, noised_out.past_key_values
    logits, logits_cd = clean_out.logits[:, -1].float(), noised_out.logits[:, -1].float()
    mask = batch["attention_mask"]
    tokens, first = [], None
    for step in range(max_new_tokens):
        cutoff = math.log(beta) + logits.max(dim=-1, keepdim=True).values
        diffs = (1 + alpha) * logits - alpha * logits_cd
        cd = diffs.masked_fill(logits < cutoff, -float("inf"))
        if step < min_new_tokens:
            for e in eos:
                cd[:, e] = -float("inf")
        if first is None:
            first = diffs[0].clone()
        token = _select(cd, protocol)
        tokens.append(int(token))
        if int(token) in eos:
            break
        mask = torch.cat([mask, mask.new_ones((mask.shape[0], 1))], dim=-1)
        step_ids = token[:, None]
        clean_out = model(input_ids=step_ids, attention_mask=mask, past_key_values=clean_cache, use_cache=True)
        noised_out = model(input_ids=step_ids, attention_mask=mask, past_key_values=noised_cache, use_cache=True)
        clean_cache, noised_cache = clean_out.past_key_values, noised_out.past_key_values
        logits, logits_cd = clean_out.logits[:, -1].float(), noised_out.logits[:, -1].float()
    return torch.tensor(tokens), first


class Method:
    """A compared method: an optional IRI-style operator or the VCD decoder."""

    def __init__(self, name, operator=None, vcd=None):
        self.name, self.operator, self.vcd = name, operator, vcd

    def generate(self, backend, image, question, *, protocol, max_new_tokens, seed_key, seed=None,
                 benchmark=None, yes_no_ids=None, min_new_tokens=None):
        batch, _ = backend.batch(image, question)
        sample = sample_seed(seed, seed_key) if protocol["do_sample"] else None
        if self.vcd is not None:
            step = self.vcd["noise_step"].get(benchmark, self.vcd["noise_step"]["default"])
            ids, first = vcd_generate(backend, batch, protocol=protocol, max_new_tokens=max_new_tokens,
                                      alpha=self.vcd["alpha"], beta=self.vcd["beta"], noise_step=step,
                                      noise_seed=sample_seed(seed or 0, "vcd-noise|" + seed_key), seed=sample,
                                      min_new_tokens=min_new_tokens or 0)
        else:
            with self.operator.apply() if self.operator is not None else nullcontext():
                ids, first = hf_generate(backend, batch, protocol=protocol, max_new_tokens=max_new_tokens,
                                         seed=sample, min_new_tokens=min_new_tokens)
        text = backend.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
        out = {"text": text, "new_tokens": int(len(ids))}
        if yes_no_ids is not None:
            out["margin"] = float(first[yes_no_ids[0]] - first[yes_no_ids[1]])
        return out


def timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start
