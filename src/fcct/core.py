"""FCCT equation (2), IRI equations (3)--(5), and scoped activation hooks."""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch

COMPONENTS = ("attention", "mlp", "hidden")
TOKEN_GROUPS = ("early_visual", "object_visual", "late_visual", "early_text",
                "object_text", "late_text", "last")


def recovery_rate(clean: float, corrupted: float, patched: float,
                  min_gap: float = 1e-6) -> float | None:
    """Unclipped probability RR; undefined for zero/negative corruption gaps.

    The exclusion policy and epsilon are implementation choices, not specified
    by the paper. Retain the raw probabilities and exclusion counts in reports.
    """
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (clean, corrupted, patched)):
        raise ValueError("Expected finite full-vocabulary probabilities in [0, 1]")
    if not math.isfinite(min_gap) or min_gap <= 0:
        raise ValueError("min_gap must be positive and finite")
    return (patched - corrupted) / (clean - corrupted) if clean - corrupted > min_gap else None


def select_layers(rr: list[float], k: int) -> list[int]:
    if not 1 <= k <= len(rr) or not all(math.isfinite(x) for x in rr):
        raise ValueError("Layer ranking requires finite RRs and 1 <= k <= layer count")
    # Stable layer-index tie break is explicit, not an author-supplied choice.
    return sorted(sorted(range(len(rr)), key=lambda i: (-rr[i], i))[:k])


def inject(original: torch.Tensor, sources: list[tuple[float, torch.Tensor]],
           scale: float, normalize: bool = True) -> torch.Tensor:
    """Inject into last-token output, preserving each batch vector's L2 norm."""
    if not math.isfinite(scale):
        raise ValueError("IRI scale must be finite")
    if not sources or scale == 0:
        return original
    # Accumulate in float32 to avoid FP16 overflow, then cast back.
    base = original.float()
    updated = base.clone()
    last = base[:, -1, :]
    mixed = last.clone()
    for weight, activation in sources:
        mixed = mixed + scale * float(weight) * activation.to(base.device).float()
    if normalize:
        old_norm = torch.linalg.vector_norm(last, dim=-1, keepdim=True)
        new_norm = torch.linalg.vector_norm(mixed, dim=-1, keepdim=True)
        # Exact cancellation has no direction; preserve the original vector.
        mixed = torch.where(new_norm > 0, mixed * old_norm / new_norm.clamp_min(1e-30), last)
    updated[:, -1, :] = mixed
    return updated.to(original.dtype)


def _tensor(output):
    return output[0] if isinstance(output, tuple) else output


def _replace(output, value):
    return (value, *output[1:]) if isinstance(output, tuple) else value


def component_module(layer, component: str):
    if component not in COMPONENTS:
        raise ValueError(f"Unknown component: {component}")
    return {"attention": layer.self_attn, "mlp": layer.mlp, "hidden": layer}[component]


@contextmanager
def capture(layers, groups: dict[str, list[int]]):
    """Cache only group positions, on CPU, without keeping computation graphs."""
    positions = sorted({p for values in groups.values() for p in values})
    cache, handles = {}, []

    def hook_for(index, component):
        def hook(module, args, output):
            tensor = _tensor(output)
            if tensor.ndim != 3 or any(p < 0 or p >= tensor.shape[1] for p in positions):
                raise ValueError("Token positions do not align with decoder activations")
            cache[index, component] = tensor[:, positions, :].detach().cpu().clone()
        return hook

    try:
        for index, layer in enumerate(layers):
            for component in COMPONENTS:
                handles.append(component_module(layer, component).register_forward_hook(hook_for(index, component)))
        yield cache, positions
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def patch_component(layer, component, positions, values):
    """Restore one component/layer/category, including tuple-valued MHSA outputs."""
    def hook(module, args, output):
        tensor = _tensor(output)
        if not positions or min(positions) < 0 or max(positions) >= tensor.shape[1]:
            raise ValueError("Invalid or empty patch positions")
        if values.shape != (tensor.shape[0], len(positions), tensor.shape[-1]):
            raise ValueError("Clean and corrupted activation shapes differ")
        patched = tensor.clone()
        patched[:, positions, :] = values.to(device=tensor.device, dtype=tensor.dtype)
        return _replace(output, patched)
    handle = component_module(layer, component).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class IRI:
    """Same-forward sources, strict k < l ordering, and no cross-request cache.

    Sources are saved before injection. Only the last current token is changed,
    both during prefill and each cached decoding step. This generation policy
    must be checked against author code if it becomes available.
    """
    def __init__(self, layers, rr: dict[str, list[float]], *, k_source=3, k_target=10,
                 attention_scale=0.26, mlp_scale=0.16, normalize=True):
        self.layers, self.rr = layers, rr
        self.sources, self.targets = {}, {}
        for component in ("attention", "mlp"):
            if len(rr[component]) != len(layers):
                raise ValueError("RR profile layer count differs from the loaded decoder")
            self.sources[component] = select_layers(rr[component], k_source)
            self.targets[component] = select_layers(rr[component], k_target)
        self.scales = {"attention": attention_scale, "mlp": mlp_scale}
        self.normalize = normalize
        self._active = False

    @contextmanager
    def apply(self):
        if self._active:
            raise RuntimeError("IRI contexts cannot be nested")
        self._active = True
        buffers, handles = {}, []

        def hook_for(index, component):
            def hook(module, args, output):
                tensor = _tensor(output)
                if index == 0:
                    buffers[component] = {}  # new forward (including decoding)
                buffer = buffers.setdefault(component, {})
                if index in self.sources[component]:
                    buffer[index] = tensor[:, -1, :].detach().clone()
                if index in self.targets[component]:
                    sources = [(self.rr[component][k], value) for k, value in buffer.items() if k < index]
                    return _replace(output, inject(tensor, sources, self.scales[component], self.normalize))
            return hook

        try:
            # Layer zero is always hooked so every forward resets buffers.
            for index, layer in enumerate(self.layers):
                for component in ("attention", "mlp"):
                    if index == 0 or index in self.sources[component] or index in self.targets[component]:
                        handles.append(component_module(layer, component).register_forward_hook(hook_for(index, component)))
            yield
        finally:
            for handle in handles:
                handle.remove()
            buffers.clear()
            self._active = False
