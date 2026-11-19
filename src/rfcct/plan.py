"""Explicit injection plans, executed with IRI Equations 3-5 unchanged.

``PlannedIRI`` is the IRI operator of ``fcct.core.IRI`` with the layer choice
and the per-source weights supplied as data. For the plan produced by
``paper_plan`` it is numerically identical to ``fcct.core.IRI`` (tested).
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path

from fcct.core import _replace, _tensor, component_module, inject, select_layers

COMPONENTS = ("attention", "mlp")


def validate_plan(plan, num_layers):
    if set(plan) != set(COMPONENTS):
        raise ValueError("A plan must define attention and mlp")
    for component in COMPONENTS:
        part = plan[component]
        sources, targets, weights = part["sources"], part["targets"], part["weights"]
        for values in (sources, targets):
            if (len(set(values)) != len(values)
                    or any(type(l) is not int or not 0 <= l < num_layers for l in values)):
                raise ValueError("Plan layers must be distinct in-range integers")
        if set(weights) != {str(s) for s in sources}:
            raise ValueError("Plan weights must be keyed by exactly the source layers")
        if any(not math.isfinite(float(w)) for w in weights.values()):
            raise ValueError("Plan weights must be finite")
    return plan


def paper_plan(rr, k_source, k_target, *, rr_weighting=True, allowed_layers=None):
    """The original IRI layer choice: independent top-k by RR (Eqs. 3-4)."""
    plan = {}
    for component in COMPONENTS:
        sources = select_layers(rr[component], k_source, allowed_layers)
        targets = select_layers(rr[component], k_target, allowed_layers)
        plan[component] = {"sources": sources, "targets": targets,
                           "weights": {str(s): float(rr[component][s]) if rr_weighting else 1.0
                                       for s in sources}}
    return plan


def effective_edges(plan):
    """Source->target edges that can actually inject (g(k, l) = 1)."""
    return {c: {str(s): [t for t in plan[c]["targets"] if t > s] for s in plan[c]["sources"]}
            for c in COMPONENTS}


class PlannedIRI:
    def __init__(self, layers, plan, *, attention_scale, mlp_scale, normalize=True):
        self.layers = layers
        self.plan = validate_plan(plan, len(layers))
        self.scales = {"attention": float(attention_scale), "mlp": float(mlp_scale)}
        if not all(math.isfinite(v) for v in self.scales.values()):
            raise ValueError("IRI scales must be finite")
        self.normalize = normalize
        self.sources = {c: set(plan[c]["sources"]) for c in COMPONENTS}
        self.targets = {c: set(plan[c]["targets"]) for c in COMPONENTS}
        self.weights = {c: {int(k): float(v) for k, v in plan[c]["weights"].items()} for c in COMPONENTS}
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
                    buffers[component] = {}  # every forward (prefill and each decode step)
                buffer = buffers.setdefault(component, {})
                if index in self.sources[component]:
                    buffer[index] = tensor[:, -1, :].detach().clone()  # saved before injection
                if index in self.targets[component]:
                    sources = [(self.weights[component][k], value) for k, value in buffer.items() if k < index]
                    return _replace(output, inject(tensor, sources, self.scales[component], self.normalize))
            return hook

        try:
            for index, layer in enumerate(self.layers):
                for component in COMPONENTS:
                    if index == 0 or index in self.sources[component] or index in self.targets[component]:
                        handles.append(component_module(layer, component).register_forward_hook(hook_for(index, component)))
            yield
        finally:
            for handle in handles:
                handle.remove()
            buffers.clear()
            self._active = False


def load_plans(path: Path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["plans"]
