"""Official metric definitions for every benchmark in the paper.

* POPE   - AoiDragon/POPE parser (``fcct.report.pope_answer``); Acc, Precision,
           Recall, F1, yes-ratio.
* MME    - original MME ``calculation.py`` (Otter parse): per subtask
           acc + acc+ (both questions of an image correct), each in percent.
* CHAIR  - OPERA ``chair.py`` (pinned by SHA-256, imported unmodified):
           CHAIR_s, CHAIR_i, Recall, Len, printed x100 exactly like
           ``print_metrics`` (Len x100 = mean tokens per caption).
* MMHal  - ``eval_gpt4.py`` judge prompt (extracted verbatim from the pinned
           file), ``rating: k`` parse, hallucination = rating < 3, per-type by
           position ``i % 8``.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import pickle
from collections import defaultdict
from pathlib import Path

from fcct.report import pope_answer

from .data import MME_COGNITION, MME_HALLUCINATION, MME_PERCEPTION

CHAIR_SHA256 = "84ea67bd831f7dc8ed605f5d3428cfb3e8f5dad3f4b73e1d2907a01c66a4af86"
MMHAL_EVAL_SHA256 = "5ba718be65d03cb9e80321ad9eb08b37a0eb0040372e1fa20f78eaf525fe8f7c"


def pope_metrics(labels, texts):
    tp = tn = fp = fn = 0
    for label, text in zip(labels, texts):
        pred = pope_answer(text)
        tp += pred == label == "yes"
        tn += pred == label == "no"
        fp += pred == "yes" and label == "no"
        fn += pred == "no" and label == "yes"
    n = tp + tn + fp + fn
    return {"n": n, "accuracy": 100 * (tp + tn) / n, "precision": 100 * tp / (tp + fp) if tp + fp else 0.0,
            "recall": 100 * tp / (tp + fn) if tp + fn else 0.0,
            "f1": 200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "yes_ratio": 100 * (tp + fp) / n, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def mme_parse(text):
    """Original MME/Otter parse (``calculation.py``)."""
    answer = text.lower().strip().replace(".", "")
    if answer in ("yes", "no"):
        return answer
    prefix = answer[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return "other"


def mme_scores(rows, texts):
    """Per-subtask score = acc + acc+ (max 200) and the paper's aggregates."""
    by_pair = defaultdict(list)
    for row, text in zip(rows, texts):
        by_pair[row["category"], row["question_id"]].append(mme_parse(text) == row["label"])
    per_category = defaultdict(list)
    for (category, _), correct in by_pair.items():
        if len(correct) != 2:
            raise ValueError("MME scoring requires exactly two questions per image")
        per_category[category].append(correct)
    scores = {}
    for category, pairs in per_category.items():
        acc = 100 * sum(sum(p) for p in pairs) / (2 * len(pairs))
        acc_plus = 100 * sum(all(p) for p in pairs) / len(pairs)
        scores[category] = acc + acc_plus
    def total(names):
        return sum(scores[c] for c in names if c in scores)
    return {"subtasks": scores, "hallucination_total": total(MME_HALLUCINATION),
            "perception_total": total(MME_PERCEPTION), "cognition_total": total(MME_COGNITION)}


def _load_pinned(path: Path, expected, name):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"{path} does not match the pinned official file (sha256 {digest})")
    import sys
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # needed so the evaluator cache can be pickled/unpickled
    spec.loader.exec_module(module)
    return module


def chair_metrics(chair_py: Path, coco_annotations: Path, cache: Path, captions_jsonl: Path):
    """Run OPERA's CHAIR evaluator unmodified; returns values as it prints them."""
    module = _load_pinned(chair_py, CHAIR_SHA256, "opera_chair")
    if cache.is_file():
        evaluator = pickle.loads(cache.read_bytes())
    else:
        evaluator = module.CHAIR(str(coco_annotations))
        cache.write_bytes(pickle.dumps(evaluator))
    output = evaluator.compute_chair(str(captions_jsonl), "image_id", "caption")
    overall = output["overall_metrics"]
    return {"CHAIRs": 100 * overall["CHAIRs"], "CHAIRi": 100 * overall["CHAIRi"],
            "Recall": 100 * overall["Recall"], "Len": 100 * overall["Len"],
            "captions": len(output["sentences"])}


def mmhal_template(eval_py: Path):
    """The judge prompt, read as data from the pinned script (never executed)."""
    source = Path(eval_py).read_bytes()
    if hashlib.sha256(source).hexdigest() != MMHAL_EVAL_SHA256:
        raise ValueError("eval_gpt4.py does not match the pinned official file")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "template" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Judge template not found in eval_gpt4.py")


def mmhal_prompt(template, item, answer):
    return template.format(", ".join(item["image_content"]), item["question"], item["gt_answer"], answer)


def mmhal_rating(response):
    found = [s for s in range(7) if f"rating: {s}" in response.lower()]
    return found[0] if len(found) == 1 else 0  # eval_gpt4.py: zero or multiple -> 0


def mmhal_metrics(ratings):
    if len(ratings) != 96:
        raise ValueError("MMHal-Bench scoring needs all 96 ratings in the original order")
    per_type = [[ratings[i] for i in range(96) if i % 8 == t] for t in range(8)]
    return {"score": sum(ratings) / 96, "hallucination_rate": 100 * sum(r < 3 for r in ratings) / 96,
            "per_type_score": [round(sum(v) / len(v), 2) for v in per_type]}


def nltk_ready(nltk_dir: Path):
    os.environ["NLTK_DATA"] = str(nltk_dir)
    import nltk
    if str(nltk_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_dir))
    return nltk
