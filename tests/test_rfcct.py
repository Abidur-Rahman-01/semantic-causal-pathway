"""Tests for the R-FCCT extension: statistics, operator equivalence, and a full
tiny-model pipeline (prepare -> trace -> select -> tune -> freeze -> evaluate
-> sample -> report) on CPU."""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image

from fcct.core import IRI
from fcct.io import read_jsonl, sha256, write_json, write_jsonl
from rfcct import pipeline
from rfcct.data import NegativeSampler, prepare
from rfcct.plan import PlannedIRI, effective_edges, paper_plan, validate_plan
from rfcct.report import build_report
from rfcct.stats import (best_threshold, bootstrap_scores, cluster_bootstrap_delta, confusion, holm,
                         image_tables, inclusion_frequency, mcnemar_exact, paper_rr_profile, ratio_scores,
                         reachable_plan, stable_plan, threshold_for_rate)
from rfcct.tracing import contrastive_trace_sample, score


def record(sample_id, image_id, label, s_clean, s_corr, patched, layers=4, p=None):
    p = p or {"clean": 0.9, "corr": 0.5, "patched": [0.7] * layers}
    return {"sample_id": sample_id, "image_id": image_id, "label": label, "status": "complete",
            "s_clean": s_clean, "s_corrupted": s_corr, "gap": s_clean - s_corr,
            "s_patched": {"attention": patched, "mlp": patched},
            "p_yes_clean": p["clean"], "p_yes_corrupted": p["corr"],
            "p_yes_patched": {"attention": p["patched"], "mlp": p["patched"]}}


class StatsTests(unittest.TestCase):
    def test_ratio_of_image_balanced_means(self):
        records = [record("a", 1, "yes", 3, 1, [1, 2, 3, 3]),      # d=2, n=[0,1,2,2]
                   record("b", 1, "yes", 5, 1, [1, 1, 1, 5]),      # d=4, n=[0,0,0,4]
                   record("c", 2, "yes", 2, 1, [2, 2, 2, 2]),      # d=1, n=[1,1,1,1]
                   record("d", 2, "no", 2, 0, [0, 1, 2, 2])]       # d=2, n=[0,1,2,2]
        ids, N, D, C = image_tables(records, "attention", 4)
        self.assertEqual(ids, [1, 2])
        # yes: image1 mean n=[0,.5,1,3], mean d=3; image2 n=[1,1,1,1], d=1 -> ratio (sum n)/(sum d)=n/4
        yes = ratio_scores(N, D, C, classes=("yes",))
        np.testing.assert_allclose(yes, [0.25, 0.375, 0.5, 1.0])
        no = ratio_scores(N, D, C, classes=("no",))
        np.testing.assert_allclose(no, [0, 0.5, 1, 1])
        np.testing.assert_allclose(ratio_scores(N, D, C), (yes + no) / 2)
        np.testing.assert_allclose(ratio_scores(N, D, C, combine="min"), np.minimum(yes, no))
        boot = bootstrap_scores(N, D, C, draws=50, seed=1)
        self.assertEqual(boot.shape, (50, 4))
        self.assertTrue(np.isfinite(boot).all())

    def test_nonpositive_gap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive margin drop"):
            image_tables([record("a", 1, "yes", 1, 1, [1] * 4)], "attention", 4)

    def test_paper_rr_profile_filters_to_measurable_drops(self):
        recs = [record("a", 1, "yes", 2, 1, [1] * 4, p={"clean": .9, "corr": .5, "patched": [.7] * 4}),
                record("b", 2, "yes", 2, 1, [1] * 4, p={"clean": .5, "corr": .49, "patched": [.9] * 4}),
                record("c", 3, "no", 2, 1, [1] * 4)]
        rr, images = paper_rr_profile(recs, "attention", 4, 0.05)
        self.assertEqual(images, 1)
        np.testing.assert_allclose(rr, [0.5] * 4)

    def test_reachable_plan_never_wastes_sources(self):
        score = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.95])
        plan = reachable_plan(score, 2, 3)
        self.assertTrue(all(any(t > s for t in plan["targets"]) for s in plan["sources"]))
        self.assertNotIn(5, plan["sources"])  # last layer has no later target
        # Without reachability the FCCT/IRI independent top-k rule is recovered.
        plain = reachable_plan(score, 2, 3, reachable=False)
        self.assertEqual(plain["sources"], [3, 5])
        self.assertEqual(plain["targets"], [3, 4, 5])

    def test_stable_plan_requires_positive_lcb(self):
        rng = np.random.default_rng(0)
        boot = np.stack([rng.normal([1, 1, -1, 0.5, 0.2, 0.1], 0.05) for _ in range(200)])
        plan = stable_plan(boot.mean(0), boot, 2, 3)
        self.assertNotIn(2, plan["sources"])
        self.assertTrue(all(plan["lcb"][s] > 0 for s in plan["sources"]))
        self.assertTrue(all(plan["weights"][str(s)] == plan["lcb"][s] for s in plan["sources"]))
        freq = inclusion_frequency(boot, 2)
        self.assertAlmostEqual(freq[0] + freq[1], 2.0)

    def test_mcnemar_holm_and_thresholds(self):
        a = np.array([True] * 10 + [False] * 10)
        b = np.array([True] * 20)
        self.assertAlmostEqual(mcnemar_exact(a, b)["p_value"], 2 * 0.5 ** 10)
        self.assertEqual(mcnemar_exact(a, a)["p_value"], 1.0)
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        margins = np.array([3., 2., 1., -1., -2.])
        tau = threshold_for_rate(margins, 0.4)
        self.assertAlmostEqual((margins > tau).mean(), 0.4)
        labels = np.array([1, 1, 0, 0, 0])
        tau = best_threshold(margins, labels)
        self.assertEqual(confusion(labels, margins > tau)["accuracy"], 100.0)
        with self.assertRaises(ValueError):
            best_threshold(np.array([1., 2.]), np.array([0, 0]), min_recall=50)

    def test_cluster_bootstrap_of_identical_predictions_is_zero(self):
        labels = np.array([1, 0, 1, 0])
        pred = np.array([1, 1, 0, 0])
        ci = cluster_bootstrap_delta([1, 1, 2, 2], labels, pred, pred, draws=100, seed=0)
        self.assertEqual(ci["accuracy"], [0.0, 0.0])

    def test_negative_sampler_strategies(self):
        freq = {1: 10, 2: 9, 3: 1, 4: 5}
        cooc = {1: {3: 7, 4: 1}, 2: {}, 3: {1: 7}, 4: {1: 1}}
        from collections import Counter
        sampler = NegativeSampler([1, 2, 3, 4], freq, {k: Counter(v) for k, v in cooc.items()}, 0)
        self.assertEqual(sampler.sample("popular", {1}, set()), 2)
        self.assertEqual(sampler.sample("adversarial", {1}, set()), 3)
        self.assertIn(sampler.sample("random", {1}, {2, 3}), {4})


class TinyModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration
        torch.manual_seed(5)
        vision = CLIPVisionConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                                  num_attention_heads=2, image_size=28, patch_size=14, projection_dim=16)
        text = LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=24, num_hidden_layers=4,
                           num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
                           pad_token_id=0, bos_token_id=1, eos_token_id=2)
        config = LlavaConfig(vision_config=vision, text_config=text, image_token_index=31,
                             image_seq_length=4, vision_feature_layer=-2)
        config._attn_implementation = "eager"
        cls.model = LlavaForConditionalGeneration(config).eval()
        cls.layers = cls.model.model.language_model.layers
        cls.ids = {"yes": 6, "no": 7}

        from fcct.llava import Llava15
        backend = Llava15.__new__(Llava15)
        backend.model, backend.layers, backend.yes_token_id, backend.patch_size = cls.model, cls.layers, 6, 14

        def batch(image, question, object_name=None):
            array = np.asarray(image.convert("RGB").resize((28, 28)), dtype=np.float32).copy() / 255.
            # Question length changes the prompt so different rows are different inputs.
            digest = __import__("hashlib").sha256(question.encode()).digest()
            tokens = [1, 31, 31, 31, 31] + [8 + b % 22 for b in digest[:6]] + [9]
            return {"input_ids": torch.tensor([tokens]), "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
                    "pixel_values": torch.from_numpy(array).permute(2, 0, 1)[None]}, {"image_positions": [1, 2, 3, 4]}
        backend.batch = batch
        # Centre the Yes/No direction on the spread of final hidden states so that the
        # tiny model answers both Yes and No (exercises both contrastive classes).
        rng = np.random.default_rng(3)
        with torch.no_grad():
            states = []
            for name in ("cat", "dog", "car", "bus", "cup", "fork", "tree", "boat"):
                image = Image.fromarray(rng.integers(0, 255, (28, 28, 3), dtype=np.uint8))
                out = cls.model(**batch(image, f"Is there a {name}?")[0], output_hidden_states=True)
                states.append(out.hidden_states[-1][0, -1])
            states = torch.stack(states)
            direction = states[0] - states.mean(0)
            head = cls.model.lm_head.weight
            head[6] = 20 * direction / direction.norm()
            head[7] = -head[6]
            # Remove the shared component so the margin is centred near zero.
            offset = float((head[6] - head[7]) @ states.mean(0))
            head[6] -= offset / 2 * states.mean(0) / states.mean(0).norm() ** 2
            head[7] = -head[6]
        backend.processor = SimpleNamespace(tokenizer=SimpleNamespace(
            decode=lambda ids, skip_special_tokens=True: " ".join("Yes" if int(i) % 2 else "No" for i in ids)))
        cls.backend = backend
        cls.clean = batch(Image.new("RGB", (28, 28), "white"), "q")[0]


class OperatorTests(TinyModel):
    def test_planned_iri_with_paper_plan_equals_fcct_iri(self):
        rr = {"attention": [0.3, 0.9, 0.5, 0.7], "mlp": [0.8, 0.2, 0.6, 0.4]}
        reference = IRI(self.layers, rr, k_source=2, k_target=3, attention_scale=0.26, mlp_scale=0.16)
        planned = PlannedIRI(self.layers, paper_plan(rr, 2, 3), attention_scale=0.26, mlp_scale=0.16)
        with reference.apply():
            expected = self.model(**self.clean, use_cache=False).logits
            gen_expected = self.model.generate(**self.clean, max_new_tokens=4, do_sample=False)
        with planned.apply():
            actual = self.model(**self.clean, use_cache=False).logits
            gen_actual = self.model.generate(**self.clean, max_new_tokens=4, do_sample=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(gen_actual, gen_expected))
        baseline = self.model(**self.clean, use_cache=False).logits
        self.assertFalse(torch.equal(baseline, actual))
        self.assertFalse(self.layers[1].self_attn._forward_hooks)

    def test_plan_validation_and_edges(self):
        plan = paper_plan({"attention": [0.1, 0.2, 0.9, 0.8], "mlp": [0.1, 0.2, 0.3, 0.9]}, 2, 2)
        self.assertEqual(effective_edges(plan)["mlp"], {"2": [3], "3": []})
        bad = json.loads(json.dumps(plan))
        bad["mlp"]["weights"] = {}
        with self.assertRaises(ValueError):
            validate_plan(bad, 4)

    def test_contrastive_trace_on_tiny_decoder(self):
        image = Image.new("RGB", (28, 28), "white")
        clean = score(self.model, self.clean, 6, 7)["margin"]
        label = "yes" if clean > 0 else "no"
        row = {"sample_id": "x", "image_id": 1, "label": label, "question": "q"}
        result = contrastive_trace_sample(self.backend, image, row, yes_id=6, no_id=7, sigma=0.5, seed=1,
                                          min_margin_gap=-math.inf)
        self.assertEqual(result["status"], "complete")
        self.assertGreater(result["s_clean"], 0)
        self.assertEqual(len(result["s_patched"]["attention"]), 4)
        # Restoring the final MLP *and* attention output equals the clean final hidden only
        # together; each single patch must still lie strictly between finite values.
        self.assertTrue(all(math.isfinite(v) for v in result["s_patched"]["mlp"]))
        strict = contrastive_trace_sample(self.backend, image, row, yes_id=6, no_id=7, sigma=0.5, seed=1,
                                          min_margin_gap=1e9)
        self.assertEqual(strict["status"], "excluded_no_measurable_drop")


class PipelineTests(TinyModel):
    """End-to-end on CPU: every stage, resumption, and the final report."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        for rel in ("src/fcct", "src/rfcct"):
            (self.repo / rel).mkdir(parents=True)
            for f in (ROOT / rel).glob("*.py"):
                (self.repo / rel / f.name).write_bytes(f.read_bytes())
        (self.repo / "rfcct.py").write_bytes((ROOT / "rfcct.py").read_bytes())
        (self.repo / "configs").mkdir()
        (self.repo / "configs/fcct_paper_targets.json").write_bytes((ROOT / "configs/fcct_paper_targets.json").read_bytes())
        self._fake_data()

    def _fake_data(self):
        root = self.repo / "data/fcct"
        (root / "images").mkdir(parents=True)
        cats = [{"id": i, "name": n} for i, n in enumerate(["cat", "dog", "car", "bus", "cup", "fork"], 1)]
        meta, anns, rng = [], [], np.random.default_rng(0)
        for image_id in range(1, 31):
            name = f"COCO_val2014_{image_id:012d}.jpg"
            Image.fromarray(rng.integers(0, 255, (28, 28, 3), dtype=np.uint8)).save(root / "images" / name)
            meta.append({"id": image_id, "file_name": name, "width": 28, "height": 28,
                         "coco_url": f"http://images.cocodataset.org/val2014/{name}"})
            for c in (1 + image_id % 6, 1 + (image_id + 1) % 6, 1 + (image_id + 3) % 6):
                anns.append({"image_id": image_id, "category_id": c, "bbox": [0, 0, 28, 28], "iscrowd": 0})
        write_json(root / "sources/instances_val2014.json", {"images": meta, "categories": cats, "annotations": anns})
        h = lambda i: sha256(root / "images" / meta[i - 1]["file_name"])
        trace = [{"sample_id": f"trace-{i}-{c}", "image_id": i, "image": f"images/{meta[i - 1]['file_name']}",
                  "image_sha256": h(i), "object": cats[c - 1]["name"], "boxes": [[0, 0, 28, 28]],
                  "question": f"Is there a {cats[c - 1]['name']} in the image? Please answer this question with one word (Yes or No)."}
                 for i in range(1, 13) for c in (1 + i % 6, 1 + (i + 1) % 6)]
        pope = []
        for split in ("random", "popular", "adversarial"):
            for i in (21, 22, 23, 24):
                for c, label in ((1 + i % 6, "yes"), (1 + (i + 2) % 6, "no")):
                    pope.append({"sample_id": f"pope-{split}-{i}-{c}", "image_id": i,
                                 "image": f"images/{meta[i - 1]['file_name']}", "image_sha256": h(i),
                                 "question": f"Is there a {cats[c - 1]['name']} in the image?", "label": label,
                                 "split": split})
        write_jsonl(root / "trace.jsonl", trace)
        write_jsonl(root / "pope.jsonl", pope)
        self.base = {"data_root": "data/fcct", "output_root": "outputs/fcct", "model_id": "tiny", "revision": "a" * 40,
                     "dtype": "float32", "k_source": 2, "k_target": 3, "attention_scale": 0.26, "mlp_scale": 0.16,
                     "normalize": True}
        write_json(self.repo / "outputs/fcct/profile.json",
                   {"rr": {"attention": [0.3, 0.9, 0.5, 0.7], "mlp": [0.8, 0.2, 0.6, 0.4]}})
        self.rc = json.loads((ROOT / "configs/rfcct_llava15.json").read_text())
        self.rc.update(noise={**self.rc["noise"], "sigma_grid": [0.3, 0.9], "calibration_questions": 6,
                              "min_median_gap": 0.0, "min_drop_rate": 0.0},
                       trace={"negatives_per_positive": 1, "min_margin_gap": 1e-6},
                       calibration_set={"images": 5, "positives": 2, "negatives": ["random", "adversarial"]},
                       selection={**self.rc["selection"], "k_source": 2, "k_target": 3, "bootstrap": 50,
                                  "paper_min_probability_gap": -1.0},
                       tuning={**self.rc["tuning"], "attention_grid": [0.0, 0.3], "mlp_grid": [0.0, 0.2],
                               "max_recall_drop_pp": 100.0},
                       sampling={**self.rc["sampling"], "seeds": [0, 1]},
                       report={**self.rc["report"], "bootstrap": 200}, max_new_tokens=3)

    def test_full_pipeline(self):
        rc, base, repo = self.rc, self.base, self.repo
        with patch("rfcct.pipeline.NUM_LAYERS", 4), patch("rfcct.tracing.object_visible", return_value=True):
            card = prepare(rc, base, repo)
            self.assertEqual(sum(card["overlaps"].values()), 0)
            self.assertEqual(card["contrastive_trace"]["labels"], {"yes": 24, "no": 24})
            first = (repo / "data/rfcct/calibration.jsonl").read_bytes()
            prepare(rc, base, repo)
            self.assertEqual(first, (repo / "data/rfcct/calibration.jsonl").read_bytes())

            import importlib.util
            spec = importlib.util.spec_from_file_location("rfcct_cli", ROOT / "rfcct.py")
            cli = importlib.util.module_from_spec(spec)
            with patch.dict(os.environ, {}), patch.object(sys, "path", list(sys.path)):
                spec.loader.exec_module(cli)
            cli.ROOT = repo  # the CLI's path layout, rooted in the temporary repository
            paths = cli.paths_for(rc, base)

            noise = pipeline.run_noise(self.backend, self.ids, rc, base, repo, paths)
            self.assertIn(noise["chosen_sigma"], (0.3, 0.9))
            # Interrupted tracing resumes to the same records.
            pipeline.run_trace(self.backend, self.ids, rc, base, repo, paths, limit=5)
            self.assertEqual(len(read_jsonl(paths["trace"])), 5)
            out = pipeline.run_trace(self.backend, self.ids, rc, base, repo, paths)
            self.assertEqual(out["completed"], 48)
            status = out["status"]
            self.assertGreater(status.get("complete", 0), 4, status)

            plans = pipeline.run_select(rc, base, repo, paths)["plans"]
            self.assertEqual(set(plans), set(rc["variants"]))
            for name in ("riri", "riri_present_only", "riri_no_stability"):
                for c in ("attention", "mlp"):
                    p = plans[name][c]
                    self.assertTrue(all(any(t > s for t in p["targets"]) for s in p["sources"]))

            pipeline.run_tune(self.backend, self.ids, rc, base, repo, paths)
            frozen = pipeline.freeze(rc, base, paths)
            self.assertEqual(set(frozen), {f"{v}@tuned" for v in rc["tuning"]["variants"]})

            ev = pipeline.run_evaluate(self.backend, self.ids, rc, base, repo, paths)
            self.assertEqual(ev["unique_prompts"], 8)  # 24 POPE rows share 8 distinct prompts
            sm = pipeline.run_evaluate(self.backend, self.ids, rc, base, repo, paths, sampling=True)
            self.assertEqual(sm["completed"], 16)
            # Sampling is reproducible: a rerun from scratch gives identical generations.
            first_sample = paths["sample"].read_bytes()
            paths["sample"].unlink()
            pipeline.run_evaluate(self.backend, self.ids, rc, base, repo, paths, sampling=True)
            self.assertEqual(first_sample, paths["sample"].read_bytes())

            out = build_report(rc, base, repo, paths)
            result = json.loads((paths["report_dir"] / "results.json").read_text())
            self.assertTrue((paths["report_dir"] / "report.md").is_file())
            self.assertEqual(result["greedy"]["random"]["regular"]["n"], 8)
            self.assertIn("threshold_control", result["greedy"]["random"])
            families = {c["family"] for c in result["comparisons"]}
            self.assertEqual(families, {"primary", "secondary"})
            self.assertTrue(all(0 <= c["p_holm"] <= 1 for c in result["comparisons"]))
            self.assertEqual(len(result["sampling"]["seeds"]), 2)

            # Changing a frozen setting invalidates resumption instead of mixing results.
            rc2 = json.loads(json.dumps(rc))
            rc2["tuning"]["attention_grid"] = [0.0, 0.5]
            with self.assertRaisesRegex(ValueError, "Resume rejected"):
                pipeline.run_tune(self.backend, self.ids, rc2, base, repo, paths)


if __name__ == "__main__":
    unittest.main()
