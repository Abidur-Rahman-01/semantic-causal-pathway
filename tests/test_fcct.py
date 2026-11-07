"""Numerical and real Transformers integration checks for independent FCCT/IRI."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image
from torch import nn

from fcct.core import COMPONENTS, TOKEN_GROUPS, IRI, capture, inject, patch_component, recovery_rate, select_layers
from fcct.data import prepare, validate_manifest
from fcct.experiment import fit_profile, open_run
from fcct.io import sha256, write_json, write_jsonl
from fcct.llava import Llava15, gaussian_corruption, object_patch_indices, token_groups
from fcct.report import binary_metrics, pope_answer, report


class MathTests(unittest.TestCase):
    def test_probability_recovery_is_not_clipped_or_log_based(self):
        self.assertAlmostEqual(recovery_rate(0.8, 0.2, 0.5), 0.5)
        self.assertAlmostEqual(recovery_rate(0.8, 0.2, 1.0), 4 / 3)
        self.assertAlmostEqual(recovery_rate(0.8, 0.2, 0.1), -1 / 6)
        self.assertIsNone(recovery_rate(0.2, 0.2, 0.5))
        self.assertIsNone(recovery_rate(0.1, 0.2, 0.5))
        for value in (float("nan"), float("inf"), -1, 2):
            with self.assertRaises(ValueError):
                recovery_rate(value, 0.2, 0.5)

    def test_iri_equations_preserve_individual_norms_and_prompt_tokens(self):
        base = torch.tensor([[[5., 8.], [3., 4.]], [[4., 1.], [0., 2.]]])
        source = torch.tensor([[1., 2.], [-3., 1.]])
        result = inject(base, [(0.7, source)], 0.26)
        manual = base[:, -1] + 0.26 * 0.7 * source
        manual *= base[:, -1].norm(dim=-1, keepdim=True) / manual.norm(dim=-1, keepdim=True)
        torch.testing.assert_close(result[:, -1], manual)
        torch.testing.assert_close(result[:, :-1], base[:, :-1], rtol=0, atol=0)
        torch.testing.assert_close(result[:, -1].norm(dim=-1), base[:, -1].norm(dim=-1))
        torch.testing.assert_close(inject(base, [(1., source)], 0), base, rtol=0, atol=0)

    def test_cancellation_does_not_create_nan(self):
        base = torch.tensor([[[1., 2.]]])
        torch.testing.assert_close(inject(base, [(1., -base[:, -1])], 1), base)

    def test_stable_independent_layer_ranking(self):
        self.assertEqual(select_layers([0.2, 0.7, 0.7, -0.1], 2), [1, 2])
        with self.assertRaises(ValueError):
            select_layers([0.1, float("nan")], 1)

    def test_noise_is_seeded_and_bounded(self):
        image = Image.new("RGB", (32, 32), (127, 127, 127))
        a = np.asarray(gaussian_corruption(image, 0.2, 17))
        b = np.asarray(gaussian_corruption(image, 0.2, 17))
        c = np.asarray(gaussian_corruption(image, 0.2, 18))
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))
        self.assertEqual(a.dtype, np.uint8)


class GeometryTests(unittest.TestCase):
    def test_seven_disjoint_groups_and_multiword_object(self):
        groups = token_groups(list(range(2, 11)), [2, 3, 5], [14, 15], 20)
        self.assertEqual(tuple(groups), TOKEN_GROUPS)
        self.assertEqual(groups["early_visual"], [2, 3])
        self.assertEqual(groups["object_visual"], [4, 5, 7])
        self.assertEqual(groups["late_visual"], [8, 9, 10])
        self.assertEqual(groups["early_text"], [11, 12, 13])
        self.assertEqual(groups["late_text"], [16, 17, 18])
        positions = [p for g in groups.values() for p in g]
        self.assertEqual(len(positions), len(set(positions)))
        with self.assertRaises(ValueError):
            token_groups([2, 3], [], [5], 8)

    def test_center_crop_does_not_ground_invisible_object(self):
        from transformers import CLIPImageProcessor
        processor = CLIPImageProcessor(size={"shortest_edge": 14}, crop_size={"height": 14, "width": 14})
        image = Image.new("RGB", (28, 14))
        self.assertEqual(object_patch_indices(image, [[0, 0, 5, 5]], processor, 7, (2, 2)), [])
        self.assertEqual(object_patch_indices(image, [[8, 0, 2, 2]], processor, 7, (2, 2)), [0])


class TinyTransformersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration
        torch.manual_seed(21)
        vision = CLIPVisionConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                                  num_attention_heads=2, image_size=28, patch_size=14, projection_dim=16)
        text = LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=24, num_hidden_layers=4,
                           num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
                           pad_token_id=0, bos_token_id=1, eos_token_id=2)
        config = LlavaConfig(vision_config=vision, text_config=text, image_token_index=31,
                             image_seq_length=4, vision_feature_layer=-2)
        config._attn_implementation = "eager"
        cls.model = LlavaForConditionalGeneration(config).eval()
        cls.layers = cls.model.model.language_model.layers
        cls.adapter = Llava15.__new__(Llava15)
        cls.adapter.model = cls.model
        cls.adapter.yes_token_id = 6
        cls.adapter.layers = cls.layers
        cls.clean = {"input_ids": torch.tensor([[1, 31, 31, 31, 31, 7, 8]]),
                     "attention_mask": torch.ones(1, 7, dtype=torch.long),
                     "pixel_values": torch.randn(1, 3, 28, 28)}
        cls.noisy = {**cls.clean, "pixel_values": torch.zeros(1, 3, 28, 28)}

    def test_fixed_vision_projection_equals_unmodified_model(self):
        expected = self.adapter.probability(self.clean)
        cached = self.adapter.trace_batch(self.clean)
        self.assertAlmostEqual(self.adapter.probability(cached), expected, places=7)

    def test_final_hidden_restoration_recovers_clean_probability(self):
        with capture(self.layers, {"last": [6]}) as (cache, positions):
            clean = self.adapter.probability(self.clean)
        corrupt = self.adapter.probability(self.noisy)
        self.assertNotEqual(clean, corrupt)
        with patch_component(self.layers[-1], "hidden", [6], cache[3, "hidden"]):
            patched = self.adapter.probability(self.noisy)
        self.assertAlmostEqual(clean, patched, places=7)
        self.assertAlmostEqual(self.adapter.probability(self.noisy), corrupt, places=7)

    def test_hooks_cleanup_on_exceptions_and_tuple_attention(self):
        baseline = self.adapter.probability(self.noisy)
        with capture(self.layers, {"last": [6]}) as (cache, _):
            self.adapter.probability(self.clean)
        for component in COMPONENTS:
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with patch_component(self.layers[1], component, [6], cache[1, component]):
                    self.adapter.probability(self.noisy)
                    raise RuntimeError("intentional")
            self.assertFalse(self.layers[1]._forward_hooks)
            self.assertFalse(self.layers[1].self_attn._forward_hooks)
            self.assertFalse(self.layers[1].mlp._forward_hooks)
        self.assertEqual(self.adapter.probability(self.noisy), baseline)

    def test_iri_generation_cache_resets_for_every_forward(self):
        rr = {"attention": [0.8, 0.7, 0.6, 0.5], "mlp": [0.9, 0.8, 0.6, 0.4]}
        iri = IRI(self.layers, rr, k_source=2, k_target=4)
        before = self.adapter.probability(self.clean)
        with iri.apply():
            first = self.adapter.probability(self.clean)
            self.adapter.probability(self.noisy)
            second = self.adapter.probability(self.clean)
            with torch.inference_mode():
                generated = self.model.generate(**self.clean, max_new_tokens=3, do_sample=False, use_cache=True)
            self.assertGreater(generated.shape[1], self.clean["input_ids"].shape[1])
        self.assertEqual(first, second)
        self.assertNotEqual(first, before)
        self.assertEqual(before, self.adapter.probability(self.clean))


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def test_resume_rejects_changed_inputs(self):
        manifest, output = self.path / "manifest.jsonl", self.path / "trace.jsonl"
        write_jsonl(manifest, [{"sample_id": "a"}])
        self.assertEqual(open_run(output, manifest, {"seed": 17}, ROOT), [])
        write_jsonl(output, [{"sample_id": "a"}])
        self.assertEqual(len(open_run(output, manifest, {"seed": 17}, ROOT)), 1)
        with self.assertRaisesRegex(ValueError, "Resume rejected"):
            open_run(output, manifest, {"seed": 18}, ROOT)

    def test_partial_profile_requires_explicit_flag_and_keeps_image_balance(self):
        manifest, trace, profile = (self.path / n for n in ("m.jsonl", "t.jsonl", "p.json"))
        write_jsonl(manifest, [{"sample_id": s, "image_id": i} for s, i in (("a", 1), ("b", 1), ("c", 2), ("d", 3))])
        rows = []
        for name, image_id, rr in (("a", 1, 0.), ("b", 1, 1.), ("c", 2, 1.)):
            rows.append({"sample_id": name, "image_id": image_id, "status": "complete", "patches": [
                {"component": component, "group": group, "layer": layer, "rr": rr}
                for component in COMPONENTS for group in TOKEN_GROUPS for layer in range(32)]})
        write_jsonl(trace, rows)
        write_json(trace.with_suffix(".meta.json"), {"manifest_sha256": sha256(manifest),
                   "config": {"model_id": "test", "revision": "a" * 40}})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            fit_profile(trace, manifest, profile)
        fit_profile(trace, manifest, profile, allow_partial=True)
        p = json.loads(profile.read_text())
        self.assertFalse(p["complete"])
        self.assertEqual(p["trace_image_ids"], [1, 2, 3])
        self.assertEqual(p["rr"]["attention"][0], 0.75)

    def test_pope_matches_upstream_parser_and_confusion_counts(self):
        self.assertEqual(pope_answer("No, it is absent. Yes later."), "no")
        self.assertEqual(pope_answer("There is not a dog."), "no")
        self.assertEqual(pope_answer("Yes. No later."), "yes")
        # Empty text is yes under upstream's rule; count it explicitly.
        self.assertEqual(pope_answer(""), "yes")
        rows = [{"label": label, "answers": {"regular": answer}} for label, answer in
                (("yes", "Yes"), ("no", "No"), ("yes", "No"), ("no", "Yes"))]
        result = binary_metrics(rows, "regular")
        self.assertEqual((result["tp"], result["tn"], result["fp"], result["fn"]), (1, 1, 1, 1))
        self.assertEqual(result["accuracy"], 50.)
        self.assertEqual(result["f1"], 50.)

    def test_partial_report_cannot_pass_paper_agreement(self):
        config = json.loads((ROOT / "configs/fcct_llava15.json").read_text())
        manifest, results, profile = (self.path / n for n in ("m.jsonl", "r.jsonl", "p.json"))
        samples = [{"sample_id": f"{split}-{i}", "image_id": i, "question": "fixture question",
                    "label": "yes" if i % 2 else "no", "split": split, "image_sha256": "test-fixture"}
                   for split in ("random", "popular", "adversarial") for i in range(3000)]
        write_jsonl(manifest, samples)
        write_jsonl(results, [{**row, "answers": {"regular": row["label"], "iri": row["label"]}}
                             for row in samples if row["image_id"] < 2])
        write_json(profile, {"complete": False})
        write_json(results.with_suffix(".meta.json"), {"manifest_sha256": sha256(manifest),
                   "extra": {"profile_sha256": sha256(profile)}, "config": config})
        with patch("fcct.report.plot_profile"):
            summary = report(results, manifest, profile, ROOT, self.path / "report", tolerance=100)
        self.assertEqual(summary["status"], "partial_engineering_run")
        self.assertFalse(summary["exact_reproduction"])
        self.assertIsNone(summary["all_pope_metrics_within_tolerance"])
        # A reference mismatch must fail before computing apparently good scores.
        bad = [{**samples[0], "label": "yes", "answers": {"regular": "yes", "iri": "yes"}}]
        write_jsonl(results, bad)
        with self.assertRaisesRegex(ValueError, "do not match"):
            report(results, manifest, profile, ROOT, self.path / "report")

    def test_dataset_preparation_is_deterministic_disjoint_and_checks_assets(self):
        config = {"data_root": "data/fcct", "seed": 17, "trace_images": 1, "questions_per_image": 6}
        root = self.path / config["data_root"]
        sources = root / "sources"
        images = self.path / "data/raw/vqav2/val2014"
        images.mkdir(parents=True)
        metadata = []
        for image_id in (1, 2, 3):
            name = f"COCO_val2014_{image_id:012d}.jpg"
            Image.new("RGB", (8, 8), "white").save(images / name)
            metadata.append({"id": image_id, "file_name": name, "width": 8, "height": 8})
        write_json(sources / "instances_val2014.json", {
            "images": metadata, "categories": [{"id": 1, "name": "cat"}],
            "annotations": [{"image_id": image_id, "category_id": 1, "bbox": [0, 0, 8, 8]}
                            for image_id in (1, 2, 3)]})
        for split in ("random", "popular", "adversarial"):
            write_jsonl(sources / f"coco_pope_{split}.json", [
                {"question_id": i, "image": metadata[0]["file_name"], "text": "Is there a cat?", "label": "yes"}
                for i in range(3000)])
        card = prepare(config, self.path)
        first = (root / "trace.jsonl").read_bytes()
        self.assertEqual(card["tracing_images"], 1)
        self.assertEqual(card["overlap_images"], 0)
        self.assertEqual(card["pope_questions"], 9000)
        prepare(config, self.path)
        self.assertEqual(first, (root / "trace.jsonl").read_bytes())
        row = json.loads(first)
        self.assertNotEqual(row["image_id"], 1)
        Image.new("RGB", (8, 8), "black").save(root / row["image"])
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_manifest([row], root, "trace")

    def test_interrupted_download_is_retried_atomically(self):
        import requests
        from scripts import setup_fcct
        from unittest.mock import Mock

        class Response:
            headers = {"Content-Length": "4"}
            def __init__(self, interrupted=False):
                self.interrupted = interrupted
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def raise_for_status(self):
                pass
            def iter_content(self, size):
                if self.interrupted:
                    yield b"bad"
                    raise requests.ConnectionError("fixture interrupted body")
                yield b"good"

        session = Mock()
        session.get.side_effect = [Response(True), Response()]
        destination = self.path / "download.bin"
        with patch.object(setup_fcct, "HTTP_SESSIONS", SimpleNamespace(session=session)), patch.object(setup_fcct.time, "sleep"):
            result = setup_fcct.download("https://example.invalid/fixture", destination)
        self.assertEqual(destination.read_bytes(), b"good")
        self.assertEqual(result["bytes"], 4)
        self.assertEqual(session.get.call_count, 2)
        self.assertFalse(destination.with_suffix(".bin.part").exists())

    def test_truncated_download_never_becomes_a_complete_asset(self):
        import requests
        from scripts import setup_fcct
        from unittest.mock import MagicMock, Mock

        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = {"Content-Length": "8"}
        response.iter_content.side_effect = lambda size: iter([b"short"])
        session = Mock()
        session.get.return_value = response
        destination = self.path / "truncated.bin"
        with patch.object(setup_fcct, "HTTP_SESSIONS", SimpleNamespace(session=session)), patch.object(setup_fcct.time, "sleep"):
            with self.assertRaises(requests.exceptions.ChunkedEncodingError):
                setup_fcct.download("https://example.invalid/fixture", destination)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
