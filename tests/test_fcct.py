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
from fcct.experiment import fit_profile, load_iri, open_run, trace_sample
from fcct.io import fingerprint, sha256, write_json, write_jsonl
from fcct.llava import Llava15, gaussian_corruption, object_patch_indices, token_groups, token_coverage
from fcct.attention import attention_statistics, summarize_attention
from fcct.ablations import ablation_settings
from fcct.report import binary_metrics, pope_answer, report


class MathTests(unittest.TestCase):
    def test_probability_recovery_is_not_clipped_or_log_based(self):
        self.assertAlmostEqual(recovery_rate(0.8, 0.2, 0.5), 0.5)
        self.assertAlmostEqual(recovery_rate(0.8, 0.2, 1.0), 4 / 3)
        self.assertAlmostEqual(recovery_rate(0.8, 0.2, 0.1), -1 / 6)
        self.assertIsNone(recovery_rate(0.2, 0.2, 0.5))
        self.assertAlmostEqual(recovery_rate(0.1, 0.2, 0.5), -3.0)
        self.assertIsNone(recovery_rate(0.1, 0.2, 0.5, gap_policy="positive_only"))
        self.assertIsNone(recovery_rate(0.20001, 0.2, 0.5, min_gap=0.001))
        self.assertIsNotNone(recovery_rate(0.20001, 0.2, 0.5))
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

    def test_layer_range_control_respects_window_and_strict_causal_order(self):
        rr = {"attention": [0.9, 0.8, 0.7, 0.6], "mlp": [0.9, 0.8, 0.7, 0.6]}
        layer = lambda: SimpleNamespace(self_attn=nn.Identity(), mlp=nn.Identity())
        iri = IRI([layer() for _ in range(4)], rr, k_source=1, k_target=2,
                  allowed_layers=[2, 3], rr_weighting=False)
        self.assertEqual(iri.sources["attention"], [2])
        self.assertEqual(iri.targets["attention"], [2, 3])
        with self.assertRaises(ValueError):
            select_layers([0.9, 0.8], 2, allowed_layers=[1])
        with self.assertRaises(ValueError):
            select_layers([0.9, 0.8], 1, allowed_layers=[2])

    def test_no_normalization_control_has_the_unscaled_vector_norm(self):
        original = torch.tensor([[[3., 4.]]])
        result = inject(original, [(0.5, torch.tensor([[4., 0.]]))], 2.0, normalize=False)
        torch.testing.assert_close(result, torch.tensor([[[7., 4.]]]))
        self.assertNotEqual(float(result.norm()), float(original.norm()))

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
        coverage = token_coverage(groups, list(range(2, 11)), 20)
        self.assertEqual(coverage["unassigned_visual_positions"], [6])
        self.assertEqual(coverage["unassigned_text_positions"], [0, 1])

    def test_external_assignments_require_exact_encoded_prompt_checksum(self):
        backend = Llava15.__new__(Llava15)
        ids = torch.tensor([[1, 31, 31, 31, 4, 5, 6, 7]])
        groups = token_groups([1, 2, 3], [1], [5], 8)
        row = {"token_groups": groups, "token_ids_sha256": fingerprint(ids[0].tolist())}
        self.assertEqual(backend.groups(None, row, {"input_ids": ids}, {"image_positions": [1, 2, 3]}), groups)
        with self.assertRaisesRegex(ValueError, "checksum|sha256"):
            backend.groups(None, {**row, "token_ids_sha256": "wrong"}, {"input_ids": ids}, {"image_positions": [1, 2, 3]})

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

    def test_attention_statistics_from_real_transformers_decoder(self):
        with torch.inference_mode():
            output = self.model(**self.clean, output_attentions=True, use_cache=False, logits_to_keep=1)
        values = attention_statistics(output.attentions, [1, 2, 3, 4], [5])
        self.assertEqual(len(values), 4)
        for row in values:
            self.assertAlmostEqual(row["visual_mass"] / 4, row["visual_per_token"])
            self.assertAlmostEqual(row["visual_object_relative"] + row["text_object_relative"], 1.)

    def test_full_tracing_orchestration_on_real_tiny_decoder(self):
        def batch(image, question, object_name=None):
            array = np.asarray(image, dtype=np.float32).copy() / 255.
            return {**self.clean, "pixel_values": torch.from_numpy(array).permute(2, 0, 1)[None]}, {"image_positions": [1, 2, 3, 4]}
        groups = {"early_visual": [1], "object_visual": [2, 3], "late_visual": [4],
                  "early_text": [0], "object_text": [5], "late_text": [], "last": [6]}
        config = {"seed": 17, "noise_sigma": 0.2, "min_probability_gap": 0., "rr_gap_policy": "nonzero"}
        with patch.object(self.adapter, "batch", side_effect=batch), patch.object(self.adapter, "groups", return_value=groups):
            result = trace_sample(self.adapter, Image.new("RGB", (28, 28), "white"),
                                  {"sample_id": "fixture", "image_id": 1, "question": "fixture", "object": "fixture"}, config)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["patches"]), 4 * 3 * 7)
        final = next(p for p in result["patches"] if p["layer"] == 3 and p["component"] == "hidden" and p["group"] == "last")
        self.assertAlmostEqual(final["p_patched"], result["p_clean"], places=7)
        self.assertAlmostEqual(final["rr"], 1., places=5)

    def test_equal_weight_ablation_equals_unit_rr_injections(self):
        rr = {"attention": [0.9, 0.8, 0.7, 0.6], "mlp": [0.9, 0.8, 0.7, 0.6]}
        unweighted = IRI(self.layers, rr, k_source=2, k_target=4, rr_weighting=False)
        unit = IRI(self.layers, {c: [1.] * 4 for c in ("attention", "mlp")}, k_source=2, k_target=4)
        with unweighted.apply():
            actual = self.adapter.probability(self.clean)
        with unit.apply():
            expected = self.adapter.probability(self.clean)
        self.assertEqual(actual, expected)


class PaperControlsTests(unittest.TestCase):
    def test_attention_mass_per_token_and_relative_are_distinct(self):
        weights = torch.tensor([[[[0.1, 0.2, 0.3, 0.4]], [[0.4, 0.2, 0.1, 0.3]]]])
        values = attention_statistics((weights,), [0, 1], [2])
        self.assertAlmostEqual(values[0]["visual_mass"], 0.45, places=6)
        self.assertAlmostEqual(values[0]["text_mass"], 0.2, places=6)
        self.assertAlmostEqual(values[0]["visual_per_token"], 0.225, places=6)
        summary = summarize_attention([{"status": "complete", "layers": values}], 3000)
        self.assertFalse(summary["paper_sample_count_met"])
        self.assertFalse(summary["author_normalization_verified"])
        with self.assertRaises(ValueError):
            attention_statistics((weights,), [0], [0])

    def test_ablation_cases_do_not_inherit_other_cases_settings(self):
        config = json.loads((ROOT / "configs/fcct_llava15.json").read_text())
        plan = json.loads((ROOT / "configs/fcct_ablation_plan.json").read_text())
        poisoned = {**config, "attention_scale": 999, "iri_allowed_layers": [0], "normalize": False}
        cases = ablation_settings(poisoned, plan, ["without_mlp", "last_ten", "without_rr_weights", "iri"])
        self.assertEqual(cases[0][1]["attention_scale"], 0.22)
        self.assertEqual(cases[0][1]["mlp_scale"], 0.)
        self.assertEqual(cases[1][1]["iri_allowed_layers"], list(range(22, 32)))
        self.assertFalse(cases[2][1]["rr_weighting"])
        self.assertTrue(cases[3][1]["normalize"])
        self.assertIsNone(cases[3][1]["iri_allowed_layers"])
        self.assertEqual(config["attention_scale"], 0.26)
        self.assertEqual(len([case for case in plan["settings"] if case["supported"]]), 12)
        with self.assertRaisesRegex(ValueError, "not implemented"):
            ablation_settings(config, plan, ["hidden_states"])


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
        self.assertAlmostEqual(p["aggregation_sensitivity"]["attention"]["last"]["question_mean"][0], 2 / 3)
        self.assertEqual(p["image_support"]["attention"]["last"][0], 2)
        self.assertEqual(p["question_support"]["attention"]["last"][0], 3)

    def test_old_or_incompatible_profile_is_rejected(self):
        from fcct import IMPLEMENTATION_VERSION
        config = json.loads((ROOT / "configs/fcct_llava15.json").read_text())
        profile = {"model_id": config["model_id"], "revision": config["revision"],
                   "implementation_version": IMPLEMENTATION_VERSION, "complete": True,
                   "trace_config": config, "rr": {"attention": [1.] * 32, "mlp": [1.] * 32}}
        path = self.path / "profile.json"
        write_json(path, profile)
        backend = SimpleNamespace(layers=[None] * 32)
        load_iri(backend, path, config)
        with self.assertRaisesRegex(ValueError, "different tracing settings"):
            load_iri(backend, path, {**config, "noise_sigma": 0.9})
        write_json(path, {**profile, "implementation_version": "0.1.0"})
        with self.assertRaisesRegex(ValueError, "predates"):
            load_iri(backend, path, config)

    def test_attention_runner_exports_markdown_and_raw_statistics(self):
        from fcct.attention import run_attention
        source = self.path / "image.png"
        Image.new("RGB", (8, 8), "white").save(source)
        manifest = self.path / "attention_manifest.jsonl"
        write_jsonl(manifest, [{"sample_id": "fixture", "image_id": 1, "question": "Is there a cat?",
                   "object": "cat", "boxes": [[0, 0, 8, 8]], "image": "image.png", "image_sha256": sha256(source)}])
        weights = torch.tensor([[[[0.1, 0.2, 0.3, 0.4]]]])
        backend = SimpleNamespace(attention_statistics=lambda image, row: {
            "layers": attention_statistics((weights,), [0, 1], [2]), "token_coverage": {}})
        result = run_attention(backend, {"data_root": str(self.path)}, ROOT, manifest,
                               self.path / "attention", allow_partial=True)
        self.assertFalse(result["exact_reproduction"])
        self.assertEqual(result["valid_questions"], 1)
        self.assertTrue((self.path / "attention/attention_report.md").is_file())
        self.assertTrue((self.path / "attention/object_attention.png").is_file())
        summary = json.loads((self.path / "attention/attention_summary.json").read_text())
        self.assertFalse(summary["paper_sample_count_met"])
        with self.assertRaisesRegex(ValueError, "3000"):
            run_attention(backend, {"data_root": str(self.path)}, ROOT, manifest, self.path / "other")

    def test_exact_mode_fails_before_inference_with_an_audit_report(self):
        import contextlib
        import importlib.util
        import io
        spec = importlib.util.spec_from_file_location("fcct_cli_test", ROOT / "fcct.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["--require-exact", "audit", "--output-dir", str(self.path)])
        self.assertEqual(raised.exception.code, 2)
        report = json.loads((self.path / "fcct_architecture_audit.json").read_text())
        self.assertFalse(report["exact_reproduction_ready"])
        self.assertTrue((self.path / "fcct_architecture_audit.md").is_file())

    def test_partial_ablation_report_keeps_measurements_separate_from_targets(self):
        from fcct.ablations import run_ablations
        from fcct.io import read_jsonl
        config = json.loads((ROOT / "configs/fcct_llava15.json").read_text())
        config.update(data_root=str(self.path / "data"), output_root=str(self.path / "output"))
        rows = [{"sample_id": f"fixture-{i}", "image_id": 1, "split": "adversarial",
                 "label": "yes" if i % 2 else "no"} for i in range(3000)]
        write_jsonl(Path(config["data_root"]) / "pope.jsonl", rows)
        write_json(Path(config["output_root"]) / "profile.json", {"complete": False})
        def generate(backend, cfg, repo, manifest, output, profile, **kwargs):
            samples = read_jsonl(manifest)[:kwargs["limit"]]
            write_jsonl(output, [{**r, "answers": {"regular": "No", "iri": "No"}} for r in samples])
        with patch("fcct.ablations.run_evaluation", side_effect=generate):
            result = run_ablations(None, config, ROOT, ROOT / "configs/fcct_ablation_plan.json",
                                   ["without_mlp"], limit=3, allow_partial=True)
        self.assertFalse(result["all_supported_controls_completed"])
        summary = json.loads((Path(config["output_root"]) / "ablations/ablation_summary.json").read_text())
        measured = next(r for r in summary["settings"] if r["id"] == "without_mlp")
        self.assertEqual(measured["measured"]["f1"], 0.)
        self.assertEqual(measured["paper"]["f1"], 84.02)
        self.assertFalse(measured["all_questions_processed"])
        hidden = next(r for r in summary["settings"] if r["id"] == "hidden_states")
        self.assertEqual(hidden["status"], "not_implemented")

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
        images = root / "images"
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
