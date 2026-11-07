"""Regression coverage for the evaluation and cleanup integration boundaries."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts", ROOT / "src"):
    sys.path.insert(0, str(directory))

from PIL import Image
from pathway import COMMANDS
from scoring.dataset_scorers import score_prediction
from prepare_failure_benchmark import prepare
from run_ollama_vision_benchmark import evaluate_model_on_manifest
from run_pilot import check_ready
from benchmark_manifest import normalize_manifest_row


class ScoringTests(unittest.TestCase):
    def test_vqa_reference_consensus(self):
        refs = ["yes"] * 9 + ["no"]
        self.assertAlmostEqual(score_prediction("No", refs, "VQA v2")[0], 0.3)
        self.assertEqual(score_prediction("yes", refs, "VQA v2")[0], 1.0)
        self.assertAlmostEqual(score_prediction("cat", ["cat"] * 3 + ["dog"] * 7, "vqav2")[0], 0.9)

    def test_gqa_rejects_substring(self):
        self.assertEqual(score_prediction("not red", ["red"], "GQA")[0], 0)
        self.assertEqual(score_prediction("Red.", ["red"], "GQA")[0], 1)

    def test_missing_labels_remain_undefined(self):
        self.assertIsNone(score_prediction("yes", None, "GQA")[0])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new("RGB", (16, 16), "white").save(self.root / "image.png")
        Image.new("L", (16, 16), 255).save(self.root / "mask.png")
        review = {"reviewer": "test reviewer", "review_date": "2026-11-07"}
        self.row = {
            "sample_id": "a", "image_id": "a", "dataset": "VQA v2", "question": "Is it visible?",
            "image": "image.png", "answers": ["yes"] * 9 + ["no"], "candidate_answers": ["yes", "no", "other"],
            "analysis_split": "train", "split": "validation", "evidence_mask": "mask.png",
            "evidence_mask_reviewed": True, "evidence_mask_review": review,
            "candidate_answers_protocol": "annotate_before_model_inference_without_consulting_answers",
            "candidate_answers_review": review,
            "control_masks": [{"image": "mask.png", "control_index": n, "reviewed": True,
                               "accepted": True, "review": review} for n in range(3)],
            "variants": [{"image": "image.png", "transform": transform, "analysis_role": role,
                          "answer_preserved": True, "critical_evidence_preserved": True,
                          "relations_preserved": True, "human_audited": True, **review}
                         for transform, role in [("jpeg_quality_92", "probe"), ("jpeg_quality_98", "probe"),
                                                 ("translate_left_2pct", "heldout"), ("translate_right_2pct", "heldout")]],
        }
        self.manifest = self.root / "manifest.jsonl"
        self.manifest.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
        self.output = self.root / "predictions.jsonl"

    def evaluate(self, answers):
        with patch("run_ollama_vision_benchmark.query_ollama", side_effect=answers) as query:
            result = evaluate_model_on_manifest("test:model", [self.row], self.manifest, self.root, self.output)
            return result, query.call_count

    def test_probe_failure_does_not_become_heldout_failure(self):
        summary, calls = self.evaluate(["yes", "no", "yes", "yes", "yes"])
        rec = json.loads(self.output.read_text())
        self.assertEqual(calls, 5)
        self.assertEqual(rec["heldout_failure"], 0)
        self.assertEqual(rec["probe_behavior_consistency"], 0.5)
        self.assertEqual(len(rec["heldout_variant_predictions"]), 2)
        self.assertEqual(summary["eligible_baseline_correct"], 1)
        _, resumed_calls = self.evaluate([])
        self.assertEqual(resumed_calls, 0)

    def test_candidate_option_is_not_a_gold_reference(self):
        summary, _ = self.evaluate(["no"] * 5)
        rec = json.loads(self.output.read_text())
        self.assertAlmostEqual(rec["vqa_consensus_score"], 0.3)
        self.assertIsNone(rec["heldout_failure"])
        self.assertEqual(summary["eligible_baseline_correct"], 0)

    def test_reject_legacy_predictions(self):
        self.output.write_text(json.dumps({"sample_id": "a"}) + "\n")
        with self.assertRaisesRegex(ValueError, "different protocol"):
            self.evaluate([])

    def test_transport_error_does_not_create_failure_record(self):
        with self.assertRaises(RuntimeError):
            self.evaluate(["yes", RuntimeError("offline")])
        self.assertEqual(self.output.read_text(), "")

    def test_preflight_requires_review_and_explicit_test_release(self):
        config = {"manifest": str(self.manifest), "data_root": str(self.root), "output_dir": str(self.root / "run")}
        check_ready(config, None)
        self.row["analysis_split"] = "test"
        self.manifest.write_text(json.dumps(self.row) + "\n")
        with self.assertRaisesRegex(ValueError, "locked test"):
            check_ready(config, None)
        config["allow_test_split"] = True
        check_ready(config, None)
        self.row["evidence_mask_reviewed"] = False
        self.manifest.write_text(json.dumps(self.row) + "\n")
        with self.assertRaisesRegex(ValueError, "evidence_mask_reviewed"):
            check_ready(config, None)

    def test_failure_data_preserves_labels_and_splits(self):
        result = prepare([{**self.row, "heldout_failure": 1},
                          {**self.row, "sample_id": "b", "image_id": "b", "heldout_failure": None}])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["split"], "train")
        self.assertEqual(result[0]["future_failure"], 1)
        with self.assertRaisesRegex(ValueError, "leakage"):
            prepare([{**self.row, "heldout_failure": 1},
                     {**self.row, "sample_id": "b", "analysis_split": "test", "heldout_failure": 0}])

    def test_schema_v1_preserves_recorded_reviews(self):
        source = {**self.row, "schema_version": "1.0", "dataset_split": "validation",
                  "gold_answers": self.row["answers"], "candidate_answers_reviewer": "test reviewer",
                  "candidate_answers_review_date": "2026-11-07",
                  "evidence": {"mask_path": "mask.png", "accepted": True,
                               "reviewer_ids": ["test reviewer"], "review_date": "2026-11-07"},
                  "variants": [{**v, "transform_name": v["transform"], "role": v["analysis_role"],
                                "image_path": v["image"], "evidence_preserved": True,
                                "reviewer_id": v["reviewer"], "artifact_introduced": False,
                                "accepted": True} for v in self.row["variants"]],
                  "control_masks": [{"mask_path": "mask.png", "accepted": True,
                                     "reviewer_id": "test reviewer", "review_date": "2026-11-07"} for _ in range(3)]}
        self.manifest.write_text(json.dumps(source) + "\n")
        config = {"manifest": str(self.manifest), "data_root": str(self.root),
                  "output_dir": str(self.root / "run"), "require_reviewed_controls": True,
                  "min_reviewed_controls": 3}
        _, _, rows = check_ready(config, None)
        self.assertTrue(rows[0]["evidence_mask_reviewed"])
        self.assertEqual(rows[0]["candidate_answers_review"]["reviewer"], "test reviewer")
        self.assertEqual(rows[0]["control_masks"][0]["review"]["reviewer"], "test reviewer")
        source["evidence"]["reviewer_ids"] = []
        self.assertEqual(normalize_manifest_row(source)["evidence_mask_review"]["reviewer"], "")

    def test_train_phase_does_not_release_test_partition(self):
        heldout = {**self.row, "sample_id": "test-a", "image_id": "test-a", "analysis_split": "test"}
        self.manifest.write_text(json.dumps(self.row) + "\n" + json.dumps(heldout) + "\n")
        config = {"manifest": str(self.manifest), "data_root": str(self.root),
                  "output_dir": str(self.root / "run"), "analysis_splits": ["train", "validation"]}
        _, _, rows = check_ready(config, None)
        self.assertEqual([r["sample_id"] for r in rows], ["a"])
        config["analysis_splits"] = ["test"]
        with self.assertRaisesRegex(ValueError, "locked test"):
            check_ready(config, None)

    def test_all_commands_have_implementations(self):
        for script, _, _ in COMMANDS.values():
            self.assertTrue((ROOT / "scripts" / script).is_file(), script)


if __name__ == "__main__":
    unittest.main()
