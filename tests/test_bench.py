"""Tests for the paper evaluation matrix (src/bench)."""
from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
import torch
from PIL import Image

from fcct.io import read_jsonl, sha256, write_json, write_jsonl
from fcct.report import binary_metrics
from bench import decoding, metrics
from bench.decoding import Method, add_diffusion_noise, hf_generate, vcd_generate
from test_rfcct import TinyModel

CHAIR_PY = Path(os.environ.get("BENCH_CHAIR_PY", ROOT / "data/bench/sources/chair/chair.py"))
MMHAL_PY = Path(os.environ.get("BENCH_MMHAL_PY", ROOT / "data/bench/sources/mmhal/eval_gpt4.py"))
NLTK_DIR = Path(os.environ.get("BENCH_NLTK_DATA", ROOT / ".venv/nltk_data"))


class MetricTests(unittest.TestCase):
    def test_pope_metrics_equal_fcct_binary_metrics(self):
        rows = [{"label": l, "answers": {"m": a}} for l, a in
                (("yes", "Yes."), ("yes", "No, there is not."), ("no", "Yes, there is"), ("no", "No"), ("no", "There is no cat."))]
        expected = binary_metrics(rows, "m")
        actual = metrics.pope_metrics([r["label"] for r in rows], [r["answers"]["m"] for r in rows])
        for k in ("tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1"):
            self.assertAlmostEqual(actual[k], expected[k])
        self.assertAlmostEqual(actual["yes_ratio"], 40.0)

    def test_mme_parse_and_pairwise_score(self):
        self.assertEqual(metrics.mme_parse("Yes."), "yes")
        self.assertEqual(metrics.mme_parse(" no"), "no")
        self.assertEqual(metrics.mme_parse("Yes, the image shows"), "yes")
        self.assertEqual(metrics.mme_parse("The answer is no"), "other")
        rows, texts = [], []
        # existence: image A both right, image B one right -> acc 75, acc+ 50 -> 125
        for qid, labels, answers in (("a.png", ("yes", "no"), ("Yes", "No")), ("b.png", ("yes", "no"), ("Yes", "Yes"))):
            for label, answer in zip(labels, answers):
                rows.append({"category": "existence", "question_id": qid, "label": label})
                texts.append(answer)
        rows += [{"category": "code_reasoning", "question_id": "c.png", "label": l} for l in ("yes", "no")]
        texts += ["Yes", "No"]
        scores = metrics.mme_scores(rows, texts)
        self.assertAlmostEqual(scores["subtasks"]["existence"], 125.0)
        self.assertAlmostEqual(scores["hallucination_total"], 125.0)
        self.assertAlmostEqual(scores["cognition_total"], 200.0)
        with self.assertRaises(ValueError):
            metrics.mme_scores(rows[:3], texts[:3])

    def test_mmhal_rating_and_metrics_follow_eval_gpt4(self):
        self.assertEqual(metrics.mmhal_rating("Explanation... Rating: 5"), 5)
        self.assertEqual(metrics.mmhal_rating("rating: 2 or rating: 4"), 0)   # multiple -> 0
        self.assertEqual(metrics.mmhal_rating("no score"), 0)
        ratings = [6, 2] * 48
        m = metrics.mmhal_metrics(ratings)
        self.assertAlmostEqual(m["score"], 4.0)
        self.assertAlmostEqual(m["hallucination_rate"], 50.0)
        self.assertEqual(m["per_type_score"], [6, 2, 6, 2, 6, 2, 6, 2])

    @unittest.skipUnless(MMHAL_PY.is_file(), "pinned eval_gpt4.py not available")
    def test_mmhal_template_is_extracted_verbatim(self):
        template = metrics.mmhal_template(MMHAL_PY)
        self.assertTrue(template.startswith("Please act as an impartial and objective judge"))
        self.assertEqual(template.count("{}"), 4)
        prompt = metrics.mmhal_prompt(template, {"image_content": ["Cat", "Dog"], "question": "Q?", "gt_answer": "G."}, "A.")
        self.assertIn("Cat, Dog", prompt)
        self.assertTrue(prompt.rstrip().endswith("A."))

    @unittest.skipUnless(CHAIR_PY.is_file() and NLTK_DIR.is_dir(), "pinned chair.py / NLTK data not available")
    def test_chair_with_pinned_opera_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cats = [{"id": 1, "name": "person"}, {"id": 18, "name": "dog"}, {"id": 17, "name": "cat"}]
            for split in ("train", "val"):
                write_json(tmp / f"instances_{split}2014.json", {
                    "info": {}, "licenses": [], "categories": cats, "images": [{"id": 1 if split == "val" else 9}],
                    "annotations": [{"image_id": 1, "category_id": 18}] if split == "val" else []})
                write_json(tmp / f"captions_{split}2014.json", {
                    "info": {}, "licenses": [], "images": [],
                    "annotations": [{"image_id": 1, "caption": "A man walking a dog."}] if split == "val" else []})
            captions = tmp / "caps.jsonl"
            write_jsonl(captions, [{"image_id": 1, "caption": "A man and his dog sit next to a cat."},
                                   {"image_id": 1, "caption": "A dog."}])
            metrics.nltk_ready(NLTK_DIR)
            m = metrics.chair_metrics(CHAIR_PY, tmp, tmp / "cache.pkl", captions)
            # gt {dog, person}; caption 1 mentions person, dog, cat (cat hallucinated); caption 2 dog.
            self.assertAlmostEqual(m["CHAIRs"], 50.0)
            self.assertAlmostEqual(m["CHAIRi"], 100 / 4)
            self.assertAlmostEqual(m["Recall"], 100 * 3 / 4)
            self.assertEqual(m["captions"], 2)

    def test_pinned_file_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "chair.py"
            fake.write_text("print('not the official file')\n")
            with self.assertRaisesRegex(ValueError, "pinned"):
                metrics.chair_metrics(fake, Path(tmp), Path(tmp) / "c.pkl", Path(tmp) / "x.jsonl")


class DecodingTests(TinyModel):
    def test_diffusion_noise_matches_closed_form(self):
        x = torch.randn(1, 3, 4, 4)
        g = torch.Generator().manual_seed(3)
        out = add_diffusion_noise(x, 500, g)
        betas = torch.sigmoid(torch.linspace(-6, 6, 1000)) * (0.5e-2 - 1e-5) + 1e-5
        abar = torch.cumprod(1 - betas, 0)[500]
        eps = torch.randn(x.shape, generator=torch.Generator().manual_seed(3))
        torch.testing.assert_close(out, abar.sqrt() * x + (1 - abar).sqrt() * eps)
        with self.assertRaises(ValueError):
            add_diffusion_noise(x, 1000)

    def test_vcd_with_alpha_zero_equals_greedy_generate(self):
        greedy = {"do_sample": False}
        hf, _ = hf_generate(self.backend, self.clean, protocol=greedy, max_new_tokens=6)
        vcd, _ = vcd_generate(self.backend, self.clean, protocol=greedy, max_new_tokens=6, alpha=0.0, beta=0.1,
                              noise_step=999, noise_seed=1)
        self.assertEqual(hf.tolist(), vcd.tolist())
        contrast, _ = vcd_generate(self.backend, self.clean, protocol=greedy, max_new_tokens=6, alpha=1.0,
                                   beta=0.1, noise_step=999, noise_seed=1)
        self.assertEqual(len(contrast), len(contrast.tolist()))

    def test_plausibility_mask_and_topk_warper(self):
        logits = torch.tensor([[5.0, 4.0, 0.0, -10.0]])
        cutoff = math.log(0.1) + 5.0
        self.assertTrue(bool((logits < cutoff)[0, 2]))
        warped = decoding._warp(logits, {"temperature": 1.0, "top_k": 2, "top_p": 1.0})
        self.assertTrue(torch.isinf(warped[0, 2]) and torch.isfinite(warped[0, :2]).all())

    def test_sampling_is_reproducible_per_seed_and_key(self):
        protocol = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 50}
        method = Method("regular")
        image = Image.new("RGB", (28, 28), "white")
        a = method.generate(self.backend, image, "q", protocol=protocol, max_new_tokens=5, seed_key="k", seed=1)
        b = method.generate(self.backend, image, "q", protocol=protocol, max_new_tokens=5, seed_key="k", seed=1)
        self.assertEqual(a, b)
        vcd = Method("vcd", vcd={"alpha": 1.0, "beta": 0.1, "noise_step": {"default": 500}})
        c = vcd.generate(self.backend, image, "q", protocol=protocol, max_new_tokens=5, seed_key="k", seed=1)
        d = vcd.generate(self.backend, image, "q", protocol=protocol, max_new_tokens=5, seed_key="k", seed=1)
        self.assertEqual(c, d)


class SmokeScriptTests(TinyModel):
    def test_real_model_smoke_checks_run_on_tiny_model(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from smoke_pipeline import run_checks
        rr = {"attention": [0.3, 0.9, 0.5, 0.7], "mlp": [0.8, 0.2, 0.6, 0.4]}
        checks = run_checks(self.backend, (6, 7), Image.new("RGB", (28, 28), "white"),
                            {"sample_id": "s", "image_id": 1, "question": "Is there a cat?"}, rr, tokens=4)
        # The tiny fixture's decoder is fake, so only the parser/margin check is exempt.
        failing = [k for k, v in checks.items() if not v["pass"] and k != "pope_parser_agrees_with_first_token_margin"]
        self.assertEqual(failing, [])
        self.assertEqual(len(checks), 10)


class MatrixPipelineTests(TinyModel):
    """Tiny-model run of generate -> ablations -> latency -> judge -> report."""

    NUM_LAYERS = 32  # the paper's IRI/ablation settings index 32 decoder layers

    def test_full_matrix(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from bench import report, runner
        from bench.data import prepare
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            repo = Path(tmp)
            for rel in ("src/fcct", "src/rfcct", "src/bench"):
                (repo / rel).mkdir(parents=True)
                for f in (ROOT / rel).glob("*.py"):
                    (repo / rel / f.name).write_bytes(f.read_bytes())
            (repo / "configs").mkdir()
            for name in ("paper_tables_llava15.json", "fcct_ablation_plan.json"):
                (repo / "configs" / name).write_bytes((ROOT / "configs" / name).read_bytes())
            fcct_data = repo / "data/fcct"
            (fcct_data / "images").mkdir(parents=True)
            rng = np.random.default_rng(0)
            pope = []
            for i in range(4):
                name = f"img{i}.jpg"
                Image.fromarray(rng.integers(0, 255, (28, 28, 3), dtype=np.uint8)).save(fcct_data / "images" / name)
                digest = sha256(fcct_data / "images" / name)
                for split in ("random", "popular", "adversarial"):
                    for obj, label in (("cat", "yes"), (f"dog{split}", "no")):
                        pope.append({"sample_id": f"{split}-{i}-{obj}", "image_id": i, "image": f"images/{name}",
                                     "image_sha256": digest, "question": f"Is there a {obj} in the image?",
                                     "label": label, "split": split})
            write_jsonl(fcct_data / "pope.jsonl", pope)
            write_json(repo / "outputs/fcct/profile.json",
                       {"rr": {c: rng.uniform(-0.2, 1.0, 32).tolist() for c in ("attention", "mlp")}})
            # MME: synthetic parquet with the real schema (two questions per image).
            mme_dir = repo / "data/bench/sources/mme"
            mme_dir.mkdir(parents=True)
            records = []
            for cat in ("existence", "count", "code_reasoning"):
                buf = io.BytesIO()
                Image.fromarray(rng.integers(0, 255, (28, 28, 3), dtype=np.uint8)).save(buf, format="PNG")
                for ans in ("Yes", "No"):
                    records.append({"question_id": f"{cat}/1.png", "image": {"bytes": buf.getvalue(), "path": "1.png"},
                                    "question": f"Is it {ans}? Please answer yes or no.", "answer": ans, "category": cat})
            pq.write_table(pa.Table.from_pylist(records[:2]), mme_dir / "p0.parquet")
            pq.write_table(pa.Table.from_pylist(records[2:4]), mme_dir / "p1.parquet")
            pq.write_table(pa.Table.from_pylist(records[4:]), mme_dir / "p2.parquet")
            pq.write_table(pa.Table.from_pylist(records[:0], schema=pa.Table.from_pylist(records).schema), mme_dir / "p3.parquet")
            cfg = json.loads((ROOT / "configs/bench_llava15.json").read_text())
            cfg["pope"].update(expected_rows=len(pope), max_new_tokens=3, methods=["regular", "vcd", "iri"],
                               runs={"greedy": [0], "vcd_sampling": [0, 1]})
            cfg["mme"].update(max_new_tokens=3, methods=["regular", "iri"], runs={"greedy": [0]})
            cfg["latency"].update(prompts=2, new_tokens=3, warmup=1, methods=["regular", "vcd", "iri"])
            cfg["ablations"]["runs"] = {"greedy": [0]}
            for b in ("chair", "mmhal"):
                cfg[b]["methods"] = []
            base = {"data_root": "data/fcct", "output_root": "outputs/fcct", "model_id": "tiny", "revision": "a" * 40,
                    "dtype": "float32", "normalize": True}
            prepare(cfg, base, repo, ["pope", "mme"])
            self.backend.device = torch.device("cpu")
            expected = {"pope": 16, "mme": 6}  # POPE: 4 images x (1 shared positive + 3 split negatives)
            for bench in ("pope", "mme"):
                for method, protocol, seed in runner.cells(cfg, bench):
                    out = runner.run_cell(self.backend, cfg, base, {}, repo, bench, method, protocol, seed,
                                          yes_no_ids=(6, 7))
                    self.assertEqual((out["completed"], out["expected"]), (expected[bench], expected[bench]))
            for method, protocol, seed in runner.ablation_cells(cfg, repo):
                runner.run_cell(self.backend, cfg, base, {}, repo, "pope_adversarial", method, protocol, seed,
                                yes_no_ids=(6, 7), limit=2)
                runner.run_cell(self.backend, cfg, base, {}, repo, "pope_adversarial", method, protocol, seed,
                                yes_no_ids=(6, 7))
            latency = runner.run_latency(self.backend, cfg, base, {}, repo)
            self.assertEqual(set(latency["methods"]), {"regular", "vcd", "iri"})
            report.build(cfg, repo)
            result = json.loads((repo / "outputs/bench/report/results.json").read_text())
            self.assertEqual(result["pope"]["regular"]["vcd_sampling"]["random_accuracy"]["n"], 2)
            self.assertIn("hallucination_total", result["mme"]["iri"]["greedy"])
            self.assertEqual(len(result["ablations"]), 12)
            text = (repo / "outputs/bench/report/report.md").read_text()
            for title in ("Table 1", "Table 2", "Table 3", "Table 4", "Table 6", "Table 7"):
                self.assertIn(title, text)
            # A changed protocol must not resume into the old file.
            cfg["protocols"]["vcd_sampling"]["top_k"] = 10
            with self.assertRaisesRegex(ValueError, "Resume rejected"):
                runner.run_cell(self.backend, cfg, base, {}, repo, "pope", "regular", "vcd_sampling", 0)

    def test_mmhal_judge_with_fake_client(self):
        from bench import runner
        if not MMHAL_PY.is_file():
            self.skipTest("pinned eval_gpt4.py not available")
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            repo = Path(tmp)
            cfg = json.loads((ROOT / "configs/bench_llava15.json").read_text())
            src = repo / cfg["data_root"] / "sources/mmhal"
            src.mkdir(parents=True)
            (src / "eval_gpt4.py").write_bytes(MMHAL_PY.read_bytes())
            items = [{"image_content": ["Cat"], "question": f"q{i}", "gt_answer": "g"} for i in range(96)]
            (src / "response_template.json").write_text(json.dumps(items))
            gen = runner.cell_path(cfg, repo, "mmhal", "regular", "greedy", 0)
            write_jsonl(gen, [{"key": f"mmhal-{i:02d}", "text": f"answer {i}"} for i in range(96)])
            calls = []

            def create(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok. Rating: 4"))])
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            out = runner.run_judge(cfg, repo, "regular", "greedy", 0, client=client)
            rows = read_jsonl(Path(out["output"]))
            self.assertEqual([r["rating"] for r in rows], [4] * 96)
            self.assertEqual(calls[0]["model"], "gpt-4-0314")
            self.assertEqual(calls[0]["temperature"], 0.0)
            self.assertIn("answer 0", calls[0]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
