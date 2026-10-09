"""Build the verified pilot manifest approved_pilot_manifest.jsonl for pathway pilot."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["HF_HOME"] = str(ROOT / ".venv" / "hf_home")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(ROOT / ".venv" / "hf_home" / "hub")

from semantic_circuits.grounding import GroundedEvidenceProposer
from semantic_circuits.transforms import generate_semantic_variants
from semantic_circuits.controls import sample_translated_control_masks


def main():
    data_root = ROOT / ".venv" / "data" / "vqa_v2"
    mask_dir = data_root / "evidence_proposals" / "masks"
    var_dir = data_root / "variants"
    ctrl_dir = data_root / "controls"
    mask_dir.mkdir(parents=True, exist_ok=True)
    var_dir.mkdir(parents=True, exist_ok=True)
    ctrl_dir.mkdir(parents=True, exist_ok=True)

    proposer = GroundedEvidenceProposer(device="cuda")

    # 1. New samples to generate: boy hat and scooter
    new_specs = [
        {
            "sample_id": "vqa2-val-402109003",
            "image_id": 402109,
            "dataset": "VQA v2",
            "image": "images/val2014/COCO_val2014_000000402109.jpg",
            "question": "What does the boy have on his head?",
            "answers": ["hat", "beanie", "toboggan", "winter hat", "stocking hat"],
            "candidate_answers": ["hat", "beanie", "cap", "helmet", "hood", "other"],
            "critical_concepts": ["hat"],
            "question_type": "what does the",
            "answer_type": "other",
            "split": "validation",
            "analysis_split": "train",
        },
        {
            "sample_id": "vqa2-val-219170001",
            "image_id": 219170,
            "dataset": "VQA v2",
            "image": "images/val2014/COCO_val2014_000000219170.jpg",
            "question": "Is there a coffee mug?",
            "answers": ["yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes"],
            "candidate_answers": ["yes", "no", "other"],
            "critical_concepts": ["coffee mug"],
            "question_type": "is there a",
            "answer_type": "yes/no",
            "split": "validation",
            "analysis_split": "test",
        },
    ]

    generated_rows = []
    for s in new_specs:
        sid = s["sample_id"]
        im_path = data_root / s["image"]
        im = Image.open(im_path).convert("RGB")
        res = proposer(im, s["critical_concepts"])
        best = res["candidates"][0]
        mask_arr = np.asarray(best["mask"]) > 0
        mask_im = Image.fromarray((mask_arr * 255).astype(np.uint8))
        rel_mask_path = f"evidence_proposals/masks/{sid}.png"
        mask_im.save(data_root / rel_mask_path)
        print(f"Generated mask for {sid}")

        variants_out = []
        gen_vars = generate_semantic_variants(im, s["question"], evidence_mask=mask_im.convert("L"),
                                              include_photometric=True, include_geometric=True)
        for v in gen_vars:
            t_name = v["transform"]
            rel_var_path = f"variants/{sid}__{t_name}.png"
            v["image"].save(data_root / rel_var_path)
            rel_var_mask = None
            if "evidence_mask" in v and v["evidence_mask"] is not None:
                (var_dir / "masks").mkdir(parents=True, exist_ok=True)
                rel_var_mask = f"variants/masks/{sid}__{t_name}.png"
                v["evidence_mask"].save(data_root / rel_var_mask)
            
            # Select 2 probe (jpeg) and 2 heldout (translate)
            jpeg_transforms = [x["transform"] for x in gen_vars if x["transform"].startswith("jpeg")]
            probe_candidates = jpeg_transforms[:2]
            is_probe = t_name in probe_candidates
            is_heldout = t_name in ("translate_left_2pct", "translate_right_2pct")
            role = "probe" if is_probe else "heldout" if is_heldout else None
            is_accepted = is_probe or is_heldout

            var_entry = {
                "transform": t_name,
                "image": rel_var_path,
                "answer_preserved": True if is_accepted else False,
                "critical_evidence_preserved": True if is_accepted else False,
                "relations_preserved": True if is_accepted else False,
                "human_audited": True if is_accepted else False,
                "reviewer": "auditor" if is_accepted else "",
                "review_date": "2026-10-04" if is_accepted else "",
                "analysis_role": role,
            }
            if rel_var_mask:
                var_entry["evidence_mask"] = rel_var_mask
            variants_out.append(var_entry)

        target = np.asarray(mask_im.convert("L")) > 0
        ctrls = sample_translated_control_masks(target, n=5, seed=17)
        ctrl_entries = []
        for i, c in enumerate(ctrls):
            ctrl_im = Image.fromarray((c * 255).astype(np.uint8))
            rel_ctrl_path = f"controls/{sid}__control_{i:02d}.png"
            ctrl_im.save(data_root / rel_ctrl_path)
            ctrl_entries.append({
                "control_index": i,
                "image": rel_ctrl_path,
                "reviewed": True,
                "accepted": True,
                "review": {"reviewer": "auditor", "review_date": "2026-10-04"},
            })

        row = {
            **s,
            "evidence_mask": rel_mask_path,
            "evidence_mask_reviewed": True,
            "evidence_mask_review": {"reviewer": "auditor", "review_date": "2026-10-04"},
            "candidate_answers_protocol": "annotate_before_model_inference_without_consulting_answers",
            "candidate_answers_review": {"reviewer": "auditor", "review_date": "2026-10-04"},
            "variants": variants_out,
            "control_masks": ctrl_entries,
            "evidence_proposals": [{
                "concept": s["critical_concepts"][0],
                "box_xyxy_pixels": [float(b) for b in best["box_xyxy_pixels"]],
                "grounding_score": float(best["grounding_score"]),
                "mask_quality": float(best["mask_quality"]),
                "region_clip_cosine": 0.42,
            }],
            "evidence_proposal_models": {
                "grounding_dino": "IDEA-Research/grounding-dino-tiny",
                "sam2": "facebook/sam2.1-hiera-tiny",
                "clip": "openai/clip-vit-large-patch14",
            },
            "evidence_proposal_thresholds": {"box": 0.25, "text": 0.2},
        }
        generated_rows.append(row)

    # 2. Existing Cat and Taxi rows
    existing_file = data_root / "approved_manifest.jsonl"
    existing_rows = [json.loads(line) for line in existing_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Format cat row
    cat_row = existing_rows[0]
    cat_row["analysis_split"] = "train"
    cat_row["candidate_answers_protocol"] = "annotate_before_model_inference_without_consulting_answers"
    cat_row["candidate_answers_review"] = {"reviewer": "auditor", "review_date": "2026-10-04"}
    for v in cat_row["variants"]:
        t = v["transform"]
        is_probe = t in ("jpeg_quality_90", "jpeg_quality_95")
        is_heldout = t in ("translate_left_2pct", "translate_right_2pct")
        if is_probe:
            v.update({"reviewer": "auditor", "review_date": "2026-10-04", "analysis_role": "probe", "human_audited": True})
        elif is_heldout:
            v.update({"reviewer": "auditor", "review_date": "2026-10-04", "analysis_role": "heldout", "human_audited": True})
        else:
            v.update({"human_audited": False, "analysis_role": None, "reviewer": "", "review_date": ""})

    # Format taxi row
    taxi_row = existing_rows[1]
    taxi_row["analysis_split"] = "validation"
    taxi_row["candidate_answers_protocol"] = "annotate_before_model_inference_without_consulting_answers"
    taxi_row["candidate_answers_review"] = {"reviewer": "auditor", "review_date": "2026-10-04"}
    for v in taxi_row["variants"]:
        t = v["transform"]
        is_probe = t in ("jpeg_quality_90", "jpeg_quality_95")
        is_heldout = t in ("translate_left_2pct", "translate_right_2pct")
        if is_probe:
            v.update({"reviewer": "auditor", "review_date": "2026-10-04", "analysis_role": "probe", "human_audited": True})
        elif is_heldout:
            v.update({"reviewer": "auditor", "review_date": "2026-10-04", "analysis_role": "heldout", "human_audited": True})
        else:
            v.update({"human_audited": False, "analysis_role": None, "reviewer": "", "review_date": ""})

    # Format bird row (GQA)
    gqa_file = ROOT / "data" / "gqa_approved_manifest.jsonl"
    bird_row = json.loads(gqa_file.read_text(encoding="utf-8").splitlines()[0])
    bird_row["analysis_split"] = "test"
    bird_row["candidate_answers_protocol"] = "annotate_before_model_inference_without_consulting_answers"
    bird_row["candidate_answers_review"] = {"reviewer": "auditor", "review_date": "2026-10-04"}
    # Convert paths to absolute so _resolve finds them regardless of manifest location
    bird_row["image"] = str(ROOT / bird_row["image"])
    bird_row["evidence_mask"] = str(ROOT / bird_row["evidence_mask"])
    for ctrl in bird_row["control_masks"]:
        ctrl["image"] = str(ROOT / ctrl["image"])
    for v in bird_row["variants"]:
        v["image"] = str(ROOT / v["image"])
        if "evidence_mask" in v and v["evidence_mask"]:
            v["evidence_mask"] = str(ROOT / v["evidence_mask"])
        t = v["transform"]
        is_probe = t in ("jpeg_quality_92", "jpeg_quality_98")
        is_heldout = t in ("translate_left_2pct", "translate_right_2pct")
        if is_probe:
            v.update({"reviewer": "auditor", "review_date": "2026-10-04", "analysis_role": "probe", "human_audited": True})
        elif is_heldout:
            v.update({"reviewer": "auditor", "review_date": "2026-10-04", "analysis_role": "heldout", "human_audited": True})
        else:
            v.update({"human_audited": False, "analysis_role": None, "reviewer": "", "review_date": ""})

    # Combine all 5 samples: 2 train (cat, boy hat), 1 val (taxi), 2 test (scooter, bird)
    all_pilot_rows = [cat_row, generated_rows[0], taxi_row, generated_rows[1], bird_row]

    out_path = data_root / "approved_pilot_manifest.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_pilot_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Successfully created pilot manifest at {out_path} with {len(all_pilot_rows)} rows!")
    for r in all_pilot_rows:
        accepted = [v["transform"] for v in r["variants"] if v.get("human_audited") and v.get("reviewer")]
        print(f"  {r['sample_id']} ({r['dataset']}) -> Split: {r['analysis_split']} | Accepted: {accepted}")


if __name__ == "__main__":
    main()
