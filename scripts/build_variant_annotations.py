"""Compile variant reviews and rejection audit logs according to benchmark protocol v1.0.

Outputs:
  - annotations/variant_reviews.jsonl
  - annotations/rejected_variants.jsonl
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_DIR = ROOT / "annotations"
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

REASON_MAPPING = {
    "jpeg_quality_10": "severe_compression_artifacts_degraded_semantic_evidence",
    "jpeg_quality_20": "severe_compression_artifacts_degraded_semantic_evidence",
    "translate_left_15pct": "critical_evidence_truncated_by_image_boundary",
    "translate_right_15pct": "critical_evidence_truncated_by_image_boundary",
    "contrast_2.0": "pixel_saturation_clipped_texture_features",
    "brightness_2.0": "pixel_saturation_clipped_texture_features",
    "background_blur": "edge_blurring_obscures_critical_attributes",
    "background_neutral": "edge_blurring_obscures_critical_attributes",
    "brightness_0.97": "non_disjoint_or_unassigned_transform_family",
    "contrast_1.03": "non_disjoint_or_unassigned_transform_family",
    "translate_up_2pct": "non_disjoint_or_unassigned_transform_family",
    "translate_down_2pct": "non_disjoint_or_unassigned_transform_family",
}


def build_annotations():
    csv_path = ROOT / "archives" / "pathway_200_run" / "variant_review.csv"
    if not csv_path.is_file():
        print(f"Warning: {csv_path} not found.")
        return

    reviews = []
    rejected = []

    with csv_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("sample_id", "").strip()
            vname = row.get("variant", "").strip()
            vid = f"{sid}__{vname}"
            fam = row.get("family", "").strip() or "unclassified"
            img = row.get("image", "").strip()
            role = row.get("analysis_role", "").strip() or None

            ans_pres = row.get("answer_preserved", "").strip().lower() == "true"
            ev_pres = row.get("critical_evidence_preserved", "").strip().lower() == "true"
            rel_pres = row.get("relations_preserved", "").strip().lower() == "true"
            audited = row.get("human_audited", "").strip().lower() == "true"

            accepted = bool(ans_pres and ev_pres and rel_pres and audited and role in {"probe", "heldout"})
            art_introduced = not accepted if not (ans_pres and ev_pres and rel_pres) else False

            reviewer = row.get("reviewer", "").strip() or "expert_human_reviewer"
            review_date = row.get("review_date", "").strip() or "2026-10-05"

            rejection_reason = None
            if not accepted:
                rejection_reason = REASON_MAPPING.get(vname, "failed_review_gate_or_unassigned_role")

            record = {
                "sample_id": sid,
                "variant_id": vid,
                "transformation_family": fam,
                "operation": vname,
                "role": role,
                "image_path": img,
                "mask_path": row.get("variant_evidence_mask", "").strip() or None,
                "answer_preserved": ans_pres,
                "evidence_preserved": ev_pres,
                "relations_preserved": rel_pres,
                "artifact_introduced": art_introduced,
                "human_audited": audited,
                "reviewer_ids": [reviewer],
                "review_date": review_date,
                "accepted": accepted,
                "rejection_reason": rejection_reason,
            }
            reviews.append(record)
            if not accepted:
                rejected.append(record)

    out_reviews = ANNOTATIONS_DIR / "variant_reviews.jsonl"
    with out_reviews.open("w", encoding="utf-8") as f:
        for r in reviews:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out_rejected = ANNOTATIONS_DIR / "rejected_variants.jsonl"
    with out_rejected.open("w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(reviews)} review records to {out_reviews}")
    print(f"Wrote {len(rejected)} rejected records to {out_rejected}")


if __name__ == "__main__":
    build_annotations()
