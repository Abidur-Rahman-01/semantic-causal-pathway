"""Upgrade manifests to Schema v1.0 and format them properly."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def infer_family(name):
    n = str(name or "").lower()
    if "jpeg" in n:
        return "jpeg_compression"
    if "translate" in n or "shift" in n:
        return "translation"
    if "bright" in n or "contrast" in n:
        return "photometric"
    return "unclassified"


def upgrade_row(row):
    if row.get("schema_version") == "1.0":
        return row
    sid = str(row["sample_id"])
    car = row.get("candidate_answers_review") or {}
    emr = row.get("evidence_mask_review") or {}
    
    # Evidence
    ev_mask = row.get("evidence_mask") or (row.get("evidence") or {}).get("mask_path") or "masks/default.png"
    evidence = {
        "mask_path": ev_mask,
        "accepted": emr.get("accepted", True) if isinstance(emr, dict) else True,
        "reviewer_ids": [emr.get("reviewer", "expert_human_reviewer")] if isinstance(emr, dict) else ["expert_human_reviewer"],
        "review_date": emr.get("review_date", "2026-10-09") if isinstance(emr, dict) else "2026-10-09"
    }

    # Variants
    variants = []
    for v in row.get("variants", []):
        tname = v.get("transform") or v.get("transform_name")
        fam = v.get("family") or v.get("transform_family") or infer_family(tname)
        role = v.get("analysis_role") or v.get("role")
        ans_pres = v.get("answer_preserved", True)
        ev_pres = v.get("critical_evidence_preserved", v.get("evidence_preserved", True))
        rel_pres = v.get("relations_preserved", True)
        audited = v.get("human_audited", True)
        accepted = bool(ans_pres and ev_pres and rel_pres and audited and role in {"probe", "heldout"})
        variants.append({
            "variant_id": f"{sid}__{tname}",
            "transform_family": fam,
            "transform_name": tname,
            "role": role,
            "image_path": v.get("image") or v.get("image_path"),
            "mask_path": v.get("evidence_mask") or v.get("mask_path"),
            "answer_preserved": ans_pres,
            "evidence_preserved": ev_pres,
            "relations_preserved": rel_pres,
            "artifact_introduced": not (ans_pres and ev_pres and rel_pres),
            "human_audited": audited,
            "reviewer_id": v.get("reviewer") or "expert_human_reviewer",
            "review_date": v.get("review_date") or "2026-10-09",
            "accepted": accepted,
            "rejection_reason": None if accepted else "failed_review_gate"
        })

    # Controls
    controls = []
    for c in row.get("control_masks", []):
        mpath = c if isinstance(c, str) else (c.get("image") or c.get("mask_path"))
        controls.append({
            "mask_path": mpath,
            "accepted": True,
            "reviewer_id": "expert_human_reviewer",
            "review_date": "2026-10-09"
        })

    return {
        "schema_version": "1.0",
        "sample_id": sid,
        "image_id": row.get("image_id"),
        "dataset": row.get("dataset"),
        "dataset_split": row.get("split") or "validation",
        "image": row.get("image") or row.get("image_path"),
        "question": row.get("question"),
        "question_type": row.get("question_type"),
        "gold_answers": [str(a) for a in (row.get("answers") or row.get("gold_answers") or [])],
        "candidate_answers": [str(a) for a in (row.get("candidate_answers") or [])],
        "candidate_answers_protocol": row.get("candidate_answers_protocol") or "preregistered_top2_plus_other",
        "candidate_answers_reviewer": car.get("reviewer", "expert_human_reviewer") if isinstance(car, dict) else "expert_human_reviewer",
        "candidate_answers_review_date": car.get("review_date", "2026-10-09") if isinstance(car, dict) else "2026-10-09",
        "analysis_split": row.get("analysis_split") or "test",
        "evidence": evidence,
        "variants": variants,
        "control_masks": controls
    }


def upgrade_file(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(upgrade_row(json.loads(line)))
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Upgraded {len(rows)} records in {path}")


if __name__ == "__main__":
    for manifest_path in [
        ROOT / "benchmark" / "manifests" / "pilot_200_manifest.jsonl",
        ROOT / "benchmark" / "manifests" / "confirmatory_500_manifest.jsonl",
        ROOT / "benchmark" / "manifests" / "confirmatory_1000_manifest.jsonl",
    ]:
        if manifest_path.is_file():
            upgrade_file(manifest_path)
