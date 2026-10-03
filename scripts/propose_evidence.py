"""Generate Grounding DINO/SAM2 evidence-mask proposals for manually specified concepts."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".venv" / "hf_home")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(ROOT / ".venv" / "hf_home" / "hub")
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path(".venv/data/vqa_v2"))
    parser.add_argument("--output", type=Path, default=Path(".venv/data/vqa_v2/evidence_proposals"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--clip-rank", action="store_true",
                        help="Record whole-image and grounded-region CLIP cosine scores (diagnostic only)")
    args = parser.parse_args()
    from semantic_circuits.grounding import GroundedEvidenceProposer
    proposer = GroundedEvidenceProposer(device=args.device)
    clip_scorer = None
    if args.clip_rank:
        from semantic_circuits.clip_proposals import ClipSemanticScorer
        clip_scorer = ClipSemanticScorer(device=args.device)
    manifest = args.manifest.resolve()
    root, output = args.data_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    out_manifest = output / "manifest_with_evidence_proposals.jsonl"
    review_path = output / "evidence_review.csv"
    review_rows = []
    with manifest.open(encoding="utf-8-sig") as source, out_manifest.open("w", encoding="utf-8") as destination:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            if args.limit is not None and index >= args.limit:
                break
            row = json.loads(line)
            concepts = row.get("critical_concepts") or []
            if not concepts:
                raise ValueError(f"{row.get('sample_id')}: add reviewed critical_concepts before proposing masks")
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = manifest.parent / image_path if (manifest.parent / image_path).exists() else root / image_path
            image = Image.open(image_path).convert("RGB")
            proposal = proposer(image, concepts)
            if clip_scorer is not None:
                proposal["models"]["clip"] = clip_scorer.model_id
                proposal.setdefault("revisions", {})["clip"] = clip_scorer.resolved_revision
                for candidate in proposal["candidates"]:
                    concept = candidate["concept"]
                    box = [int(round(value)) for value in candidate["box_xyxy_pixels"]]
                    x0, y0, x1, y1 = box
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(image.width, x1), min(image.height, y1)
                    candidate["global_clip_cosine"] = clip_scorer(image, concept)
                    candidate["region_clip_cosine"] = (
                        clip_scorer(image.crop((x0, y0, x1, y1)), concept)
                        if x1 > x0 and y1 > y0 else None)
            masks = [item["mask"].astype(bool) for item in proposal["candidates"]]
            if masks:
                combined = np.logical_or.reduce(masks).astype("uint8") * 255
                mask_path = output / "masks" / f"{row['sample_id']}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(combined).save(mask_path)
                row["evidence_mask"] = str(mask_path.relative_to(root)) if mask_path.is_relative_to(root) else str(mask_path)
            else:
                row["evidence_mask"] = None
            row["evidence_mask_reviewed"] = False
            row["evidence_proposals"] = [{k: v for k, v in item.items() if k != "mask"} for item in proposal["candidates"]]
            row["evidence_proposal_models"] = proposal["models"]
            row["evidence_proposal_revisions"] = proposal.get("revisions", {})
            row["evidence_proposal_thresholds"] = proposal["thresholds"]
            review_rows.append({"sample_id": row["sample_id"], "mask": row["evidence_mask"] or "",
                                "critical_concepts": ";".join(concepts), "mask_accepted": "",
                                "reviewer": "", "review_date": ""})
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    with review_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "mask", "critical_concepts", "mask_accepted", "reviewer", "review_date"])
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"Wrote unreviewed mask proposals to {out_manifest} and review sheet {review_path}")


if __name__ == "__main__":
    main()
