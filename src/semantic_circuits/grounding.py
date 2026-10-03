"""Optional Hugging Face Grounding DINO + SAM2 proposal adapter."""
from __future__ import annotations

import torch
from .runtime import configure_local_runtime, resolve_cached_revision, resolve_device


class GroundedEvidenceProposer:
    """Ground phrase candidates and segment the highest-scoring box.

    Checkpoint IDs are explicit so methods/checkpoints can be recorded in experiment metadata.
    """
    def __init__(self, grounding_model_id="IDEA-Research/grounding-dino-tiny",
                 sam2_model_id="facebook/sam2.1-hiera-tiny", device="auto",
                 box_threshold=0.25, text_threshold=0.20,
                 grounding_revision=None, sam2_revision=None):
        from pathlib import Path
        configure_local_runtime(Path(__file__).resolve().parents[2])
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Processor, Sam2Model
        self.device = resolve_device(device)
        self.grounding_model_id, self.sam2_model_id = grounding_model_id, sam2_model_id
        self.grounding_revision = grounding_revision
        self.sam2_revision = sam2_revision
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.ground_processor = AutoProcessor.from_pretrained(grounding_model_id, revision=grounding_revision)
        self.ground_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id, revision=grounding_revision).to(self.device).eval()
        self.sam_processor = Sam2Processor.from_pretrained(sam2_model_id, revision=sam2_revision)
        self.sam_model = Sam2Model.from_pretrained(sam2_model_id, revision=sam2_revision).to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, image, concepts: list[str]) -> dict:
        """Return candidates containing pixel xyxy boxes and SAM2 masks for each phrase."""
        image = image.convert("RGB")
        labels = [[str(c).strip().rstrip(".") for c in concepts if str(c).strip()]]
        if not labels[0]:
            return {"candidates": [], "models": {"grounding_dino": self.grounding_model_id, "sam2": self.sam2_model_id},
                    "revisions": {"grounding_dino": resolve_cached_revision(self.grounding_model_id, self.grounding_revision),
                                  "sam2": resolve_cached_revision(self.sam2_model_id, self.sam2_revision)}}
        batch = self.ground_processor(images=image, text=labels, return_tensors="pt").to(self.device)
        output = self.ground_model(**batch)
        detected = self.ground_processor.post_process_grounded_object_detection(
            output, batch.input_ids, threshold=self.box_threshold,
            text_threshold=self.text_threshold, target_sizes=[image.size[::-1]])[0]
        candidates = []
        for box, score, label in zip(detected["boxes"], detected["scores"], detected["text_labels"]):
            box_list = [float(x) for x in box.tolist()]
            sam_inputs = self.sam_processor(images=image, input_boxes=[[box_list]], return_tensors="pt").to(self.device)
            sam_output = self.sam_model(**sam_inputs, multimask_output=True)
            masks = self.sam_processor.post_process_masks(sam_output.pred_masks.cpu(), sam_inputs["original_sizes"])[0]
            quality = sam_output.iou_scores.reshape(-1)
            choice = int(quality.argmax().item())
            mask = masks[0, choice].numpy().astype("uint8")
            candidates.append({"concept": str(label), "box_xyxy_pixels": box_list,
                "grounding_score": float(score.item()), "mask": mask,
                "mask_quality": float(quality[choice].item())})
        return {"candidates": candidates,
                "models": {"grounding_dino": self.grounding_model_id, "sam2": self.sam2_model_id},
                "revisions": {"grounding_dino": resolve_cached_revision(self.grounding_model_id, self.grounding_revision),
                              "sam2": resolve_cached_revision(self.sam2_model_id, self.sam2_revision)},
                "thresholds": {"box": self.box_threshold, "text": self.text_threshold}}
