"""Question-critical evidence schema and conservative deterministic parser."""
from __future__ import annotations

import re


def parse_question_requirements(question: str) -> dict:
    """Transparent baseline parser; replace with annotation or validated structured parser."""
    q = question.strip()
    low = q.lower()
    qtype = "object"
    if re.search(r"\b(color|colour)\b", low): qtype = "color"
    elif re.search(r"\bhow many|number of|count\b", low): qtype = "count"
    elif re.search(r"\b(left|right|above|below|behind|front|under|next to)\b", low): qtype = "spatial_relation"
    elif re.search(r"\b(holding|riding|wearing|standing|sitting|carrying)\b", low): qtype = "action_relation"
    elif re.search(r"\b(read|written|text|sign|says)\b", low): qtype = "ocr"
    elif re.search(r"\bwhat is|what are|which object|what object\b", low): qtype = "object"
    return {"question": q, "answer_type": qtype,
            "critical_objects": [], "critical_attributes": [qtype] if qtype in {"color", "count", "ocr"} else [],
            "critical_relations": [], "parser": "heuristic_v1", "requires_review": True}


def attach_evidence_proposals(requirements: dict, concepts: list[dict]) -> dict:
    """Join grounded evidence proposals to required slots without declaring them true."""
    return {**requirements, "evidence_proposals": concepts,
            "evidence_status": "unverified_proposals"}
