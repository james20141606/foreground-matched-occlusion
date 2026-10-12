"""Common dataclasses shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Example:
    """A single benchmark probe."""
    example_id: str
    image_path: str | None
    question: str
    answer: str | None                 # ground-truth, None for unanswerable
    answer_type: str                   # yes_no | mcq | open
    source: str                        # slake | vqa_rad | omni | path
    modality: str                      # xray | ct | mri | path | derm | fundus | us
    dimension: str                     # D1..D6
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    """A single model response captured to disk."""
    example_id: str
    model_id: str
    prediction: str
    confidence: float | None           # in [0, 1]; None if not elicited
    raw_response: dict[str, Any]
    request_hash: str
    latency_ms: int
    timestamp: str
