"""Klippbok VLM captioning pipeline.

Standalone caption generation, auditing, and scoring tools for video
training datasets. Supports Gemini (direct API) and Replicate backends.

Quick start:
    from klippbok.caption import caption_clips
    from klippbok.caption.models import CaptionConfig

    config = CaptionConfig(provider="gemini")
    results = caption_clips("./clips", config)

Scoring (no API needed):
    from klippbok.caption import score_caption, score_directory

    score = score_caption("A person walks through a sunlit forest...")
    print(score.overall, score.issues)
"""

from klippbok.caption.captioner import audit_captions, caption_clips
from klippbok.caption.models import AuditResult, CaptionConfig, CaptionResult
from klippbok.caption.scoring import (
    CaptionScore,
    ScoringConfig,
    format_score_report,
    score_caption,
    score_directory,
)

__all__ = [
    "caption_clips",
    "audit_captions",
    "CaptionConfig",
    "CaptionResult",
    "AuditResult",
    "score_caption",
    "score_directory",
    "format_score_report",
    "CaptionScore",
    "ScoringConfig",
]
