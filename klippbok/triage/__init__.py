"""Clip triage — sort clips by content using reference image matching.

Triage is the step between splitting/normalizing and captioning.
It matches video clips against user-provided reference images to
automatically categorize scenes by concept type (character, setting, style, etc.).

Three modes:
  1. Reference-guided: user provides reference images in concepts/ folder
  2. Zero-shot: model auto-groups similar clips (future)
  3. Manual: user sorts clips themselves (no triage needed)

Usage:
    python -m klippbok.video triage <clips_dir> --concepts <concepts_dir>
"""

from klippbok.triage.models import (
    ClipMatch,
    ClipTriage,
    ConceptReference,
    ConceptType,
    SceneTriage,
    TriageReport,
    VideoTriageReport,
    resolve_concept_type,
)
from klippbok.triage.concepts import discover_concepts
from klippbok.triage.triage import organize_clips, triage_clips, triage_videos

__all__ = [
    "ClipMatch",
    "ClipTriage",
    "ConceptReference",
    "ConceptType",
    "SceneTriage",
    "TriageReport",
    "VideoTriageReport",
    "discover_concepts",
    "organize_clips",
    "resolve_concept_type",
    "triage_clips",
    "triage_videos",
]
