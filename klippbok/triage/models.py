"""Data models for clip triage — sorting clips by content.

Triage matches video clips against user-provided reference images to
automatically categorize scenes. A reference photo of a character gets
matched against sampled frames from each clip using CLIP embeddings.

Concept types correspond to captioning use cases — once clips are
triaged, the right captioning prompt is automatically selected.

TYPE_ALIASES maps common folder names to canonical types so users
don't have to memorize exact names. "humans/" works just as well
as "character/".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ConceptType(str, Enum):
    """Types of concepts that can appear in training clips.

    These map directly to captioning use cases — a CHARACTER clip
    gets the character prompt (skip appearance, describe action),
    a SETTING clip gets the object/setting prompt, etc.
    """
    CHARACTER = "character"
    STYLE = "style"
    MOTION = "motion"
    OBJECT = "object"
    SETTING = "setting"


# Maps common folder names to canonical concept types.
# Lookup is case-insensitive with hyphens/underscores/spaces stripped.
# This lets users name folders naturally ("humans", "People", "actors")
# and Klippbok still understands what they mean.
TYPE_ALIASES: dict[str, ConceptType] = {
    # Character
    "character": ConceptType.CHARACTER,
    "characters": ConceptType.CHARACTER,
    "person": ConceptType.CHARACTER,
    "people": ConceptType.CHARACTER,
    "human": ConceptType.CHARACTER,
    "humans": ConceptType.CHARACTER,
    "face": ConceptType.CHARACTER,
    "faces": ConceptType.CHARACTER,
    "actor": ConceptType.CHARACTER,
    "actors": ConceptType.CHARACTER,
    # Style
    "style": ConceptType.STYLE,
    "styles": ConceptType.STYLE,
    "aesthetic": ConceptType.STYLE,
    "aesthetics": ConceptType.STYLE,
    "look": ConceptType.STYLE,
    "looks": ConceptType.STYLE,
    # Motion
    "motion": ConceptType.MOTION,
    "movement": ConceptType.MOTION,
    "action": ConceptType.MOTION,
    "actions": ConceptType.MOTION,
    # Object
    "object": ConceptType.OBJECT,
    "objects": ConceptType.OBJECT,
    "thing": ConceptType.OBJECT,
    "things": ConceptType.OBJECT,
    "item": ConceptType.OBJECT,
    "items": ConceptType.OBJECT,
    "prop": ConceptType.OBJECT,
    "props": ConceptType.OBJECT,
    # Setting
    "setting": ConceptType.SETTING,
    "settings": ConceptType.SETTING,
    "location": ConceptType.SETTING,
    "locations": ConceptType.SETTING,
    "place": ConceptType.SETTING,
    "places": ConceptType.SETTING,
    "environment": ConceptType.SETTING,
    "environments": ConceptType.SETTING,
    "scene": ConceptType.SETTING,
    "scenes": ConceptType.SETTING,
    "background": ConceptType.SETTING,
    "backgrounds": ConceptType.SETTING,
}

# File extensions recognized for each media type
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def resolve_concept_type(folder_name: str) -> ConceptType | None:
    """Map a folder name to a canonical concept type.

    Case-insensitive. Strips hyphens, underscores, and spaces before
    looking up in TYPE_ALIASES. Returns None if the folder name isn't
    recognized — the reference images can still be used for matching,
    but won't auto-select a captioning prompt.

    Args:
        folder_name: The folder name to resolve (e.g. "humans", "People").

    Returns:
        The canonical ConceptType, or None if unrecognized.

    Examples:
        >>> resolve_concept_type("character")
        ConceptType.CHARACTER
        >>> resolve_concept_type("Humans")
        ConceptType.CHARACTER
        >>> resolve_concept_type("my-custom-folder")
        None
    """
    normalized = folder_name.lower().strip().replace("-", "").replace("_", "").replace(" ", "")
    return TYPE_ALIASES.get(normalized)


@dataclass(frozen=True)
class ConceptReference:
    """A reference image representing a known concept.

    Discovered from the concepts/ folder structure:
        concepts/character/holly.jpg -> ConceptReference(
            name="holly",
            concept_type=ConceptType.CHARACTER,
            image_path=Path("concepts/character/holly.jpg"),
            folder_name="character",
        )
    """
    name: str
    """Concept name derived from filename (without extension)."""

    concept_type: ConceptType | None
    """Canonical type, or None if the folder wasn't recognized."""

    image_path: Path
    """Absolute path to the reference image."""

    folder_name: str
    """Original folder name as the user created it."""


@dataclass(frozen=True)
class ClipMatch:
    """A match between a clip frame and a concept reference.

    Produced by comparing CLIP embeddings of sampled clip frames
    against concept reference embeddings.
    """
    concept: ConceptReference
    """Which concept reference was matched."""

    similarity: float
    """Cosine similarity score (0.0 to 1.0). Higher = stronger match."""

    best_frame_index: int
    """Which of the sampled frames had the highest similarity."""


@dataclass
class ClipTriage:
    """Triage result for a single video clip.

    Contains all matches above threshold, sorted by similarity.
    A clip can match multiple concepts (e.g. a character in front
    of a recognized setting).
    """
    clip_path: Path
    """Path to the video clip."""

    matches: list[ClipMatch] = field(default_factory=list)
    """All concept matches above threshold, sorted by similarity (highest first)."""

    has_text_overlay: bool = False
    """True if the clip was detected as containing title cards, credits, or text overlays."""

    text_overlay_score: float = 0.0
    """CLIP similarity score for text overlay detection (for debugging/tuning)."""

    @property
    def best_match(self) -> ClipMatch | None:
        """The strongest match, or None if no matches."""
        if not self.matches:
            return None
        return self.matches[0]  # Already sorted highest-first

    @property
    def is_matched(self) -> bool:
        """True if at least one concept was matched."""
        return len(self.matches) > 0

    @property
    def concept_types(self) -> list[ConceptType]:
        """Unique concept types present in this clip's matches."""
        seen: set[ConceptType] = set()
        result: list[ConceptType] = []
        for m in self.matches:
            if m.concept.concept_type and m.concept.concept_type not in seen:
                seen.add(m.concept.concept_type)
                result.append(m.concept.concept_type)
        return result


@dataclass
class TriageReport:
    """Complete triage results for a directory of clips.

    Includes per-clip matches, the list of concept references used,
    and summary statistics.
    """
    clips: list[ClipTriage] = field(default_factory=list)
    """Per-clip triage results."""

    concepts: list[ConceptReference] = field(default_factory=list)
    """All concept references that were used for matching."""

    threshold: float = 0.70
    """Similarity threshold used for matching."""

    model_name: str = "openai/clip-vit-base-patch32"
    """CLIP model used for embeddings."""

    @property
    def total(self) -> int:
        """Total number of clips processed."""
        return len(self.clips)

    @property
    def matched_count(self) -> int:
        """Number of clips with at least one match."""
        return sum(1 for c in self.clips if c.is_matched)

    @property
    def unmatched_count(self) -> int:
        """Number of clips with no matches."""
        return sum(1 for c in self.clips if not c.is_matched)

    @property
    def text_overlay_count(self) -> int:
        """Number of clips flagged as text overlays / title cards."""
        return sum(1 for c in self.clips if c.has_text_overlay)


# ---------------------------------------------------------------------------
# Scene-level triage (for raw/long videos)
# ---------------------------------------------------------------------------

@dataclass
class SceneTriage:
    """Triage result for a single scene within a long video.

    Produced by scene-aware triage: detect scenes in a raw video,
    sample 1-2 frames per scene, match against references. This lets
    users triage BEFORE ingesting — only split the scenes they want.

    Unlike ClipTriage (which operates on pre-cut clips), SceneTriage
    tracks scene boundaries (start/end times) so filtered ingest can
    split only matching scenes.
    """
    source_video: Path
    """Path to the source video file this scene belongs to."""

    scene_index: int
    """Zero-based index of this scene within its source video."""

    start_time: float
    """Scene start time in seconds."""

    end_time: float
    """Scene end time in seconds."""

    matches: list[ClipMatch] = field(default_factory=list)
    """Concept matches above threshold, sorted by similarity (highest first)."""

    has_text_overlay: bool = False
    """True if the scene was detected as containing text overlays."""

    text_overlay_score: float = 0.0
    """CLIP similarity score for text overlay detection."""

    @property
    def duration(self) -> float:
        """Scene duration in seconds."""
        return self.end_time - self.start_time

    @property
    def best_match(self) -> ClipMatch | None:
        """The strongest match, or None if no matches."""
        if not self.matches:
            return None
        return self.matches[0]

    @property
    def is_matched(self) -> bool:
        """True if at least one concept was matched."""
        return len(self.matches) > 0


@dataclass
class VideoTriageReport:
    """Complete scene-level triage results for a set of raw videos.

    Groups SceneTriage results by source video. Used when triaging
    long videos before ingesting — the report becomes a scene triage
    manifest that filtered ingest reads.
    """
    scenes: list[SceneTriage] = field(default_factory=list)
    """All scene triage results across all videos."""

    concepts: list[ConceptReference] = field(default_factory=list)
    """Concept references used for matching."""

    threshold: float = 0.70
    """Similarity threshold used for matching."""

    model_name: str = "openai/clip-vit-base-patch32"
    """CLIP model used for embeddings."""

    scene_detection_threshold: float = 27.0
    """Scene detection threshold used for splitting raw videos."""

    frames_per_scene: int = 2
    """Number of frames sampled per scene for matching."""

    @property
    def total(self) -> int:
        """Total number of scenes processed."""
        return len(self.scenes)

    @property
    def matched_count(self) -> int:
        """Number of scenes with at least one match."""
        return sum(1 for s in self.scenes if s.is_matched)

    @property
    def unmatched_count(self) -> int:
        """Number of scenes with no matches."""
        return sum(1 for s in self.scenes if not s.is_matched)

    @property
    def text_overlay_count(self) -> int:
        """Number of scenes flagged as text overlays."""
        return sum(1 for s in self.scenes if s.has_text_overlay)

    @property
    def videos(self) -> list[Path]:
        """Unique source video paths, in order of first appearance."""
        seen: set[Path] = set()
        result: list[Path] = []
        for s in self.scenes:
            if s.source_video not in seen:
                seen.add(s.source_video)
                result.append(s.source_video)
        return result
