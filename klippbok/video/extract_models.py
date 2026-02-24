"""Data models for image extraction from video clips.

These models describe HOW to extract reference images (ExtractionConfig),
WHAT happened during extraction (ExtractionResult), and the batch
summary (ExtractionReport). They are separate from the video pipeline
models because extraction produces images, not videos.

All models are immutable (frozen) — they represent facts about what
was extracted, not mutable state.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ExtractionStrategy(str, Enum):
    """How to choose which frame to extract from a video clip.

    - FIRST_FRAME: frame 0 — the standard I2V reference. Fast, deterministic.
    - BEST_FRAME: sample N frames, pick the sharpest by Laplacian variance.
      Good when the first frame is a fade-in or motion blur.
    - USER_SELECTED: read frame numbers from a JSON manifest. For curators
      who want to hand-pick the exact reference frame per clip.
    """
    FIRST_FRAME = "first_frame"
    BEST_FRAME = "best_frame"
    USER_SELECTED = "user_selected"


class ExtractionConfig(BaseModel):
    """Configuration for reference image extraction.

    Controls which strategy to use, how many frames to sample (for
    best_frame), and whether to overwrite existing extractions.
    """
    model_config = ConfigDict(frozen=True)

    strategy: ExtractionStrategy = Field(
        default=ExtractionStrategy.FIRST_FRAME,
        description=(
            "How to choose the frame to extract. "
            "'first_frame' = frame 0 (fast, standard for I2V). "
            "'best_frame' = sample frames and pick sharpest. "
            "'user_selected' = read from a JSON manifest."
        ),
    )
    sample_count: int = Field(
        default=10,
        ge=2,
        description=(
            "Number of frames to sample when strategy is 'best_frame'. "
            "More samples = better quality pick, but slower. "
            "Frames are sampled evenly across the clip duration."
        ),
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "If True, re-extract even if the output PNG already exists. "
            "If False, skip clips that already have a reference image."
        ),
    )


class ExtractionResult(BaseModel):
    """Result of extracting a reference image from one source file.

    Records what happened — which file was processed, where the output
    went, which frame was picked, and quality metrics. Failed extractions
    have success=False and an error message.
    """
    model_config = ConfigDict(frozen=True)

    source: Path
    """Path to the source file (video clip or image)."""

    output: Path | None = None
    """Path to the extracted PNG. None if extraction failed."""

    frame_number: int | None = None
    """Which frame was extracted (0-based). None for image pass-through."""

    strategy: ExtractionStrategy | None = None
    """Which strategy was used for this extraction."""

    sharpness: float | None = None
    """Laplacian variance of the extracted image. Higher = sharper."""

    source_type: str = "video"
    """Whether the source was a 'video' or 'image' file."""

    success: bool = True
    """True if extraction succeeded, False if it failed."""

    error: str | None = None
    """Error message if extraction failed. None on success."""

    skipped: bool = False
    """True if extraction was skipped (output already exists, overwrite=False)."""


class ImageValidation(BaseModel):
    """Quality validation result for an extracted image.

    Checks resolution, sharpness (Laplacian variance), and whether
    the image is blank (uniform color / black / white).
    """
    model_config = ConfigDict(frozen=True)

    path: Path
    """Path to the image that was validated."""

    width: int
    """Image width in pixels."""

    height: int
    """Image height in pixels."""

    sharpness: float
    """Laplacian variance — higher means sharper. Typical range 50-5000+."""

    is_blank: bool
    """True if the image is effectively uniform (black, white, or flat color)."""

    resolution_ok: bool
    """True if the image matches the expected dimensions."""

    expected_width: int | None = None
    """Expected width (from video metadata). None if not checked."""

    expected_height: int | None = None
    """Expected height (from video metadata). None if not checked."""


class ExtractionReport(BaseModel):
    """Batch extraction results across a directory.

    Aggregates per-file results with summary statistics.
    """
    model_config = ConfigDict(frozen=True)

    results: list[ExtractionResult] = Field(default_factory=list)
    """Per-source-file extraction results."""

    @property
    def total(self) -> int:
        """Total number of source files processed."""
        return len(self.results)

    @property
    def succeeded(self) -> int:
        """Number of successful extractions."""
        return sum(1 for r in self.results if r.success and not r.skipped)

    @property
    def failed(self) -> int:
        """Number of failed extractions."""
        return sum(1 for r in self.results if not r.success)

    @property
    def skipped(self) -> int:
        """Number of skipped extractions (already existed)."""
        return sum(1 for r in self.results if r.skipped)

    @property
    def videos(self) -> int:
        """Number of video sources processed."""
        return sum(1 for r in self.results if r.source_type == "video")

    @property
    def images(self) -> int:
        """Number of image sources processed (pass-through)."""
        return sum(1 for r in self.results if r.source_type == "image")
