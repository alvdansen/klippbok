"""Klippbok video pipeline data models.

Pydantic v2 models for video metadata, validation results, scene boundaries,
and processing reports. These models flow through the entire pipeline:

    probe → validate → detect scenes → split/normalize → report

All models are immutable (frozen) — they represent facts about videos,
not mutable state.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """How serious a validation issue is.

    - error: clip cannot be used as-is, must be fixed or excluded
    - warning: clip can be used but will be re-encoded/trimmed
    - info: informational, no action needed
    """
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class IssueCode(str, Enum):
    """Machine-readable validation issue codes.

    Each code maps to a specific structural check. Downstream tools can
    filter or react to specific codes without parsing human-readable messages.

    Naming convention: {WHAT}_{PROBLEM}
    """
    # Resolution issues
    RESOLUTION_BELOW_MIN = "RESOLUTION_BELOW_MIN"
    RESOLUTION_BELOW_TARGET = "RESOLUTION_BELOW_TARGET"
    RESOLUTION_ABOVE_TARGET = "RESOLUTION_ABOVE_TARGET"

    # Frame rate
    FPS_MISMATCH = "FPS_MISMATCH"

    # Frame count
    INVALID_FRAME_COUNT = "INVALID_FRAME_COUNT"
    FRAME_COUNT_TOO_SHORT = "FRAME_COUNT_TOO_SHORT"
    FRAME_COUNT_LARGE_TRIM = "FRAME_COUNT_LARGE_TRIM"

    # SAR (sample aspect ratio)
    NON_SQUARE_SAR = "NON_SQUARE_SAR"

    # Scene detection
    SCENE_CUT_DETECTED = "SCENE_CUT_DETECTED"

    # File issues
    PROBE_FAILED = "PROBE_FAILED"
    NO_VIDEO_STREAM = "NO_VIDEO_STREAM"

    # Dataset-level issues (Phase 4 — dataset validation)
    CAPTION_MISSING = "CAPTION_MISSING"
    CAPTION_EMPTY = "CAPTION_EMPTY"
    CAPTION_TOO_LONG = "CAPTION_TOO_LONG"
    REFERENCE_MISSING = "REFERENCE_MISSING"
    REFERENCE_BLANK = "REFERENCE_BLANK"
    FILE_TYPE_INVALID = "FILE_TYPE_INVALID"
    FILE_CORRUPTED = "FILE_CORRUPTED"
    DUPLICATE_DETECTED = "DUPLICATE_DETECTED"
    ORPHANED_FILE = "ORPHANED_FILE"
    EXPOSURE_OUT_OF_RANGE = "EXPOSURE_OUT_OF_RANGE"
    BLUR_BELOW_THRESHOLD = "BLUR_BELOW_THRESHOLD"
    MOTION_BELOW_MIN = "MOTION_BELOW_MIN"
    MOTION_ABOVE_MAX = "MOTION_ABOVE_MAX"
    DATASET_EMPTY = "DATASET_EMPTY"
    BUCKET_UNDERSIZED = "BUCKET_UNDERSIZED"


# ---------------------------------------------------------------------------
# Video metadata from ffprobe
# ---------------------------------------------------------------------------

class VideoMetadata(BaseModel):
    """Everything ffprobe tells us about a video file.

    This is the raw probe result — no validation or judgment, just facts.
    The validate module compares these facts against a VideoConfig to
    produce ValidationIssues.

    Fields marked Optional may be unavailable for some codecs/containers.
    """
    model_config = ConfigDict(frozen=True)

    path: Path
    """Absolute path to the video file."""

    width: int
    """Frame width in pixels."""

    height: int
    """Frame height in pixels."""

    fps: float
    """Frames per second (from r_frame_rate, averaged if VFR)."""

    frame_count: int
    """Total number of frames. May be estimated from duration × fps
    if the container doesn't store nb_frames."""

    duration: float
    """Duration in seconds."""

    codec: str
    """Video codec name (e.g. 'h264', 'hevc', 'prores')."""

    pix_fmt: str | None = None
    """Pixel format (e.g. 'yuv420p', 'yuv422p10le')."""

    sar: str = "1:1"
    """Sample aspect ratio. '1:1' means square pixels (normal).
    Non-square SAR means the display aspect differs from storage."""

    bit_rate: int | None = None
    """Video stream bit rate in bits/second, if available."""

    file_size: int | None = None
    """File size in bytes."""

    has_audio: bool = False
    """Whether the file contains an audio stream."""

    container: str | None = None
    """Container format name (e.g. 'mov', 'mp4', 'matroska')."""

    @property
    def aspect_ratio(self) -> float:
        """Display aspect ratio as a float (width / height).

        Accounts for non-square SAR if present.
        """
        sar_w, sar_h = 1, 1
        if self.sar and ":" in self.sar:
            parts = self.sar.split(":")
            try:
                sar_w, sar_h = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                pass
        return (self.width * sar_w) / (self.height * sar_h)

    @property
    def is_square_sar(self) -> bool:
        """True if pixels are square (SAR is 1:1)."""
        return self.sar in ("1:1", "1/1", None)

    @property
    def display_resolution(self) -> str:
        """Human-readable resolution string like '1920x1080'."""
        return f"{self.width}x{self.height}"


# ---------------------------------------------------------------------------
# Validation results
# ---------------------------------------------------------------------------

class ValidationIssue(BaseModel):
    """A single validation finding about a video clip.

    Every issue has:
    - A machine-readable code for programmatic handling
    - A severity level (error/warning/info)
    - A human-readable message explaining what's wrong AND how to fix it
    - The field that triggered the issue (for structured error reports)
    """
    model_config = ConfigDict(frozen=True)

    code: IssueCode
    """Machine-readable issue identifier."""

    severity: Severity
    """How serious this issue is."""

    message: str
    """Human-readable description: what's wrong + how to fix it."""

    field: str
    """Which VideoConfig field this issue relates to (e.g. 'fps', 'resolution')."""

    actual: Any = None
    """The actual value found in the clip (for error messages)."""

    expected: Any = None
    """The expected/target value from config (for error messages)."""


class ClipValidation(BaseModel):
    """Complete validation result for a single video clip.

    Combines the raw metadata with all validation findings and
    a summary of what processing is needed.
    """
    model_config = ConfigDict(frozen=True)

    metadata: VideoMetadata
    """The probed metadata for this clip."""

    issues: list[ValidationIssue] = Field(default_factory=list)
    """All validation findings, in check order."""

    needs_reencode: bool = False
    """True if any issue requires re-encoding (fps change, resolution
    change, SAR correction). False means the clip can be stream-copied."""

    recommended_frame_count: int | None = None
    """The nearest valid 4n+1 frame count ≤ the actual count.
    None if the actual count is already valid."""

    @property
    def is_valid(self) -> bool:
        """True if there are no errors (warnings are acceptable)."""
        return not any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Only the error-severity issues."""
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Only the warning-severity issues."""
        return [i for i in self.issues if i.severity == Severity.WARNING]


# ---------------------------------------------------------------------------
# Scene detection
# ---------------------------------------------------------------------------

class SceneBoundary(BaseModel):
    """A detected scene cut within a video.

    Scene boundaries indicate where the content changes abruptly —
    a visual discontinuity that would break temporal coherence in training.
    """
    model_config = ConfigDict(frozen=True)

    frame_number: int
    """Frame index where the cut occurs (0-based)."""

    timecode: float
    """Time in seconds where the cut occurs."""

    confidence: float
    """Detection confidence (0.0–1.0). Higher = more certain it's a real cut."""


# ---------------------------------------------------------------------------
# Processing results
# ---------------------------------------------------------------------------

class ClipInfo(BaseModel):
    """Result of processing a single clip (normalize or split).

    Records what happened during processing for the manifest.
    """
    model_config = ConfigDict(frozen=True)

    source: Path
    """Path to the original source file."""

    output: Path
    """Path to the produced output file."""

    frame_count: int
    """Frame count of the output clip."""

    duration: float
    """Duration of the output clip in seconds."""

    width: int
    """Width of the output clip."""

    height: int
    """Height of the output clip."""

    fps: float
    """FPS of the output clip."""

    was_reencoded: bool
    """True if the clip was re-encoded (not stream-copied)."""

    trimmed_frames: int = 0
    """Number of frames trimmed from the end to reach valid 4n+1 count."""

    scene_index: int | None = None
    """If split from a longer video, which scene number (0-based)."""


# ---------------------------------------------------------------------------
# Scan report
# ---------------------------------------------------------------------------

class ScanReport(BaseModel):
    """Results of scanning a directory of video clips.

    Aggregates validation results across all clips in a folder,
    providing both per-clip details and summary statistics.
    """
    model_config = ConfigDict(frozen=True)

    directory: Path
    """The scanned directory."""

    clips: list[ClipValidation] = Field(default_factory=list)
    """Per-clip validation results."""

    @property
    def total(self) -> int:
        """Total number of clips scanned."""
        return len(self.clips)

    @property
    def valid(self) -> int:
        """Number of clips with no errors."""
        return sum(1 for c in self.clips if c.is_valid)

    @property
    def invalid(self) -> int:
        """Number of clips with at least one error."""
        return self.total - self.valid

    @property
    def needs_reencode(self) -> int:
        """Number of clips that need re-encoding."""
        return sum(1 for c in self.clips if c.needs_reencode)

    @property
    def all_issues(self) -> list[ValidationIssue]:
        """Flattened list of all issues across all clips."""
        return [issue for clip in self.clips for issue in clip.issues]

    @property
    def issue_summary(self) -> dict[IssueCode, int]:
        """Count of each issue code across all clips."""
        counts: dict[IssueCode, int] = {}
        for issue in self.all_issues:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        return counts
