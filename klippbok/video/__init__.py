"""Klippbok video ingestion and scene detection.

Standalone tools for going from raw footage to clean training clips.
Works with any trainer — musubi-tuner, ai-toolkit, or Klippbok itself.

Quick start:
    from klippbok.video import probe_video, validate_clip
    from klippbok.config import KlippbokDataConfig

    meta = probe_video("clip.mp4")
    config = KlippbokDataConfig(datasets=[{"path": "."}])
    result = validate_clip(meta, config.video)
"""

# Auto-discover ffmpeg if installed via WinGet/Chocolatey/Scoop
import klippbok.video._ffmpeg  # noqa: F401

from klippbok.video.errors import (
    KlippbokVideoError,
    ExtractionError,
    FFmpegNotFoundError,
    ProbeError,
    SceneDetectNotFoundError,
    SplitError,
)
from klippbok.video.extract_models import (
    ExtractionConfig,
    ExtractionReport,
    ExtractionResult,
    ExtractionStrategy,
    ImageValidation,
)
from klippbok.video.models import (
    ClipInfo,
    ClipValidation,
    IssueCode,
    ScanReport,
    SceneBoundary,
    Severity,
    ValidationIssue,
    VideoMetadata,
)
from klippbok.video.probe import probe_directory, probe_video
from klippbok.video.validate import (
    format_scan_report,
    nearest_valid_frame_count,
    validate_clip,
    validate_directory,
)

__all__ = [
    # Errors
    "KlippbokVideoError",
    "ExtractionError",
    "FFmpegNotFoundError",
    "ProbeError",
    "SceneDetectNotFoundError",
    "SplitError",
    # Video models
    "ClipInfo",
    "ClipValidation",
    "IssueCode",
    "ScanReport",
    "SceneBoundary",
    "Severity",
    "ValidationIssue",
    "VideoMetadata",
    # Extraction models
    "ExtractionConfig",
    "ExtractionReport",
    "ExtractionResult",
    "ExtractionStrategy",
    "ImageValidation",
    # Probe
    "probe_video",
    "probe_directory",
    # Validate
    "validate_clip",
    "validate_directory",
    "nearest_valid_frame_count",
    "format_scan_report",
]
