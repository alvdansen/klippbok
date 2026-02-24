"""Klippbok dataset validation data models.

Models for dataset-level validation: structure detection, sample pairing,
per-sample and per-dataset issue tracking, and aggregate reporting.

These operate at a higher level than klippbok.video.models — they describe
relationships BETWEEN files (clip ↔ caption ↔ reference image) rather than
properties of a single file.

All models are immutable (frozen) — they represent facts about a dataset's
state, not mutable processing state.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from klippbok.video.models import IssueCode, Severity, ValidationIssue


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrganizeLayout(str, Enum):
    """Output directory layout for the organize command.

    - flat: all files in one folder, stem-matched. Universal trainer compat
      (musubi-tuner, ai-toolkit, kohya, ostris).
    - klippbok: hierarchical structure separating targets from control signals.
      training/targets/ + training/signals/captions/ + training/signals/references/
    """
    FLAT = "flat"
    DIMLJUS = "klippbok"


class StructureType(str, Enum):
    """Detected directory layout of a dataset source.

    Klippbok supports two layouts:

    - flat: all files in one folder, paired by filename stem.
      Example: clip_001.mp4, clip_001.txt, clip_001.png
      This is the musubi-tuner / ai-toolkit convention.

    - klippbok: organized into training/targets/ and training/signals/ subfolders.
      Example: training/targets/clip_001.mp4, training/signals/captions/clip_001.txt
      This is the Klippbok convention that teaches users about target/signal separation.
    """
    FLAT = "flat"
    DIMLJUS = "klippbok"


# ---------------------------------------------------------------------------
# Sample-level models
# ---------------------------------------------------------------------------

class SamplePair(BaseModel):
    """A single training sample: one target video paired with its signals.

    The stem (filename without extension) is the join key — all files
    belonging to the same sample share the same stem.

    Issues are collected during validation and attached here so the report
    can show per-sample problems.
    """
    model_config = ConfigDict(frozen=True)

    stem: str
    """Filename stem that joins this sample's files (e.g. 'clip_001')."""

    target: Path
    """Path to the target video file."""

    caption: Path | None = None
    """Path to the caption .txt file, or None if missing."""

    reference: Path | None = None
    """Path to the reference image, or None if missing."""

    issues: list[ValidationIssue] = Field(default_factory=list)
    """Validation issues found for this sample."""

    # Video metadata cached during validation (optional — populated by validator)
    width: int | None = None
    """Video width in pixels (populated during validation)."""

    height: int | None = None
    """Video height in pixels (populated during validation)."""

    frame_count: int | None = None
    """Video frame count (populated during validation)."""

    fps: float | None = None
    """Video FPS (populated during validation)."""

    @property
    def is_valid(self) -> bool:
        """True if there are no error-severity issues."""
        return not any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Only the error-severity issues."""
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Only the warning-severity issues."""
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def has_caption(self) -> bool:
        """True if a caption file was found."""
        return self.caption is not None

    @property
    def has_reference(self) -> bool:
        """True if a reference image was found."""
        return self.reference is not None


# ---------------------------------------------------------------------------
# Dataset-level models
# ---------------------------------------------------------------------------

class DatasetValidation(BaseModel):
    """Validation result for a single dataset source folder.

    Combines structure detection, file pairing, and per-sample validation
    into one result object. Multiple DatasetValidation objects are
    aggregated into a DatasetReport.
    """
    model_config = ConfigDict(frozen=True)

    source_path: Path
    """Path to the dataset source folder."""

    structure: StructureType
    """Detected directory layout (flat or klippbok)."""

    samples: list[SamplePair] = Field(default_factory=list)
    """All discovered and validated training samples."""

    orphaned_files: list[Path] = Field(default_factory=list)
    """Files that couldn't be paired with any target video."""

    dataset_issues: list[ValidationIssue] = Field(default_factory=list)
    """Dataset-level issues (not tied to a specific sample).
    Examples: DATASET_EMPTY, cross-sample duplicates."""

    @property
    def total_samples(self) -> int:
        """Total number of training samples found."""
        return len(self.samples)

    @property
    def valid_samples(self) -> int:
        """Number of samples with no errors."""
        return sum(1 for s in self.samples if s.is_valid)

    @property
    def invalid_samples(self) -> int:
        """Number of samples with at least one error."""
        return self.total_samples - self.valid_samples

    @property
    def all_issues(self) -> list[ValidationIssue]:
        """All issues: dataset-level + per-sample, flattened."""
        issues = list(self.dataset_issues)
        for sample in self.samples:
            issues.extend(sample.issues)
        return issues

    @property
    def error_count(self) -> int:
        """Total number of error-severity issues."""
        return sum(1 for i in self.all_issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        """Total number of warning-severity issues."""
        return sum(1 for i in self.all_issues if i.severity == Severity.WARNING)

    @property
    def is_valid(self) -> bool:
        """True if the entire dataset has no errors."""
        return self.error_count == 0

    @property
    def issue_summary(self) -> dict[IssueCode, int]:
        """Count of each issue code across all samples."""
        counts: dict[IssueCode, int] = {}
        for issue in self.all_issues:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        return counts


class DatasetReport(BaseModel):
    """Aggregate report across all dataset sources in a config.

    This is the top-level result of running `validate_all()` — it spans
    multiple dataset source folders and provides cross-dataset checks
    (like duplicate detection across sources).
    """
    model_config = ConfigDict(frozen=True)

    datasets: list[DatasetValidation] = Field(default_factory=list)
    """Per-source validation results."""

    cross_dataset_issues: list[ValidationIssue] = Field(default_factory=list)
    """Issues that span multiple dataset sources (e.g. cross-source duplicates)."""

    @property
    def total_sources(self) -> int:
        """Number of dataset source folders."""
        return len(self.datasets)

    @property
    def total_samples(self) -> int:
        """Total samples across all sources."""
        return sum(d.total_samples for d in self.datasets)

    @property
    def valid_samples(self) -> int:
        """Total valid samples across all sources."""
        return sum(d.valid_samples for d in self.datasets)

    @property
    def invalid_samples(self) -> int:
        """Total invalid samples across all sources."""
        return self.total_samples - self.valid_samples

    @property
    def all_issues(self) -> list[ValidationIssue]:
        """Every issue across all sources + cross-dataset issues."""
        issues = list(self.cross_dataset_issues)
        for ds in self.datasets:
            issues.extend(ds.all_issues)
        return issues

    @property
    def error_count(self) -> int:
        """Total errors across all sources."""
        return sum(1 for i in self.all_issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        """Total warnings across all sources."""
        return sum(1 for i in self.all_issues if i.severity == Severity.WARNING)

    @property
    def is_valid(self) -> bool:
        """True if all datasets pass validation (no errors anywhere)."""
        return self.error_count == 0

    @property
    def issue_summary(self) -> dict[IssueCode, int]:
        """Count of each issue code across the entire report."""
        counts: dict[IssueCode, int] = {}
        for issue in self.all_issues:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Organize result models
# ---------------------------------------------------------------------------

class OrganizedSample(BaseModel):
    """Record of one sample's outcome during organize.

    Tracks where each file ended up (or why the sample was skipped).
    Used for the organize report and for generating trainer configs.
    """
    model_config = ConfigDict(frozen=True)

    stem: str
    """Filename stem of the sample."""

    target_dest: Path | None = None
    """Where the target video was placed (None if skipped)."""

    caption_dest: Path | None = None
    """Where the caption file was placed (None if missing/skipped)."""

    reference_dest: Path | None = None
    """Where the reference image was placed (None if missing/skipped)."""

    skipped: bool = False
    """True if this sample was excluded from output."""

    skip_reason: str = ""
    """Why the sample was skipped (empty if not skipped)."""

    frame_count: int | None = None
    """Frame count of the target video (for trainer config generation)."""

    width: int | None = None
    """Video width in pixels."""

    height: int | None = None
    """Video height in pixels."""


class OrganizeResult(BaseModel):
    """Complete result of running organize_dataset().

    Contains everything needed for the summary report, trainer config
    generation, and manifest output.
    """
    model_config = ConfigDict(frozen=True)

    output_dir: Path
    """Where organized files were placed."""

    layout: OrganizeLayout
    """Which directory layout was used."""

    organized: list[OrganizedSample] = Field(default_factory=list)
    """Samples that were successfully organized."""

    skipped: list[OrganizedSample] = Field(default_factory=list)
    """Samples that were excluded (errors or strict-mode warnings)."""

    trainer_configs: list[Path] = Field(default_factory=list)
    """Paths to generated trainer config files."""

    dry_run: bool = False
    """True if this was a preview run (no files touched)."""

    @property
    def organized_count(self) -> int:
        """Number of samples successfully organized."""
        return len(self.organized)

    @property
    def skipped_count(self) -> int:
        """Number of samples skipped."""
        return len(self.skipped)

    @property
    def total_count(self) -> int:
        """Total samples considered."""
        return self.organized_count + self.skipped_count
