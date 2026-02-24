"""Bucketing preview for dataset validation.

Training requires batches of clips with compatible dimensions. Bucketing
groups clips so each batch is uniform — avoiding padding waste and
ensuring efficient GPU utilization.

This module provides an OPT-IN preview of how clips would be bucketed.
It doesn't actually do bucketing for training (that's Phase 7+) — it
previews the distribution so curators can spot problems early:
- Buckets with too few samples (waste GPU time with padding)
- Extreme aspect ratio buckets (may need cropping)
- Uneven bucket distribution (training imbalance)

Bucket keys snap to a configurable step size (default 16px) matching
standard model architectures.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from klippbok.dataset.models import DatasetReport, SamplePair
from klippbok.video.models import IssueCode, Severity, ValidationIssue


# ---------------------------------------------------------------------------
# Bucketing models
# ---------------------------------------------------------------------------

class BucketAssignment(BaseModel):
    """A single sample's bucket assignment."""
    model_config = ConfigDict(frozen=True)

    stem: str
    """Sample identifier."""

    bucket_key: str
    """The bucket this sample belongs to (e.g. '320x480x17')."""

    width: int
    height: int
    frame_count: int


class BucketGroup(BaseModel):
    """A group of samples sharing the same bucket key."""
    model_config = ConfigDict(frozen=True)

    bucket_key: str
    """The bucket identifier (e.g. '320x480x17')."""

    count: int
    """Number of samples in this bucket."""

    samples: list[str] = Field(default_factory=list)
    """Stems of samples in this bucket."""


class BucketingResult(BaseModel):
    """Complete bucketing preview result."""
    model_config = ConfigDict(frozen=True)

    assignments: list[BucketAssignment] = Field(default_factory=list)
    """Per-sample bucket assignments."""

    buckets: list[BucketGroup] = Field(default_factory=list)
    """Bucket distribution (sorted by count descending)."""

    issues: list[ValidationIssue] = Field(default_factory=list)
    """Bucketing-related issues (e.g. undersized buckets)."""

    step_size: int = 16
    """Pixel step size used for bucket key computation."""

    @property
    def total_buckets(self) -> int:
        return len(self.buckets)

    @property
    def total_assigned(self) -> int:
        return len(self.assignments)

    @property
    def total_unassigned(self) -> int:
        """Samples that couldn't be bucketed (missing dimensions)."""
        return 0  # All samples with dimensions get assigned


# ---------------------------------------------------------------------------
# Bucket key computation
# ---------------------------------------------------------------------------

def compute_bucket_key(
    width: int,
    height: int,
    frame_count: int,
    step_size: int = 16,
) -> str:
    """Compute a bucket key by snapping dimensions to the step grid.

    Snaps width and height DOWN to the nearest multiple of step_size.
    Frame count is kept as-is (already constrained to 4n+1 by the pipeline).

    Examples:
        >>> compute_bucket_key(854, 480, 17)
        '848x480x17'
        >>> compute_bucket_key(320, 240, 81, step_size=16)
        '320x240x81'
        >>> compute_bucket_key(333, 245, 17)
        '320x240x17'

    Args:
        width: Video width in pixels.
        height: Video height in pixels.
        frame_count: Number of frames.
        step_size: Pixel step for snapping. Default 16.

    Returns:
        Bucket key string in format '{w}x{h}x{fc}'.
    """
    snapped_w = (width // step_size) * step_size
    snapped_h = (height // step_size) * step_size
    return f"{snapped_w}x{snapped_h}x{frame_count}"


# ---------------------------------------------------------------------------
# Bucketing preview
# ---------------------------------------------------------------------------

def preview_bucketing(
    report: DatasetReport,
    min_bucket_size: int = 2,
    step_size: int = 16,
) -> BucketingResult:
    """Preview how samples would be distributed into buckets.

    Only samples with known dimensions (width, height, frame_count) can
    be bucketed. Samples missing these fields are skipped.

    Args:
        report: Validated dataset report with sample metadata.
        min_bucket_size: Minimum samples per bucket. Buckets below this
            generate a BUCKET_UNDERSIZED warning.
        step_size: Pixel step for bucket key computation.

    Returns:
        BucketingResult with assignments, groups, and any issues.
    """
    assignments: list[BucketAssignment] = []
    bucket_map: dict[str, list[str]] = {}

    for ds in report.datasets:
        for sample in ds.samples:
            if sample.width is None or sample.height is None or sample.frame_count is None:
                continue

            key = compute_bucket_key(
                sample.width, sample.height, sample.frame_count, step_size,
            )
            assignments.append(BucketAssignment(
                stem=sample.stem,
                bucket_key=key,
                width=sample.width,
                height=sample.height,
                frame_count=sample.frame_count,
            ))
            bucket_map.setdefault(key, []).append(sample.stem)

    # Build bucket groups sorted by count descending
    buckets = sorted(
        [
            BucketGroup(bucket_key=key, count=len(stems), samples=stems)
            for key, stems in bucket_map.items()
        ],
        key=lambda b: (-b.count, b.bucket_key),
    )

    # Check for undersized buckets
    issues: list[ValidationIssue] = []
    for bucket in buckets:
        if bucket.count < min_bucket_size:
            issues.append(ValidationIssue(
                code=IssueCode.BUCKET_UNDERSIZED,
                severity=Severity.WARNING,
                message=(
                    f"Bucket '{bucket.bucket_key}' has only {bucket.count} sample(s) "
                    f"(minimum is {min_bucket_size}). "
                    f"Small buckets waste GPU time with padding. "
                    f"Consider cropping or resizing clips to match larger buckets."
                ),
                field="bucketing",
                actual=bucket.count,
                expected=min_bucket_size,
            ))

    return BucketingResult(
        assignments=assignments,
        buckets=buckets,
        issues=issues,
        step_size=step_size,
    )
