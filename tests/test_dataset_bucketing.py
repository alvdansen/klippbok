"""Tests for klippbok.dataset.bucketing — bucket key computation and preview."""

from __future__ import annotations

from pathlib import Path

import pytest

from klippbok.dataset.bucketing import (
    BucketAssignment,
    BucketGroup,
    BucketingResult,
    compute_bucket_key,
    preview_bucketing,
)
from klippbok.dataset.models import (
    DatasetReport,
    DatasetValidation,
    SamplePair,
    StructureType,
)
from klippbok.video.models import IssueCode


# ---------------------------------------------------------------------------
# compute_bucket_key tests
# ---------------------------------------------------------------------------

class TestComputeBucketKey:
    def test_exact_multiple(self):
        """Dimensions already on grid stay unchanged."""
        assert compute_bucket_key(320, 480, 17) == "320x480x17"

    def test_snap_down(self):
        """Non-multiples snap down to nearest grid point."""
        assert compute_bucket_key(333, 245, 17) == "320x240x17"

    def test_854_snaps(self):
        """854 (common 480p width) snaps to 848."""
        assert compute_bucket_key(854, 480, 17) == "848x480x17"

    def test_1280x720(self):
        """720p stays on grid."""
        assert compute_bucket_key(1280, 720, 81) == "1280x720x81"

    def test_custom_step_size(self):
        assert compute_bucket_key(100, 100, 17, step_size=32) == "96x96x17"

    def test_step_size_1(self):
        """Step size 1 means no snapping."""
        assert compute_bucket_key(333, 245, 17, step_size=1) == "333x245x17"

    def test_frame_count_preserved(self):
        """Frame count is not snapped — it's already 4n+1."""
        assert compute_bucket_key(320, 240, 81) == "320x240x81"
        assert compute_bucket_key(320, 240, 1) == "320x240x1"

    def test_small_dimensions(self):
        """Dimensions smaller than step size snap to 0."""
        assert compute_bucket_key(10, 10, 5, step_size=16) == "0x0x5"

    def test_different_frame_counts_different_keys(self):
        k1 = compute_bucket_key(320, 240, 17)
        k2 = compute_bucket_key(320, 240, 81)
        assert k1 != k2


# ---------------------------------------------------------------------------
# BucketingResult model tests
# ---------------------------------------------------------------------------

class TestBucketingModels:
    def test_bucket_assignment_frozen(self):
        ba = BucketAssignment(
            stem="clip", bucket_key="320x240x17",
            width=320, height=240, frame_count=17,
        )
        with pytest.raises(Exception):
            ba.stem = "other"  # type: ignore[misc]

    def test_bucket_group_frozen(self):
        bg = BucketGroup(bucket_key="320x240x17", count=3, samples=["a", "b", "c"])
        assert bg.count == 3

    def test_bucketing_result_properties(self):
        br = BucketingResult(
            assignments=[
                BucketAssignment(stem="a", bucket_key="k1", width=320, height=240, frame_count=17),
                BucketAssignment(stem="b", bucket_key="k1", width=320, height=240, frame_count=17),
            ],
            buckets=[BucketGroup(bucket_key="k1", count=2, samples=["a", "b"])],
            step_size=16,
        )
        assert br.total_buckets == 1
        assert br.total_assigned == 2


# ---------------------------------------------------------------------------
# preview_bucketing tests
# ---------------------------------------------------------------------------

def _sample_with_dims(stem: str, w: int, h: int, fc: int) -> SamplePair:
    return SamplePair(
        stem=stem,
        target=Path(f"/data/{stem}.mp4"),
        width=w, height=h, frame_count=fc,
    )


def _make_report_with_samples(samples: list[SamplePair]) -> DatasetReport:
    ds = DatasetValidation(
        source_path=Path("/data"),
        structure=StructureType.FLAT,
        samples=samples,
    )
    return DatasetReport(datasets=[ds])


class TestPreviewBucketing:
    def test_single_bucket(self):
        samples = [
            _sample_with_dims("a", 320, 240, 17),
            _sample_with_dims("b", 320, 240, 17),
            _sample_with_dims("c", 320, 240, 17),
        ]
        report = _make_report_with_samples(samples)
        result = preview_bucketing(report)
        assert result.total_buckets == 1
        assert result.total_assigned == 3
        assert result.buckets[0].count == 3

    def test_multiple_buckets(self):
        samples = [
            _sample_with_dims("a", 320, 240, 17),
            _sample_with_dims("b", 640, 480, 17),
            _sample_with_dims("c", 320, 240, 81),
        ]
        report = _make_report_with_samples(samples)
        result = preview_bucketing(report)
        assert result.total_buckets == 3  # all different

    def test_snapping_groups_similar(self):
        """Clips with slightly different widths snap to same bucket."""
        samples = [
            _sample_with_dims("a", 320, 240, 17),
            _sample_with_dims("b", 325, 240, 17),  # snaps to 320
            _sample_with_dims("c", 330, 240, 17),  # snaps to 320
        ]
        report = _make_report_with_samples(samples)
        result = preview_bucketing(report)
        assert result.total_buckets == 1

    def test_undersized_bucket_warning(self):
        """Buckets with fewer than min_bucket_size get warned."""
        samples = [
            _sample_with_dims("a", 320, 240, 17),
            _sample_with_dims("b", 640, 480, 17),  # alone in its bucket
        ]
        report = _make_report_with_samples(samples)
        result = preview_bucketing(report, min_bucket_size=2)
        assert len(result.issues) == 2  # both buckets have 1 sample
        assert all(i.code == IssueCode.BUCKET_UNDERSIZED for i in result.issues)

    def test_no_undersized_warning_when_enough(self):
        samples = [
            _sample_with_dims("a", 320, 240, 17),
            _sample_with_dims("b", 320, 240, 17),
        ]
        report = _make_report_with_samples(samples)
        result = preview_bucketing(report, min_bucket_size=2)
        assert len(result.issues) == 0

    def test_skips_samples_without_dimensions(self):
        """Samples missing width/height/frame_count are skipped."""
        samples = [
            SamplePair(stem="no_dims", target=Path("/data/no_dims.mp4")),
            _sample_with_dims("with_dims", 320, 240, 17),
        ]
        report = _make_report_with_samples(samples)
        result = preview_bucketing(report)
        assert result.total_assigned == 1

    def test_custom_step_size(self):
        samples = [
            _sample_with_dims("a", 100, 100, 17),
        ]
        report = _make_report_with_samples(samples)
        result = preview_bucketing(report, step_size=32)
        assert result.step_size == 32
        assert result.assignments[0].bucket_key == "96x96x17"

    def test_sorted_by_count_descending(self):
        samples = [
            _sample_with_dims("a", 320, 240, 17),
            _sample_with_dims("b", 320, 240, 17),
            _sample_with_dims("c", 320, 240, 17),
            _sample_with_dims("d", 640, 480, 17),
        ]
        report = _make_report_with_samples(samples)
        result = preview_bucketing(report)
        assert result.buckets[0].count >= result.buckets[-1].count

    def test_empty_report(self):
        report = DatasetReport()
        result = preview_bucketing(report)
        assert result.total_buckets == 0
        assert result.total_assigned == 0

    def test_multi_source(self):
        """Bucketing spans all dataset sources."""
        ds1 = DatasetValidation(
            source_path=Path("/ds1"),
            structure=StructureType.FLAT,
            samples=[_sample_with_dims("a", 320, 240, 17)],
        )
        ds2 = DatasetValidation(
            source_path=Path("/ds2"),
            structure=StructureType.FLAT,
            samples=[_sample_with_dims("b", 320, 240, 17)],
        )
        report = DatasetReport(datasets=[ds1, ds2])
        result = preview_bucketing(report)
        assert result.total_assigned == 2
        assert result.total_buckets == 1  # same dimensions
