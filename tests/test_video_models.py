"""Tests for klippbok.video.models — pure Python, no external tools.

Tests the Pydantic models for video metadata, validation issues,
scan reports, etc. All data is hand-crafted — no ffprobe needed.
"""

from pathlib import Path

import pytest

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


# ---------------------------------------------------------------------------
# VideoMetadata
# ---------------------------------------------------------------------------

class TestVideoMetadata:
    """Tests for the VideoMetadata model."""

    def test_basic_creation(self) -> None:
        """Create metadata with required fields."""
        meta = VideoMetadata(
            path=Path("/test/clip.mp4"),
            width=1920,
            height=1080,
            fps=24.0,
            frame_count=240,
            duration=10.0,
            codec="h264",
        )
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.fps == 24.0
        assert meta.frame_count == 240
        assert meta.duration == 10.0
        assert meta.codec == "h264"

    def test_default_sar(self) -> None:
        """SAR defaults to 1:1 (square pixels)."""
        meta = VideoMetadata(
            path=Path("/test/clip.mp4"),
            width=1920, height=1080, fps=24.0,
            frame_count=240, duration=10.0, codec="h264",
        )
        assert meta.sar == "1:1"
        assert meta.is_square_sar is True

    def test_non_square_sar(self) -> None:
        """Non-square SAR is detected correctly."""
        meta = VideoMetadata(
            path=Path("/test/clip.mp4"),
            width=720, height=480, fps=29.97,
            frame_count=300, duration=10.01, codec="h264",
            sar="32:27",
        )
        assert meta.sar == "32:27"
        assert meta.is_square_sar is False

    def test_display_resolution(self) -> None:
        """Display resolution is formatted as WxH."""
        meta = VideoMetadata(
            path=Path("/test/clip.mp4"),
            width=1920, height=1080, fps=24.0,
            frame_count=240, duration=10.0, codec="h264",
        )
        assert meta.display_resolution == "1920x1080"

    def test_aspect_ratio_square_sar(self) -> None:
        """Aspect ratio calculation with square pixels."""
        meta = VideoMetadata(
            path=Path("/test/clip.mp4"),
            width=1920, height=1080, fps=24.0,
            frame_count=240, duration=10.0, codec="h264",
            sar="1:1",
        )
        assert abs(meta.aspect_ratio - (1920 / 1080)) < 0.001

    def test_aspect_ratio_non_square_sar(self) -> None:
        """Aspect ratio accounts for non-square SAR."""
        meta = VideoMetadata(
            path=Path("/test/clip.mp4"),
            width=720, height=480, fps=29.97,
            frame_count=300, duration=10.01, codec="h264",
            sar="4:3",
        )
        # Display: (720 * 4) / (480 * 3) = 2880 / 1440 = 2.0
        assert abs(meta.aspect_ratio - 2.0) < 0.001

    def test_optional_fields(self) -> None:
        """Optional fields default to None/False."""
        meta = VideoMetadata(
            path=Path("/test/clip.mp4"),
            width=1920, height=1080, fps=24.0,
            frame_count=240, duration=10.0, codec="h264",
        )
        assert meta.pix_fmt is None
        assert meta.bit_rate is None
        assert meta.file_size is None
        assert meta.has_audio is False
        assert meta.container is None

    def test_all_fields(self) -> None:
        """Create metadata with all fields populated."""
        meta = VideoMetadata(
            path=Path("/test/clip.mp4"),
            width=1920, height=1080, fps=23.976,
            frame_count=240, duration=10.01, codec="h264",
            pix_fmt="yuv420p", sar="1:1",
            bit_rate=5_000_000, file_size=6_250_000,
            has_audio=True, container="mov",
        )
        assert meta.pix_fmt == "yuv420p"
        assert meta.bit_rate == 5_000_000
        assert meta.file_size == 6_250_000
        assert meta.has_audio is True
        assert meta.container == "mov"

    def test_frozen(self) -> None:
        """Metadata is immutable."""
        meta = VideoMetadata(
            path=Path("/test/clip.mp4"),
            width=1920, height=1080, fps=24.0,
            frame_count=240, duration=10.0, codec="h264",
        )
        with pytest.raises(Exception):
            meta.width = 1280  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ValidationIssue
# ---------------------------------------------------------------------------

class TestValidationIssue:
    """Tests for ValidationIssue model."""

    def test_creation(self) -> None:
        """Create an issue with all fields."""
        issue = ValidationIssue(
            code=IssueCode.FPS_MISMATCH,
            severity=Severity.WARNING,
            message="FPS is 30 — target is 16",
            field="fps",
            actual=30,
            expected=16,
        )
        assert issue.code == IssueCode.FPS_MISMATCH
        assert issue.severity == Severity.WARNING
        assert issue.field == "fps"

    def test_optional_actual_expected(self) -> None:
        """actual and expected are optional."""
        issue = ValidationIssue(
            code=IssueCode.PROBE_FAILED,
            severity=Severity.ERROR,
            message="Failed to probe",
            field="file",
        )
        assert issue.actual is None
        assert issue.expected is None


# ---------------------------------------------------------------------------
# ClipValidation
# ---------------------------------------------------------------------------

class TestClipValidation:
    """Tests for ClipValidation model."""

    def _make_meta(self, **kwargs) -> VideoMetadata:
        """Helper: create VideoMetadata with defaults."""
        defaults = dict(
            path=Path("/test/clip.mp4"),
            width=854, height=480, fps=16.0,
            frame_count=81, duration=5.0625, codec="h264",
        )
        defaults.update(kwargs)
        return VideoMetadata(**defaults)

    def test_valid_clip(self) -> None:
        """Clip with no issues is valid."""
        cv = ClipValidation(metadata=self._make_meta())
        assert cv.is_valid is True
        assert cv.errors == []
        assert cv.warnings == []
        assert cv.needs_reencode is False

    def test_clip_with_error(self) -> None:
        """Clip with an error is invalid."""
        issue = ValidationIssue(
            code=IssueCode.RESOLUTION_BELOW_TARGET,
            severity=Severity.ERROR,
            message="Too small",
            field="resolution",
        )
        cv = ClipValidation(metadata=self._make_meta(), issues=[issue])
        assert cv.is_valid is False
        assert len(cv.errors) == 1
        assert len(cv.warnings) == 0

    def test_clip_with_warning_only(self) -> None:
        """Clip with only warnings is still valid."""
        issue = ValidationIssue(
            code=IssueCode.FPS_MISMATCH,
            severity=Severity.WARNING,
            message="Wrong fps",
            field="fps",
        )
        cv = ClipValidation(
            metadata=self._make_meta(),
            issues=[issue],
            needs_reencode=True,
        )
        assert cv.is_valid is True
        assert len(cv.errors) == 0
        assert len(cv.warnings) == 1

    def test_recommended_frame_count(self) -> None:
        """Recommended frame count is set when trimming needed."""
        cv = ClipValidation(
            metadata=self._make_meta(frame_count=83),
            recommended_frame_count=81,
        )
        assert cv.recommended_frame_count == 81


# ---------------------------------------------------------------------------
# SceneBoundary
# ---------------------------------------------------------------------------

class TestSceneBoundary:
    """Tests for SceneBoundary model."""

    def test_creation(self) -> None:
        """Create a scene boundary."""
        sb = SceneBoundary(
            frame_number=240,
            timecode=10.0,
            confidence=0.95,
        )
        assert sb.frame_number == 240
        assert sb.timecode == 10.0
        assert sb.confidence == 0.95


# ---------------------------------------------------------------------------
# ClipInfo
# ---------------------------------------------------------------------------

class TestClipInfo:
    """Tests for ClipInfo model."""

    def test_basic(self) -> None:
        """Create a clip info with required fields."""
        ci = ClipInfo(
            source=Path("/input/clip.mov"),
            output=Path("/output/clip.mp4"),
            frame_count=81,
            duration=5.0625,
            width=854,
            height=480,
            fps=16.0,
            was_reencoded=True,
        )
        assert ci.source == Path("/input/clip.mov")
        assert ci.output == Path("/output/clip.mp4")
        assert ci.trimmed_frames == 0
        assert ci.scene_index is None

    def test_with_trim_and_scene(self) -> None:
        """Clip info with trimming and scene index."""
        ci = ClipInfo(
            source=Path("/input/long_video.mp4"),
            output=Path("/output/scene_002.mp4"),
            frame_count=81,
            duration=5.0625,
            width=854,
            height=480,
            fps=16.0,
            was_reencoded=True,
            trimmed_frames=2,
            scene_index=2,
        )
        assert ci.trimmed_frames == 2
        assert ci.scene_index == 2


# ---------------------------------------------------------------------------
# ScanReport
# ---------------------------------------------------------------------------

class TestScanReport:
    """Tests for ScanReport model."""

    def _make_meta(self, name: str = "clip.mp4", **kwargs) -> VideoMetadata:
        defaults = dict(
            path=Path(f"/test/{name}"),
            width=854, height=480, fps=16.0,
            frame_count=81, duration=5.0625, codec="h264",
        )
        defaults.update(kwargs)
        return VideoMetadata(**defaults)

    def test_empty_report(self) -> None:
        """Empty report has zero counts."""
        report = ScanReport(directory=Path("/test"))
        assert report.total == 0
        assert report.valid == 0
        assert report.invalid == 0
        assert report.needs_reencode == 0
        assert report.issue_summary == {}

    def test_all_valid(self) -> None:
        """Report with all valid clips."""
        clips = [
            ClipValidation(metadata=self._make_meta("a.mp4")),
            ClipValidation(metadata=self._make_meta("b.mp4")),
        ]
        report = ScanReport(directory=Path("/test"), clips=clips)
        assert report.total == 2
        assert report.valid == 2
        assert report.invalid == 0

    def test_mixed_results(self) -> None:
        """Report with both valid and invalid clips."""
        error_issue = ValidationIssue(
            code=IssueCode.RESOLUTION_BELOW_TARGET,
            severity=Severity.ERROR,
            message="Too small",
            field="resolution",
        )
        warning_issue = ValidationIssue(
            code=IssueCode.FPS_MISMATCH,
            severity=Severity.WARNING,
            message="Wrong fps",
            field="fps",
        )
        clips = [
            ClipValidation(metadata=self._make_meta("good.mp4")),
            ClipValidation(metadata=self._make_meta("bad.mp4"), issues=[error_issue]),
            ClipValidation(
                metadata=self._make_meta("reencode.mp4"),
                issues=[warning_issue],
                needs_reencode=True,
            ),
        ]
        report = ScanReport(directory=Path("/test"), clips=clips)
        assert report.total == 3
        assert report.valid == 2  # warnings don't make invalid
        assert report.invalid == 1
        assert report.needs_reencode == 1

    def test_issue_summary(self) -> None:
        """Issue summary counts each code across all clips."""
        fps_issue = ValidationIssue(
            code=IssueCode.FPS_MISMATCH,
            severity=Severity.WARNING,
            message="Wrong fps",
            field="fps",
        )
        clips = [
            ClipValidation(metadata=self._make_meta("a.mp4"), issues=[fps_issue]),
            ClipValidation(metadata=self._make_meta("b.mp4"), issues=[fps_issue]),
        ]
        report = ScanReport(directory=Path("/test"), clips=clips)
        assert report.issue_summary[IssueCode.FPS_MISMATCH] == 2
