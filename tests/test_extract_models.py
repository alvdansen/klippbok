"""Tests for klippbok.video.extract_models — data types for image extraction.

Pure Python tests (no ffmpeg). Verifies model creation, defaults,
immutability, and report summary properties.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from klippbok.video.extract_models import (
    ExtractionConfig,
    ExtractionReport,
    ExtractionResult,
    ExtractionStrategy,
    ImageValidation,
)


# ---------------------------------------------------------------------------
# ExtractionStrategy enum
# ---------------------------------------------------------------------------

class TestExtractionStrategy:
    """Tests for the ExtractionStrategy enum."""

    def test_values(self) -> None:
        """All three strategies have the expected string values."""
        assert ExtractionStrategy.FIRST_FRAME == "first_frame"
        assert ExtractionStrategy.BEST_FRAME == "best_frame"
        assert ExtractionStrategy.USER_SELECTED == "user_selected"

    def test_from_string(self) -> None:
        """Can construct from string value."""
        assert ExtractionStrategy("first_frame") is ExtractionStrategy.FIRST_FRAME
        assert ExtractionStrategy("best_frame") is ExtractionStrategy.BEST_FRAME


# ---------------------------------------------------------------------------
# ExtractionConfig
# ---------------------------------------------------------------------------

class TestExtractionConfig:
    """Tests for ExtractionConfig defaults and validation."""

    def test_defaults(self) -> None:
        """Default config uses first_frame, 10 samples, no overwrite."""
        config = ExtractionConfig()
        assert config.strategy == ExtractionStrategy.FIRST_FRAME
        assert config.sample_count == 10
        assert config.overwrite is False

    def test_custom_values(self) -> None:
        """Can set custom strategy and sample count."""
        config = ExtractionConfig(
            strategy=ExtractionStrategy.BEST_FRAME,
            sample_count=20,
            overwrite=True,
        )
        assert config.strategy == ExtractionStrategy.BEST_FRAME
        assert config.sample_count == 20
        assert config.overwrite is True

    def test_frozen(self) -> None:
        """Config is immutable after creation."""
        config = ExtractionConfig()
        with pytest.raises(Exception):
            config.strategy = ExtractionStrategy.BEST_FRAME  # type: ignore[misc]

    def test_sample_count_minimum(self) -> None:
        """sample_count must be at least 2."""
        with pytest.raises(Exception):
            ExtractionConfig(sample_count=1)

    def test_strategy_from_string(self) -> None:
        """Can pass strategy as a string value."""
        config = ExtractionConfig(strategy="best_frame")  # type: ignore[arg-type]
        assert config.strategy == ExtractionStrategy.BEST_FRAME


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------

class TestExtractionResult:
    """Tests for ExtractionResult creation and defaults."""

    def test_success_result(self) -> None:
        """Successful extraction records all metadata."""
        result = ExtractionResult(
            source=Path("clips/clip_001.mp4"),
            output=Path("refs/clip_001.png"),
            frame_number=0,
            strategy=ExtractionStrategy.FIRST_FRAME,
            sharpness=250.5,
        )
        assert result.success is True
        assert result.error is None
        assert result.skipped is False
        assert result.source_type == "video"

    def test_failed_result(self) -> None:
        """Failed extraction has success=False and an error message."""
        result = ExtractionResult(
            source=Path("clips/bad.mp4"),
            success=False,
            error="ffmpeg returned non-zero exit code",
        )
        assert result.success is False
        assert result.output is None
        assert "ffmpeg" in result.error

    def test_skipped_result(self) -> None:
        """Skipped extraction (already exists) is marked accordingly."""
        result = ExtractionResult(
            source=Path("clips/clip_001.mp4"),
            output=Path("refs/clip_001.png"),
            skipped=True,
        )
        assert result.skipped is True
        assert result.success is True

    def test_image_source_type(self) -> None:
        """Image pass-through records source_type='image'."""
        result = ExtractionResult(
            source=Path("clips/still.jpg"),
            output=Path("refs/still.png"),
            source_type="image",
        )
        assert result.source_type == "image"

    def test_frozen(self) -> None:
        """Result is immutable."""
        result = ExtractionResult(source=Path("test.mp4"))
        with pytest.raises(Exception):
            result.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ImageValidation
# ---------------------------------------------------------------------------

class TestImageValidation:
    """Tests for ImageValidation model."""

    def test_sharp_image(self) -> None:
        """Sharp image has high sharpness, not blank, resolution ok."""
        val = ImageValidation(
            path=Path("test.png"),
            width=320,
            height=240,
            sharpness=350.0,
            is_blank=False,
            resolution_ok=True,
        )
        assert val.sharpness == 350.0
        assert val.is_blank is False
        assert val.resolution_ok is True

    def test_blank_image(self) -> None:
        """Blank image is flagged."""
        val = ImageValidation(
            path=Path("blank.png"),
            width=320,
            height=240,
            sharpness=1.2,
            is_blank=True,
            resolution_ok=True,
        )
        assert val.is_blank is True

    def test_resolution_mismatch(self) -> None:
        """Resolution mismatch is flagged."""
        val = ImageValidation(
            path=Path("test.png"),
            width=640,
            height=480,
            sharpness=200.0,
            is_blank=False,
            resolution_ok=False,
            expected_width=320,
            expected_height=240,
        )
        assert val.resolution_ok is False
        assert val.expected_width == 320

    def test_frozen(self) -> None:
        """Validation result is immutable."""
        val = ImageValidation(
            path=Path("test.png"),
            width=320, height=240,
            sharpness=100.0, is_blank=False, resolution_ok=True,
        )
        with pytest.raises(Exception):
            val.sharpness = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ExtractionReport
# ---------------------------------------------------------------------------

class TestExtractionReport:
    """Tests for ExtractionReport summary properties."""

    def _make_report(self) -> ExtractionReport:
        """Create a report with mixed results for testing summaries."""
        return ExtractionReport(results=[
            # Successful video extraction
            ExtractionResult(
                source=Path("clip_001.mp4"),
                output=Path("clip_001.png"),
                success=True,
                source_type="video",
            ),
            # Successful image pass-through
            ExtractionResult(
                source=Path("still_002.jpg"),
                output=Path("still_002.png"),
                success=True,
                source_type="image",
            ),
            # Failed extraction
            ExtractionResult(
                source=Path("bad_003.mp4"),
                success=False,
                error="corrupted",
                source_type="video",
            ),
            # Skipped (already exists)
            ExtractionResult(
                source=Path("clip_004.mp4"),
                output=Path("clip_004.png"),
                skipped=True,
                source_type="video",
            ),
        ])

    def test_total(self) -> None:
        """Total counts all results."""
        report = self._make_report()
        assert report.total == 4

    def test_succeeded(self) -> None:
        """Succeeded counts successful, non-skipped results."""
        report = self._make_report()
        assert report.succeeded == 2

    def test_failed(self) -> None:
        """Failed counts only failures."""
        report = self._make_report()
        assert report.failed == 1

    def test_skipped(self) -> None:
        """Skipped counts only skipped results."""
        report = self._make_report()
        assert report.skipped == 1

    def test_videos(self) -> None:
        """Videos counts video source types."""
        report = self._make_report()
        assert report.videos == 3

    def test_images(self) -> None:
        """Images counts image source types."""
        report = self._make_report()
        assert report.images == 1

    def test_empty_report(self) -> None:
        """Empty report has all zeroes."""
        report = ExtractionReport()
        assert report.total == 0
        assert report.succeeded == 0
        assert report.failed == 0
        assert report.skipped == 0
