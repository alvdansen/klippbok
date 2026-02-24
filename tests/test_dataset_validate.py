"""Tests for klippbok.dataset.validate — core validation engine.

Mix of mock SamplePairs (no I/O) and real files in tmp_path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.conftest import requires_ffmpeg

from klippbok.config.data_schema import (
    ControlsConfig,
    KlippbokDataConfig,
    ImagesControlConfig,
    MotionQualityConfig,
    QualityConfig,
    ReferenceImageConfig,
    TextControlConfig,
)
from klippbok.dataset.discover import discover_dataset
from klippbok.dataset.models import (
    DatasetValidation,
    SamplePair,
    StructureType,
)
from klippbok.dataset.validate import (
    validate_all,
    validate_dataset,
    validate_sample,
)
from klippbok.video.models import IssueCode, Severity, ValidationIssue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _touch(path: Path, content: bytes = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _save_image(path: Path, pixels: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), pixels)
    return path


def _make_textured_image(path: Path, size: int = 64) -> Path:
    """Create an image with enough texture to not be blank."""
    rng = np.random.RandomState(42)
    img = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return _save_image(path, img)


def _make_blank_image(path: Path, size: int = 64) -> Path:
    """Create a solid-color image (will be detected as blank)."""
    img = np.full((size, size, 3), 128, dtype=np.uint8)
    return _save_image(path, img)


def _default_config(
    caption_required: bool = True,
    reference_required: bool = False,
    blur_threshold: float | None = None,
    exposure_range: tuple[float, float] | None = None,
    check_duplicates: bool = False,
    min_motion: float | None = None,
    max_motion: float | None = None,
    path: str = ".",
) -> KlippbokDataConfig:
    return KlippbokDataConfig(
        datasets=[{"path": path}],
        controls=ControlsConfig(
            text=TextControlConfig(required=caption_required),
            images=ImagesControlConfig(
                reference=ReferenceImageConfig(required=reference_required),
            ),
        ),
        quality=QualityConfig(
            blur_threshold=blur_threshold,
            exposure_range=exposure_range,
            check_duplicates=check_duplicates,
            motion=MotionQualityConfig(
                min_intensity=min_motion,
                max_intensity=max_motion,
            ),
        ),
    )


def _make_flat_dataset(tmp_path: Path, stems: list[str], with_refs: bool = True) -> Path:
    for stem in stems:
        _touch(tmp_path / f"{stem}.mp4")
        _touch(tmp_path / f"{stem}.txt", f"A caption for {stem}.".encode())
        if with_refs:
            _make_textured_image(tmp_path / f"{stem}.png")
    return tmp_path


# ---------------------------------------------------------------------------
# validate_sample tests
# ---------------------------------------------------------------------------

class TestValidateSample:
    def test_perfect_sample(self, tmp_path: Path):
        """A sample with valid caption and textured reference has no issues."""
        _touch(tmp_path / "clip.mp4")
        _touch(tmp_path / "clip.txt", b"A girl walks through a garden.")
        _make_textured_image(tmp_path / "clip.png")

        sample = SamplePair(
            stem="clip",
            target=tmp_path / "clip.mp4",
            caption=tmp_path / "clip.txt",
            reference=tmp_path / "clip.png",
        )
        config = _default_config()
        result = validate_sample(sample, config)
        assert result.is_valid

    def test_empty_caption(self, tmp_path: Path):
        _touch(tmp_path / "clip.mp4")
        _touch(tmp_path / "clip.txt", b"")

        sample = SamplePair(
            stem="clip",
            target=tmp_path / "clip.mp4",
            caption=tmp_path / "clip.txt",
        )
        config = _default_config()
        result = validate_sample(sample, config)
        assert any(i.code == IssueCode.CAPTION_EMPTY for i in result.issues)

    def test_whitespace_only_caption(self, tmp_path: Path):
        _touch(tmp_path / "clip.mp4")
        _touch(tmp_path / "clip.txt", b"   \n\t  ")

        sample = SamplePair(
            stem="clip",
            target=tmp_path / "clip.mp4",
            caption=tmp_path / "clip.txt",
        )
        config = _default_config()
        result = validate_sample(sample, config)
        assert any(i.code == IssueCode.CAPTION_EMPTY for i in result.issues)

    def test_caption_too_long(self, tmp_path: Path):
        # ~500 words at 1.3 tokens/word = ~650 estimated tokens > 512
        long_text = " ".join(["word"] * 500)
        _touch(tmp_path / "clip.mp4")
        _touch(tmp_path / "clip.txt", long_text.encode())

        sample = SamplePair(
            stem="clip",
            target=tmp_path / "clip.mp4",
            caption=tmp_path / "clip.txt",
        )
        config = _default_config()
        result = validate_sample(sample, config)
        assert any(i.code == IssueCode.CAPTION_TOO_LONG for i in result.issues)

    def test_caption_within_limit(self, tmp_path: Path):
        short_text = "A girl walks through a garden with flowers."
        _touch(tmp_path / "clip.mp4")
        _touch(tmp_path / "clip.txt", short_text.encode())

        sample = SamplePair(
            stem="clip",
            target=tmp_path / "clip.mp4",
            caption=tmp_path / "clip.txt",
        )
        config = _default_config()
        result = validate_sample(sample, config)
        assert not any(i.code == IssueCode.CAPTION_TOO_LONG for i in result.issues)

    def test_blank_reference(self, tmp_path: Path):
        _touch(tmp_path / "clip.mp4")
        _touch(tmp_path / "clip.txt", b"Caption")
        _make_blank_image(tmp_path / "clip.png")

        sample = SamplePair(
            stem="clip",
            target=tmp_path / "clip.mp4",
            caption=tmp_path / "clip.txt",
            reference=tmp_path / "clip.png",
        )
        config = _default_config()
        result = validate_sample(sample, config)
        assert any(i.code == IssueCode.REFERENCE_BLANK for i in result.issues)

    def test_textured_reference_ok(self, tmp_path: Path):
        _touch(tmp_path / "clip.mp4")
        _touch(tmp_path / "clip.txt", b"Caption")
        _make_textured_image(tmp_path / "clip.png")

        sample = SamplePair(
            stem="clip",
            target=tmp_path / "clip.mp4",
            caption=tmp_path / "clip.txt",
            reference=tmp_path / "clip.png",
        )
        config = _default_config()
        result = validate_sample(sample, config)
        assert not any(i.code == IssueCode.REFERENCE_BLANK for i in result.issues)

    def test_blur_threshold(self, tmp_path: Path):
        """Reference below blur threshold triggers warning."""
        _touch(tmp_path / "clip.mp4")
        _touch(tmp_path / "clip.txt", b"Caption")
        # Create a slightly blurry but not blank image
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        # Add slight gradient so it's not blank
        img[:, :, 0] = np.tile(np.linspace(100, 150, 64, dtype=np.uint8), (64, 1))
        _save_image(tmp_path / "clip.png", img)

        sample = SamplePair(
            stem="clip",
            target=tmp_path / "clip.mp4",
            caption=tmp_path / "clip.txt",
            reference=tmp_path / "clip.png",
        )
        # Set a very high threshold to trigger the warning
        config = _default_config(blur_threshold=10000.0)
        result = validate_sample(sample, config)
        assert any(i.code == IssueCode.BLUR_BELOW_THRESHOLD for i in result.issues)

    def test_exposure_out_of_range_dark(self, tmp_path: Path):
        _touch(tmp_path / "clip.mp4")
        _touch(tmp_path / "clip.txt", b"Caption")
        # Very dark image
        img = np.full((64, 64, 3), 5, dtype=np.uint8)
        # Add some texture so it's not blank
        rng = np.random.RandomState(42)
        img += rng.randint(0, 10, (64, 64, 3), dtype=np.uint8)
        _save_image(tmp_path / "clip.png", img)

        sample = SamplePair(
            stem="clip",
            target=tmp_path / "clip.mp4",
            caption=tmp_path / "clip.txt",
            reference=tmp_path / "clip.png",
        )
        config = _default_config(exposure_range=(0.2, 0.8))
        result = validate_sample(sample, config)
        assert any(i.code == IssueCode.EXPOSURE_OUT_OF_RANGE for i in result.issues)

    def test_exposure_out_of_range_bright(self, tmp_path: Path):
        _touch(tmp_path / "clip.mp4")
        _touch(tmp_path / "clip.txt", b"Caption")
        # Very bright image
        img = np.full((64, 64, 3), 250, dtype=np.uint8)
        rng = np.random.RandomState(42)
        img = (img.astype(int) + rng.randint(-5, 5, (64, 64, 3))).clip(0, 255).astype(np.uint8)
        _save_image(tmp_path / "clip.png", img)

        sample = SamplePair(
            stem="clip",
            target=tmp_path / "clip.mp4",
            caption=tmp_path / "clip.txt",
            reference=tmp_path / "clip.png",
        )
        config = _default_config(exposure_range=(0.2, 0.8))
        result = validate_sample(sample, config)
        assert any(i.code == IssueCode.EXPOSURE_OUT_OF_RANGE for i in result.issues)

    def test_exposure_in_range(self, tmp_path: Path):
        _touch(tmp_path / "clip.mp4")
        _touch(tmp_path / "clip.txt", b"Caption")
        _make_textured_image(tmp_path / "clip.png")

        sample = SamplePair(
            stem="clip",
            target=tmp_path / "clip.mp4",
            caption=tmp_path / "clip.txt",
            reference=tmp_path / "clip.png",
        )
        config = _default_config(exposure_range=(0.0, 1.0))
        result = validate_sample(sample, config)
        assert not any(i.code == IssueCode.EXPOSURE_OUT_OF_RANGE for i in result.issues)

    def test_preserves_existing_issues(self, tmp_path: Path):
        """Issues from discovery phase are preserved."""
        _touch(tmp_path / "clip.mp4")
        existing_issue = ValidationIssue(
            code=IssueCode.CAPTION_MISSING,
            severity=Severity.ERROR,
            message="No caption",
            field="caption",
        )
        sample = SamplePair(
            stem="clip",
            target=tmp_path / "clip.mp4",
            issues=[existing_issue],
        )
        config = _default_config()
        result = validate_sample(sample, config)
        assert any(i.code == IssueCode.CAPTION_MISSING for i in result.issues)

    def test_corrupted_reference(self, tmp_path: Path):
        _touch(tmp_path / "clip.mp4")
        _touch(tmp_path / "clip.txt", b"Caption")
        _touch(tmp_path / "clip.png", b"not a real image")

        sample = SamplePair(
            stem="clip",
            target=tmp_path / "clip.mp4",
            caption=tmp_path / "clip.txt",
            reference=tmp_path / "clip.png",
        )
        config = _default_config()
        result = validate_sample(sample, config)
        assert any(i.code == IssueCode.FILE_CORRUPTED for i in result.issues)

    def test_no_reference_no_image_checks(self, tmp_path: Path):
        """No reference means image quality checks are skipped."""
        _touch(tmp_path / "clip.mp4")
        _touch(tmp_path / "clip.txt", b"Caption")

        sample = SamplePair(
            stem="clip",
            target=tmp_path / "clip.mp4",
            caption=tmp_path / "clip.txt",
        )
        config = _default_config(blur_threshold=100.0, exposure_range=(0.2, 0.8))
        result = validate_sample(sample, config)
        assert not any(i.code == IssueCode.BLUR_BELOW_THRESHOLD for i in result.issues)
        assert not any(i.code == IssueCode.EXPOSURE_OUT_OF_RANGE for i in result.issues)


# ---------------------------------------------------------------------------
# validate_dataset tests
# ---------------------------------------------------------------------------

class TestValidateDataset:
    def test_valid_flat_dataset(self, tmp_path: Path):
        _make_flat_dataset(tmp_path, ["a", "b", "c"])
        config = _default_config()
        discovered = discover_dataset(tmp_path, config)
        result = validate_dataset(discovered, config)
        assert result.is_valid

    def test_invalid_samples_counted(self, tmp_path: Path):
        _touch(tmp_path / "a.mp4")
        _touch(tmp_path / "b.mp4")
        # Only one caption — b is missing
        _touch(tmp_path / "a.txt", b"Caption for a")
        config = _default_config(caption_required=True)
        discovered = discover_dataset(tmp_path, config)
        result = validate_dataset(discovered, config)
        assert result.invalid_samples == 1

    def test_empty_dataset_issue(self, tmp_path: Path):
        config = _default_config()
        discovered = discover_dataset(tmp_path, config)
        result = validate_dataset(discovered, config)
        assert any(i.code == IssueCode.DATASET_EMPTY for i in result.dataset_issues)

    def test_duplicate_detection_on(self, tmp_path: Path):
        """With check_duplicates=True, identical references are flagged."""
        # Create two clips with identical reference images
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        _touch(tmp_path / "a.mp4")
        _touch(tmp_path / "a.txt", b"Caption a")
        cv2.imwrite(str(tmp_path / "a.png"), img)
        _touch(tmp_path / "b.mp4")
        _touch(tmp_path / "b.txt", b"Caption b")
        cv2.imwrite(str(tmp_path / "b.png"), img)

        config = _default_config(check_duplicates=True)
        discovered = discover_dataset(tmp_path, config)
        result = validate_dataset(discovered, config)
        assert any(i.code == IssueCode.DUPLICATE_DETECTED for i in result.dataset_issues)

    def test_duplicate_detection_off(self, tmp_path: Path):
        """With check_duplicates=False (default), no duplicate check."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        _touch(tmp_path / "a.mp4")
        _touch(tmp_path / "a.txt", b"Caption a")
        cv2.imwrite(str(tmp_path / "a.png"), img)
        _touch(tmp_path / "b.mp4")
        _touch(tmp_path / "b.txt", b"Caption b")
        cv2.imwrite(str(tmp_path / "b.png"), img)

        config = _default_config(check_duplicates=False)
        discovered = discover_dataset(tmp_path, config)
        result = validate_dataset(discovered, config)
        assert not any(i.code == IssueCode.DUPLICATE_DETECTED for i in result.dataset_issues)

    def test_klippbok_structure(self, tmp_path: Path):
        for stem in ["clip_001", "clip_002"]:
            _touch(tmp_path / "training" / "targets" / f"{stem}.mp4")
            _touch(
                tmp_path / "training" / "signals" / "captions" / f"{stem}.txt",
                b"Caption",
            )
            _make_textured_image(
                tmp_path / "training" / "signals" / "references" / f"{stem}.png",
            )
        config = _default_config()
        discovered = discover_dataset(tmp_path, config)
        result = validate_dataset(discovered, config)
        assert result.structure == StructureType.DIMLJUS
        assert result.total_samples == 2
        assert result.is_valid

    @requires_ffmpeg
    def test_motion_below_min(self, tmp_path: Path):
        """Static video triggers MOTION_BELOW_MIN."""
        out = tmp_path / "static.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=c=blue:size=64x64:rate=16:d=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out),
        ], capture_output=True, timeout=30)
        _touch(tmp_path / "static.txt", b"Static clip")

        config = _default_config(caption_required=True, min_motion=5.0)
        discovered = discover_dataset(tmp_path, config)
        result = validate_dataset(discovered, config)
        all_issues = result.all_issues
        assert any(i.code == IssueCode.MOTION_BELOW_MIN for i in all_issues)

    @requires_ffmpeg
    def test_motion_above_max(self, tmp_path: Path):
        """Dynamic video triggers MOTION_ABOVE_MAX with low threshold."""
        out = tmp_path / "dynamic.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc2=size=64x64:rate=16:d=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out),
        ], capture_output=True, timeout=30)
        _touch(tmp_path / "dynamic.txt", b"Dynamic clip")

        config = _default_config(caption_required=True, max_motion=0.001)
        discovered = discover_dataset(tmp_path, config)
        result = validate_dataset(discovered, config)
        all_issues = result.all_issues
        assert any(i.code == IssueCode.MOTION_ABOVE_MAX for i in all_issues)


# ---------------------------------------------------------------------------
# validate_all tests
# ---------------------------------------------------------------------------

class TestValidateAll:
    def test_single_source(self, tmp_path: Path):
        ds = tmp_path / "ds1"
        _make_flat_dataset(ds, ["a", "b"])
        config = KlippbokDataConfig(
            datasets=[{"path": str(ds)}],
        )
        report = validate_all(config, config_dir=tmp_path)
        assert report.total_sources == 1
        assert report.total_samples == 2
        assert report.is_valid

    def test_multi_source(self, tmp_path: Path):
        ds1 = tmp_path / "ds1"
        ds2 = tmp_path / "ds2"
        _make_flat_dataset(ds1, ["a"])
        _make_flat_dataset(ds2, ["b", "c"])
        config = KlippbokDataConfig(
            datasets=[{"path": str(ds1)}, {"path": str(ds2)}],
        )
        report = validate_all(config, config_dir=tmp_path)
        assert report.total_sources == 2
        assert report.total_samples == 3

    def test_cross_dataset_duplicates(self, tmp_path: Path):
        """Identical images across sources flagged as cross-dataset duplicates."""
        ds1 = tmp_path / "ds1"
        ds2 = tmp_path / "ds2"
        ds1.mkdir()
        ds2.mkdir()

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        _touch(ds1 / "clip.mp4")
        _touch(ds1 / "clip.txt", b"Caption 1")
        cv2.imwrite(str(ds1 / "clip.png"), img)
        _touch(ds2 / "clip.mp4")
        _touch(ds2 / "clip.txt", b"Caption 2")
        cv2.imwrite(str(ds2 / "clip.png"), img)

        config = KlippbokDataConfig(
            datasets=[{"path": str(ds1)}, {"path": str(ds2)}],
            quality=QualityConfig(check_duplicates=True),
        )
        report = validate_all(config, config_dir=tmp_path)
        assert any(i.code == IssueCode.DUPLICATE_DETECTED for i in report.cross_dataset_issues)

    def test_issue_summary(self, tmp_path: Path):
        ds = tmp_path / "ds"
        ds.mkdir()
        _touch(ds / "a.mp4")
        _touch(ds / "b.mp4")
        # Both missing captions
        config = KlippbokDataConfig(
            datasets=[{"path": str(ds)}],
            controls=ControlsConfig(
                text=TextControlConfig(required=True),
            ),
        )
        report = validate_all(config, config_dir=tmp_path)
        summary = report.issue_summary
        assert summary.get(IssueCode.CAPTION_MISSING, 0) == 2


# ---------------------------------------------------------------------------
# Integration with sample character fixture
# ---------------------------------------------------------------------------

class TestSampleCharacterFixtureIntegration:
    """Test against the sample character 5-clip fixture if it exists."""

    @pytest.fixture
    def sample_path(self) -> Path:
        p = Path(__file__).parent / "fixtures" / "sample_character"
        if not p.is_dir():
            pytest.skip("Sample character fixture not available")
        return p

    def test_sample_discovery(self, sample_path: Path):
        """Sample character fixture should be detected as flat structure."""
        config = _default_config(caption_required=False)
        discovered = discover_dataset(sample_path, config)
        assert discovered.structure == StructureType.FLAT
        # Should find .mov files as targets
        assert discovered.total_samples > 0
