"""Tests for klippbok.video.extract — reference image extraction engine.

Most tests require ffmpeg (uses the tiny_video fixture from conftest.py).
A few tests for image pass-through only need OpenCV (no ffmpeg).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from klippbok.video.extract_models import (
    ExtractionConfig,
    ExtractionStrategy,
)
from tests.conftest import requires_ffmpeg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_test_image(
    path: Path,
    width: int = 320,
    height: int = 240,
    pattern: str = "checkerboard",
) -> None:
    """Write a test image with known properties."""
    if pattern == "checkerboard":
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(0, height, 8):
            for x in range(0, width, 8):
                if ((x // 8) + (y // 8)) % 2 == 0:
                    img[y:y+8, x:x+8] = 255
    elif pattern == "blank":
        img = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# extract_first_frame
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestExtractFirstFrame:
    """Tests for extract_first_frame() — requires ffmpeg."""

    def test_basic_extraction(self, tiny_video: Path, tmp_path: Path) -> None:
        """Extracts frame 0 as a PNG file."""
        from klippbok.video.extract import extract_first_frame

        output = tmp_path / "ref.png"
        result = extract_first_frame(tiny_video, output)

        assert result.success is True
        assert result.frame_number == 0
        assert result.strategy == ExtractionStrategy.FIRST_FRAME
        assert result.source_type == "video"
        assert output.exists()
        assert output.suffix == ".png"

    def test_creates_output_dir(self, tiny_video: Path, tmp_path: Path) -> None:
        """Creates output directory if it doesn't exist."""
        from klippbok.video.extract import extract_first_frame

        output = tmp_path / "nested" / "deep" / "ref.png"
        result = extract_first_frame(tiny_video, output)

        assert result.success is True
        assert output.exists()

    def test_has_sharpness(self, tiny_video: Path, tmp_path: Path) -> None:
        """Extracted frame gets a sharpness score."""
        from klippbok.video.extract import extract_first_frame

        output = tmp_path / "ref.png"
        result = extract_first_frame(tiny_video, output)

        assert result.sharpness is not None
        assert result.sharpness > 0.0

    def test_output_is_valid_image(self, tiny_video: Path, tmp_path: Path) -> None:
        """Extracted PNG is a valid, readable image."""
        from klippbok.video.extract import extract_first_frame

        output = tmp_path / "ref.png"
        extract_first_frame(tiny_video, output)

        img = cv2.imread(str(output))
        assert img is not None
        assert img.shape[0] > 0  # height
        assert img.shape[1] > 0  # width


# ---------------------------------------------------------------------------
# extract_frame_at
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestExtractFrameAt:
    """Tests for extract_frame_at() — specific frame extraction."""

    def test_by_frame_number(self, tiny_video: Path, tmp_path: Path) -> None:
        """Extract a specific frame by number."""
        from klippbok.video.extract import extract_frame_at

        output = tmp_path / "frame5.png"
        result = extract_frame_at(tiny_video, output, frame_number=5)

        assert result.success is True
        assert result.frame_number == 5
        assert result.strategy == ExtractionStrategy.USER_SELECTED
        assert output.exists()

    def test_by_timestamp(self, tiny_video: Path, tmp_path: Path) -> None:
        """Extract a frame by timestamp."""
        from klippbok.video.extract import extract_frame_at

        output = tmp_path / "at_half.png"
        result = extract_frame_at(tiny_video, output, timestamp=0.5)

        assert result.success is True
        assert result.strategy == ExtractionStrategy.USER_SELECTED
        assert output.exists()

    def test_frame_zero(self, tiny_video: Path, tmp_path: Path) -> None:
        """Frame 0 works (edge case: first frame)."""
        from klippbok.video.extract import extract_frame_at

        output = tmp_path / "frame0.png"
        result = extract_frame_at(tiny_video, output, frame_number=0)

        assert result.success is True
        assert result.frame_number == 0

    def test_neither_frame_nor_timestamp_raises(self, tiny_video: Path, tmp_path: Path) -> None:
        """Raises ValueError if neither frame_number nor timestamp given."""
        from klippbok.video.extract import extract_frame_at

        with pytest.raises(ValueError, match="exactly one"):
            extract_frame_at(tiny_video, tmp_path / "out.png")

    def test_both_frame_and_timestamp_raises(self, tiny_video: Path, tmp_path: Path) -> None:
        """Raises ValueError if both frame_number and timestamp given."""
        from klippbok.video.extract import extract_frame_at

        with pytest.raises(ValueError, match="exactly one"):
            extract_frame_at(
                tiny_video, tmp_path / "out.png",
                frame_number=5, timestamp=0.5,
            )


# ---------------------------------------------------------------------------
# extract_best_frame
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestExtractBestFrame:
    """Tests for extract_best_frame() — sharpness-based selection."""

    def test_produces_output(self, tiny_video: Path, tmp_path: Path) -> None:
        """Extracts a frame and produces a PNG."""
        from klippbok.video.extract import extract_best_frame

        output = tmp_path / "best.png"
        result = extract_best_frame(tiny_video, output, sample_count=5)

        assert result.success is True
        assert result.strategy == ExtractionStrategy.BEST_FRAME
        assert result.sharpness is not None
        assert output.exists()

    def test_has_sharpness_score(self, tiny_video: Path, tmp_path: Path) -> None:
        """Result includes the sharpness of the picked frame."""
        from klippbok.video.extract import extract_best_frame

        output = tmp_path / "best.png"
        result = extract_best_frame(tiny_video, output, sample_count=5)

        assert result.sharpness is not None
        assert result.sharpness > 0.0

    def test_cleans_up_candidates(self, tiny_video: Path, tmp_path: Path) -> None:
        """Candidate frames are cleaned up after selection."""
        from klippbok.video.extract import extract_best_frame

        output = tmp_path / "out" / "best.png"
        extract_best_frame(tiny_video, output, sample_count=5)

        # The candidates directory should be cleaned up
        candidates_dirs = list((tmp_path / "out").glob("_candidates_*"))
        assert len(candidates_dirs) == 0


# ---------------------------------------------------------------------------
# copy_image_as_reference
# ---------------------------------------------------------------------------

class TestCopyImageAsReference:
    """Tests for copy_image_as_reference() — no ffmpeg needed."""

    def test_png_copy(self, tmp_path: Path) -> None:
        """PNG image is copied to output."""
        from klippbok.video.extract import copy_image_as_reference

        src = tmp_path / "source.png"
        _write_test_image(src)
        output = tmp_path / "ref.png"

        result = copy_image_as_reference(src, output)

        assert result.success is True
        assert result.source_type == "image"
        assert result.frame_number is None
        assert result.strategy is None
        assert output.exists()

    def test_jpg_to_png_conversion(self, tmp_path: Path) -> None:
        """JPG image is converted to PNG output."""
        from klippbok.video.extract import copy_image_as_reference

        src = tmp_path / "source.jpg"
        _write_test_image(src)
        output = tmp_path / "ref.png"

        result = copy_image_as_reference(src, output)

        assert result.success is True
        assert output.exists()
        # Verify it's actually a PNG by reading back
        img = cv2.imread(str(output))
        assert img is not None

    def test_has_sharpness(self, tmp_path: Path) -> None:
        """Copied image gets a sharpness score."""
        from klippbok.video.extract import copy_image_as_reference

        src = tmp_path / "source.png"
        _write_test_image(src)
        output = tmp_path / "ref.png"

        result = copy_image_as_reference(src, output)
        assert result.sharpness is not None
        assert result.sharpness > 0.0

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        """Creates output directory if needed."""
        from klippbok.video.extract import copy_image_as_reference

        src = tmp_path / "source.png"
        _write_test_image(src)
        output = tmp_path / "nested" / "ref.png"

        result = copy_image_as_reference(src, output)
        assert result.success is True
        assert output.exists()

    def test_source_not_found(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing source."""
        from klippbok.video.extract import copy_image_as_reference

        with pytest.raises(FileNotFoundError):
            copy_image_as_reference(tmp_path / "nope.png", tmp_path / "out.png")


# ---------------------------------------------------------------------------
# extract_reference_image (unified dispatch)
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestExtractReferenceImage:
    """Tests for extract_reference_image() — unified dispatch."""

    def test_video_first_frame(self, tiny_video: Path, tmp_path: Path) -> None:
        """Video file with first_frame strategy."""
        from klippbok.video.extract import extract_reference_image

        config = ExtractionConfig(strategy=ExtractionStrategy.FIRST_FRAME)
        output = tmp_path / "ref.png"
        result = extract_reference_image(tiny_video, output, config)

        assert result.success is True
        assert result.source_type == "video"
        assert result.strategy == ExtractionStrategy.FIRST_FRAME

    def test_video_best_frame(self, tiny_video: Path, tmp_path: Path) -> None:
        """Video file with best_frame strategy."""
        from klippbok.video.extract import extract_reference_image

        config = ExtractionConfig(strategy=ExtractionStrategy.BEST_FRAME, sample_count=3)
        output = tmp_path / "ref.png"
        result = extract_reference_image(tiny_video, output, config)

        assert result.success is True
        assert result.strategy == ExtractionStrategy.BEST_FRAME

    def test_image_passthrough(self, tmp_path: Path) -> None:
        """Image file is copied regardless of strategy setting."""
        from klippbok.video.extract import extract_reference_image

        src = tmp_path / "still.jpg"
        _write_test_image(src)
        output = tmp_path / "ref.png"

        config = ExtractionConfig(strategy=ExtractionStrategy.FIRST_FRAME)
        result = extract_reference_image(src, output, config)

        assert result.success is True
        assert result.source_type == "image"

    def test_skip_existing(self, tiny_video: Path, tmp_path: Path) -> None:
        """Skips extraction when output exists and overwrite=False."""
        from klippbok.video.extract import extract_reference_image

        output = tmp_path / "ref.png"
        # Create a dummy output file
        output.write_text("existing")

        config = ExtractionConfig(overwrite=False)
        result = extract_reference_image(tiny_video, output, config)

        assert result.skipped is True
        assert result.success is True
        # Original file should be untouched
        assert output.read_text() == "existing"

    def test_overwrite_existing(self, tiny_video: Path, tmp_path: Path) -> None:
        """Overwrites when overwrite=True."""
        from klippbok.video.extract import extract_reference_image

        output = tmp_path / "ref.png"
        output.write_text("old")

        config = ExtractionConfig(overwrite=True)
        result = extract_reference_image(tiny_video, output, config)

        assert result.success is True
        assert result.skipped is False
        # File should be a real image now, not "old"
        img = cv2.imread(str(output))
        assert img is not None

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        """Unsupported file extension produces a failed result."""
        from klippbok.video.extract import extract_reference_image

        src = tmp_path / "data.csv"
        src.write_text("a,b,c")
        output = tmp_path / "ref.png"

        result = extract_reference_image(src, output)
        assert result.success is False
        assert "Unsupported file type" in result.error


# ---------------------------------------------------------------------------
# extract_directory (batch)
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestExtractDirectory:
    """Tests for extract_directory() — batch extraction."""

    def test_video_directory(self, tiny_video: Path, tmp_path: Path) -> None:
        """Extracts reference images for all videos in a directory."""
        from klippbok.video.extract import extract_directory

        # Set up a source directory with a copy of the tiny video
        src_dir = tmp_path / "clips"
        src_dir.mkdir()
        import shutil
        shutil.copy2(str(tiny_video), str(src_dir / "clip_001.mp4"))
        shutil.copy2(str(tiny_video), str(src_dir / "clip_002.mp4"))

        out_dir = tmp_path / "refs"
        report = extract_directory(src_dir, out_dir)

        assert report.total == 2
        assert report.succeeded == 2
        assert report.failed == 0
        assert (out_dir / "clip_001.png").exists()
        assert (out_dir / "clip_002.png").exists()

    def test_mixed_directory(self, tiny_video: Path, tmp_path: Path) -> None:
        """Handles mixed video + image directories."""
        from klippbok.video.extract import extract_directory

        src_dir = tmp_path / "clips"
        src_dir.mkdir()
        import shutil
        shutil.copy2(str(tiny_video), str(src_dir / "clip_001.mp4"))
        _write_test_image(src_dir / "still_002.jpg")

        out_dir = tmp_path / "refs"
        report = extract_directory(src_dir, out_dir)

        assert report.total == 2
        assert report.succeeded == 2
        assert report.videos == 1
        assert report.images == 1
        assert (out_dir / "clip_001.png").exists()
        assert (out_dir / "still_002.png").exists()

    def test_skip_existing(self, tiny_video: Path, tmp_path: Path) -> None:
        """Skips files that already have output PNGs."""
        from klippbok.video.extract import extract_directory

        src_dir = tmp_path / "clips"
        src_dir.mkdir()
        import shutil
        shutil.copy2(str(tiny_video), str(src_dir / "clip_001.mp4"))

        out_dir = tmp_path / "refs"
        out_dir.mkdir()
        (out_dir / "clip_001.png").write_text("existing")

        config = ExtractionConfig(overwrite=False)
        report = extract_directory(src_dir, out_dir, config)

        assert report.total == 1
        assert report.skipped == 1
        assert report.succeeded == 0

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty source directory returns empty report."""
        from klippbok.video.extract import extract_directory

        src_dir = tmp_path / "empty"
        src_dir.mkdir()
        out_dir = tmp_path / "refs"

        report = extract_directory(src_dir, out_dir)
        assert report.total == 0

    def test_manifest_written(self, tiny_video: Path, tmp_path: Path) -> None:
        """Writes reference_images.json manifest."""
        from klippbok.video.extract import extract_directory

        src_dir = tmp_path / "clips"
        src_dir.mkdir()
        import shutil
        shutil.copy2(str(tiny_video), str(src_dir / "clip_001.mp4"))

        out_dir = tmp_path / "refs"
        extract_directory(src_dir, out_dir)

        manifest_path = out_dir / "reference_images.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert len(manifest) == 1
        assert manifest[0]["source_type"] == "video"
        assert "sharpness" in manifest[0]

    def test_manifest_source_types(self, tiny_video: Path, tmp_path: Path) -> None:
        """Manifest records correct source_type for videos and images."""
        from klippbok.video.extract import extract_directory

        src_dir = tmp_path / "clips"
        src_dir.mkdir()
        import shutil
        shutil.copy2(str(tiny_video), str(src_dir / "clip_001.mp4"))
        _write_test_image(src_dir / "still_002.jpg")

        out_dir = tmp_path / "refs"
        extract_directory(src_dir, out_dir)

        manifest = json.loads((out_dir / "reference_images.json").read_text())
        types = {entry["source_type"] for entry in manifest}
        assert types == {"video", "image"}

    def test_best_frame_strategy(self, tiny_video: Path, tmp_path: Path) -> None:
        """Directory extraction with best_frame strategy."""
        from klippbok.video.extract import extract_directory

        src_dir = tmp_path / "clips"
        src_dir.mkdir()
        import shutil
        shutil.copy2(str(tiny_video), str(src_dir / "clip_001.mp4"))

        out_dir = tmp_path / "refs"
        config = ExtractionConfig(strategy=ExtractionStrategy.BEST_FRAME, sample_count=3)
        report = extract_directory(src_dir, out_dir, config)

        assert report.succeeded == 1
        assert (out_dir / "clip_001.png").exists()


# ---------------------------------------------------------------------------
# generate_selection_template
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestGenerateSelectionTemplate:
    """Tests for generate_selection_template()."""

    def test_creates_template(self, tiny_video: Path, tmp_path: Path) -> None:
        """Generates a JSON template with video entries."""
        from klippbok.video.extract import generate_selection_template

        src_dir = tmp_path / "clips"
        src_dir.mkdir()
        import shutil
        shutil.copy2(str(tiny_video), str(src_dir / "clip_001.mp4"))
        shutil.copy2(str(tiny_video), str(src_dir / "clip_002.mp4"))

        template_path = tmp_path / "selections.json"
        result = generate_selection_template(src_dir, template_path)

        assert result == template_path
        assert template_path.exists()

        data = json.loads(template_path.read_text())
        assert "clip_001.mp4" in data
        assert "clip_002.mp4" in data
        assert data["clip_001.mp4"]["frame"] == 0
        assert data["clip_002.mp4"]["frame"] == 0

    def test_images_marked_auto(self, tiny_video: Path, tmp_path: Path) -> None:
        """Image files are marked with auto:true in the template."""
        from klippbok.video.extract import generate_selection_template

        src_dir = tmp_path / "clips"
        src_dir.mkdir()
        import shutil
        shutil.copy2(str(tiny_video), str(src_dir / "clip_001.mp4"))
        _write_test_image(src_dir / "still_002.jpg")

        template_path = tmp_path / "selections.json"
        generate_selection_template(src_dir, template_path)

        data = json.loads(template_path.read_text())
        assert data["clip_001.mp4"]["frame"] == 0
        assert data["still_002.jpg"]["auto"] is True


# ---------------------------------------------------------------------------
# extract_from_selections
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestExtractFromSelections:
    """Tests for extract_from_selections() — user-selected frames."""

    def test_basic_selection(self, tiny_video: Path, tmp_path: Path) -> None:
        """Extracts frames at specified numbers."""
        from klippbok.video.extract import extract_from_selections

        src_dir = tmp_path / "clips"
        src_dir.mkdir()
        import shutil
        shutil.copy2(str(tiny_video), str(src_dir / "clip_001.mp4"))

        selections_path = tmp_path / "selections.json"
        selections = {"clip_001.mp4": {"frame": 5}}
        selections_path.write_text(json.dumps(selections))

        out_dir = tmp_path / "refs"
        report = extract_from_selections(src_dir, out_dir, selections_path)

        assert report.succeeded == 1
        assert (out_dir / "clip_001.png").exists()

    def test_auto_image(self, tmp_path: Path) -> None:
        """Image files marked auto are copied."""
        from klippbok.video.extract import extract_from_selections

        src_dir = tmp_path / "clips"
        src_dir.mkdir()
        _write_test_image(src_dir / "still_001.jpg")

        selections_path = tmp_path / "selections.json"
        selections = {"still_001.jpg": {"auto": True}}
        selections_path.write_text(json.dumps(selections))

        out_dir = tmp_path / "refs"
        report = extract_from_selections(src_dir, out_dir, selections_path)

        assert report.succeeded == 1
        assert (out_dir / "still_001.png").exists()

    def test_missing_source_file(self, tmp_path: Path) -> None:
        """Missing source file produces a failed result (not an exception)."""
        from klippbok.video.extract import extract_from_selections

        src_dir = tmp_path / "clips"
        src_dir.mkdir()

        selections_path = tmp_path / "selections.json"
        selections = {"nonexistent.mp4": {"frame": 0}}
        selections_path.write_text(json.dumps(selections))

        out_dir = tmp_path / "refs"
        report = extract_from_selections(src_dir, out_dir, selections_path)

        assert report.failed == 1
        assert report.succeeded == 0

    def test_writes_manifest(self, tiny_video: Path, tmp_path: Path) -> None:
        """Writes reference_images.json after extraction."""
        from klippbok.video.extract import extract_from_selections

        src_dir = tmp_path / "clips"
        src_dir.mkdir()
        import shutil
        shutil.copy2(str(tiny_video), str(src_dir / "clip_001.mp4"))

        selections_path = tmp_path / "selections.json"
        selections = {"clip_001.mp4": {"frame": 0}}
        selections_path.write_text(json.dumps(selections))

        out_dir = tmp_path / "refs"
        extract_from_selections(src_dir, out_dir, selections_path)

        assert (out_dir / "reference_images.json").exists()
