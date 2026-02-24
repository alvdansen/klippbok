"""Tests for klippbok.video.image_quality — sharpness and blank detection.

Pure Python + OpenCV tests (no ffmpeg). Test images are generated
with numpy arrays written via OpenCV — sharp images have texture,
blank images are solid colors, blurry images are Gaussian-filtered.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from klippbok.video.image_quality import compute_sharpness, is_blank, validate_extracted_image


# ---------------------------------------------------------------------------
# Helpers — generate test images with known properties
# ---------------------------------------------------------------------------

def _write_sharp_image(path: Path, width: int = 320, height: int = 240) -> None:
    """Write a sharp test image with high-frequency texture.

    Uses a checkerboard pattern — maximum edge density, so Laplacian
    variance will be very high.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Checkerboard: alternating 4x4 blocks of black and white
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            if ((x // 4) + (y // 4)) % 2 == 0:
                img[y:y+4, x:x+4] = 255
    cv2.imwrite(str(path), img)


def _write_blank_image(
    path: Path,
    color: tuple[int, int, int] = (0, 0, 0),
    width: int = 320,
    height: int = 240,
) -> None:
    """Write a solid-color (blank) image. Default is black."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_blurry_image(path: Path, width: int = 320, height: int = 240) -> None:
    """Write a blurry image — sharp texture + heavy Gaussian blur.

    Starts with a checkerboard (sharp), then applies a large Gaussian
    kernel to remove all detail. Should have sharpness between blank
    and sharp.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            if ((x // 4) + (y // 4)) % 2 == 0:
                img[y:y+4, x:x+4] = 255
    # Heavy blur — kernel 51x51
    blurred = cv2.GaussianBlur(img, (51, 51), 0)
    cv2.imwrite(str(path), blurred)


def _write_noisy_image(path: Path, width: int = 320, height: int = 240) -> None:
    """Write a random noise image — moderate sharpness everywhere."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# compute_sharpness
# ---------------------------------------------------------------------------

class TestComputeSharpness:
    """Tests for compute_sharpness() — Laplacian variance metric."""

    def test_sharp_image_high_score(self, tmp_path: Path) -> None:
        """A checkerboard pattern has very high sharpness."""
        img_path = tmp_path / "sharp.png"
        _write_sharp_image(img_path)
        sharpness = compute_sharpness(img_path)
        # Checkerboard: expect very high variance (thousands)
        assert sharpness > 100.0

    def test_blank_image_low_score(self, tmp_path: Path) -> None:
        """A solid black image has near-zero sharpness."""
        img_path = tmp_path / "blank.png"
        _write_blank_image(img_path)
        sharpness = compute_sharpness(img_path)
        assert sharpness < 1.0

    def test_white_image_low_score(self, tmp_path: Path) -> None:
        """A solid white image has near-zero sharpness."""
        img_path = tmp_path / "white.png"
        _write_blank_image(img_path, color=(255, 255, 255))
        sharpness = compute_sharpness(img_path)
        assert sharpness < 1.0

    def test_gray_image_low_score(self, tmp_path: Path) -> None:
        """A solid gray image has near-zero sharpness."""
        img_path = tmp_path / "gray.png"
        _write_blank_image(img_path, color=(128, 128, 128))
        sharpness = compute_sharpness(img_path)
        assert sharpness < 1.0

    def test_blurry_less_than_sharp(self, tmp_path: Path) -> None:
        """A blurred image has less sharpness than a sharp one."""
        sharp_path = tmp_path / "sharp.png"
        blurry_path = tmp_path / "blurry.png"
        _write_sharp_image(sharp_path)
        _write_blurry_image(blurry_path)

        sharp_score = compute_sharpness(sharp_path)
        blurry_score = compute_sharpness(blurry_path)
        assert blurry_score < sharp_score

    def test_noise_image(self, tmp_path: Path) -> None:
        """Random noise has moderate sharpness (lots of high-frequency content)."""
        img_path = tmp_path / "noise.png"
        _write_noisy_image(img_path)
        sharpness = compute_sharpness(img_path)
        # Noise has edges everywhere — moderate to high
        assert sharpness > 10.0

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            compute_sharpness(tmp_path / "nonexistent.png")

    def test_invalid_image_file(self, tmp_path: Path) -> None:
        """Raises ValueError for non-image files."""
        bad_file = tmp_path / "not_an_image.txt"
        bad_file.write_text("this is not an image")
        with pytest.raises(ValueError, match="Cannot read image"):
            compute_sharpness(bad_file)

    def test_returns_float(self, tmp_path: Path) -> None:
        """Return value is always a float."""
        img_path = tmp_path / "test.png"
        _write_sharp_image(img_path)
        result = compute_sharpness(img_path)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# is_blank
# ---------------------------------------------------------------------------

class TestIsBlank:
    """Tests for is_blank() — uniform/solid frame detection."""

    def test_black_is_blank(self, tmp_path: Path) -> None:
        """Solid black frame is detected as blank."""
        img_path = tmp_path / "black.png"
        _write_blank_image(img_path)
        assert is_blank(img_path) is True

    def test_white_is_blank(self, tmp_path: Path) -> None:
        """Solid white frame is detected as blank."""
        img_path = tmp_path / "white.png"
        _write_blank_image(img_path, color=(255, 255, 255))
        assert is_blank(img_path) is True

    def test_sharp_not_blank(self, tmp_path: Path) -> None:
        """Textured image is not blank."""
        img_path = tmp_path / "sharp.png"
        _write_sharp_image(img_path)
        assert is_blank(img_path) is False

    def test_custom_threshold(self, tmp_path: Path) -> None:
        """Custom threshold changes the sensitivity."""
        img_path = tmp_path / "blurry.png"
        _write_blurry_image(img_path)
        sharpness = compute_sharpness(img_path)
        # With a very high threshold, even blurry images are "blank"
        assert is_blank(img_path, threshold=sharpness + 100.0) is True
        # With a very low threshold, nothing is blank
        assert is_blank(img_path, threshold=0.001) is False


# ---------------------------------------------------------------------------
# validate_extracted_image
# ---------------------------------------------------------------------------

class TestValidateExtractedImage:
    """Tests for validate_extracted_image() — combined quality check."""

    def test_good_image(self, tmp_path: Path) -> None:
        """Sharp image with correct dimensions passes all checks."""
        img_path = tmp_path / "good.png"
        _write_sharp_image(img_path, width=320, height=240)

        val = validate_extracted_image(img_path, expected_width=320, expected_height=240)
        assert val.width == 320
        assert val.height == 240
        assert val.sharpness > 100.0
        assert val.is_blank is False
        assert val.resolution_ok is True

    def test_blank_image_detected(self, tmp_path: Path) -> None:
        """Blank image is flagged even if resolution matches."""
        img_path = tmp_path / "blank.png"
        _write_blank_image(img_path, width=320, height=240)

        val = validate_extracted_image(img_path, expected_width=320, expected_height=240)
        assert val.is_blank is True
        assert val.resolution_ok is True

    def test_resolution_mismatch(self, tmp_path: Path) -> None:
        """Wrong resolution is flagged."""
        img_path = tmp_path / "test.png"
        _write_sharp_image(img_path, width=640, height=480)

        val = validate_extracted_image(img_path, expected_width=320, expected_height=240)
        assert val.resolution_ok is False
        assert val.width == 640
        assert val.height == 480

    def test_no_expected_dimensions(self, tmp_path: Path) -> None:
        """When no expected dimensions given, resolution_ok is True."""
        img_path = tmp_path / "test.png"
        _write_sharp_image(img_path, width=640, height=480)

        val = validate_extracted_image(img_path)
        assert val.resolution_ok is True
        assert val.expected_width is None
        assert val.expected_height is None

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            validate_extracted_image(tmp_path / "nope.png")

    def test_path_is_stored(self, tmp_path: Path) -> None:
        """The validated path is stored in the result."""
        img_path = tmp_path / "test.png"
        _write_sharp_image(img_path)
        val = validate_extracted_image(img_path)
        assert val.path == img_path
