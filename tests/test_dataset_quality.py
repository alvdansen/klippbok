"""Tests for klippbok.dataset.quality — exposure, motion, dHash metrics.

Most tests use numpy-generated synthetic images written to tmp_path.
Motion tests that need real videos are marked @requires_ffmpeg.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.conftest import requires_ffmpeg

from klippbok.dataset.quality import (
    compute_dhash,
    compute_exposure,
    compute_motion_intensity,
    find_duplicates,
    hamming_distance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_gray_image(path: Path, pixels: np.ndarray) -> Path:
    """Write a grayscale numpy array as a PNG file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), pixels)
    return path


def _make_solid(tmp_path: Path, name: str, value: int, size: int = 64) -> Path:
    """Create a solid-color grayscale image."""
    img = np.full((size, size), value, dtype=np.uint8)
    return _save_gray_image(tmp_path / name, img)


def _make_gradient(tmp_path: Path, name: str, size: int = 64) -> Path:
    """Create a horizontal gradient image (left=0, right=255)."""
    row = np.linspace(0, 255, size, dtype=np.uint8)
    img = np.tile(row, (size, 1))
    return _save_gray_image(tmp_path / name, img)


def _make_noise(tmp_path: Path, name: str, size: int = 64, seed: int = 42) -> Path:
    """Create a random noise image."""
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, (size, size), dtype=np.uint8)
    return _save_gray_image(tmp_path / name, img)


def _make_checkerboard(tmp_path: Path, name: str, size: int = 64, block: int = 8) -> Path:
    """Create a checkerboard pattern image."""
    img = np.zeros((size, size), dtype=np.uint8)
    for r in range(size):
        for c in range(size):
            if (r // block + c // block) % 2 == 0:
                img[r, c] = 255
    return _save_gray_image(tmp_path / name, img)


# ---------------------------------------------------------------------------
# Exposure tests
# ---------------------------------------------------------------------------

class TestComputeExposure:
    def test_black_image(self, tmp_path: Path):
        path = _make_solid(tmp_path, "black.png", 0)
        mean, std = compute_exposure(path)
        assert mean == pytest.approx(0.0, abs=0.01)
        assert std == pytest.approx(0.0, abs=0.01)

    def test_white_image(self, tmp_path: Path):
        path = _make_solid(tmp_path, "white.png", 255)
        mean, std = compute_exposure(path)
        assert mean == pytest.approx(1.0, abs=0.01)
        assert std == pytest.approx(0.0, abs=0.01)

    def test_mid_gray(self, tmp_path: Path):
        path = _make_solid(tmp_path, "gray.png", 128)
        mean, std = compute_exposure(path)
        assert 0.45 < mean < 0.55  # ~0.50
        assert std == pytest.approx(0.0, abs=0.01)

    def test_high_contrast(self, tmp_path: Path):
        """An image with both black and white has high std."""
        path = _make_checkerboard(tmp_path, "checker.png")
        mean, std = compute_exposure(path)
        assert 0.4 < mean < 0.6  # roughly centered
        assert std > 0.3  # high contrast

    def test_gradient_exposure(self, tmp_path: Path):
        path = _make_gradient(tmp_path, "grad.png")
        mean, std = compute_exposure(path)
        assert 0.45 < mean < 0.55  # roughly centered
        assert std > 0.2  # has spread

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            compute_exposure(tmp_path / "nonexistent.png")

    def test_unreadable_file(self, tmp_path: Path):
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image")
        with pytest.raises(ValueError, match="Cannot read"):
            compute_exposure(bad)

    def test_returns_tuple(self, tmp_path: Path):
        path = _make_gradient(tmp_path, "grad.png")
        result = compute_exposure(path)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


# ---------------------------------------------------------------------------
# dHash tests
# ---------------------------------------------------------------------------

class TestComputeDhash:
    def test_deterministic(self, tmp_path: Path):
        """Same image produces the same hash every time."""
        path = _make_gradient(tmp_path, "grad.png")
        h1 = compute_dhash(path)
        h2 = compute_dhash(path)
        assert h1 == h2

    def test_returns_integer(self, tmp_path: Path):
        path = _make_gradient(tmp_path, "grad.png")
        h = compute_dhash(path)
        assert isinstance(h, int)

    def test_different_images_different_hashes(self, tmp_path: Path):
        """Visually different images should produce different hashes."""
        grad = _make_gradient(tmp_path, "grad.png")
        noise = _make_noise(tmp_path, "noise.png")
        h_grad = compute_dhash(grad)
        h_noise = compute_dhash(noise)
        assert h_grad != h_noise

    def test_solid_images_similar(self, tmp_path: Path):
        """Two solid-color images should be very similar."""
        black = _make_solid(tmp_path, "black.png", 0)
        dark = _make_solid(tmp_path, "dark.png", 10)
        h_black = compute_dhash(black)
        h_dark = compute_dhash(dark)
        assert hamming_distance(h_black, h_dark) <= 5

    def test_custom_hash_size(self, tmp_path: Path):
        path = _make_gradient(tmp_path, "grad.png")
        h4 = compute_dhash(path, hash_size=4)
        h8 = compute_dhash(path, hash_size=8)
        # Different hash sizes produce different-magnitude values
        assert isinstance(h4, int)
        assert isinstance(h8, int)

    def test_similar_images_low_distance(self, tmp_path: Path):
        """Same image at different sizes should have similar hashes."""
        # Create a 128x128 gradient and a 64x64 gradient
        big = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
        big_path = tmp_path / "big.png"
        cv2.imwrite(str(big_path), big)

        small = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
        small_path = tmp_path / "small.png"
        cv2.imwrite(str(small_path), small)

        h_big = compute_dhash(big_path)
        h_small = compute_dhash(small_path)
        assert hamming_distance(h_big, h_small) <= 5

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            compute_dhash(tmp_path / "nonexistent.png")

    def test_unreadable_file(self, tmp_path: Path):
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image")
        with pytest.raises(ValueError, match="Cannot read"):
            compute_dhash(bad)


# ---------------------------------------------------------------------------
# Hamming distance tests
# ---------------------------------------------------------------------------

class TestHammingDistance:
    def test_identical(self):
        assert hamming_distance(0, 0) == 0
        assert hamming_distance(0xFFFF, 0xFFFF) == 0

    def test_one_bit(self):
        assert hamming_distance(0b0000, 0b0001) == 1
        assert hamming_distance(0b1000, 0b0000) == 1

    def test_all_different(self):
        assert hamming_distance(0b0000, 0b1111) == 4

    def test_known_distance(self):
        assert hamming_distance(0b10101010, 0b01010101) == 8

    def test_symmetric(self):
        assert hamming_distance(123, 456) == hamming_distance(456, 123)


# ---------------------------------------------------------------------------
# find_duplicates tests
# ---------------------------------------------------------------------------

class TestFindDuplicates:
    def test_no_duplicates(self, tmp_path: Path):
        """Different images produce no duplicate groups."""
        paths = [
            _make_gradient(tmp_path, "grad.png"),
            _make_noise(tmp_path, "noise.png"),
            _make_checkerboard(tmp_path, "checker.png"),
        ]
        groups = find_duplicates(paths, threshold=3)
        assert len(groups) == 0

    def test_identical_images(self, tmp_path: Path):
        """Identical images (same content, different path) form a group."""
        img = np.full((64, 64), 128, dtype=np.uint8)
        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        cv2.imwrite(str(p1), img)
        cv2.imwrite(str(p2), img)
        groups = find_duplicates([p1, p2])
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_near_duplicates(self, tmp_path: Path):
        """Images with tiny differences are grouped as duplicates."""
        rng = np.random.RandomState(42)
        base = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        p1 = tmp_path / "orig.png"
        cv2.imwrite(str(p1), base)

        # Add very slight noise (< 5 pixel values)
        noisy = base.copy()
        noisy[0:2, 0:2] = (noisy[0:2, 0:2].astype(int) + 1).clip(0, 255).astype(np.uint8)
        p2 = tmp_path / "noisy.png"
        cv2.imwrite(str(p2), noisy)

        groups = find_duplicates([p1, p2], threshold=6)
        assert len(groups) == 1

    def test_single_image(self, tmp_path: Path):
        """Single image can't be a duplicate."""
        p = _make_gradient(tmp_path, "grad.png")
        groups = find_duplicates([p])
        assert len(groups) == 0

    def test_empty_list(self):
        groups = find_duplicates([])
        assert len(groups) == 0

    def test_unreadable_files_skipped(self, tmp_path: Path):
        """Unreadable files are skipped, not crashed on."""
        good = _make_gradient(tmp_path, "good.png")
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image")
        groups = find_duplicates([good, bad])
        assert len(groups) == 0

    def test_three_duplicates_one_group(self, tmp_path: Path):
        """Three identical images form one group."""
        img = np.full((64, 64), 200, dtype=np.uint8)
        paths = []
        for name in ["a.png", "b.png", "c.png"]:
            p = tmp_path / name
            cv2.imwrite(str(p), img)
            paths.append(p)
        groups = find_duplicates(paths)
        assert len(groups) == 1
        assert len(groups[0]) == 3


# ---------------------------------------------------------------------------
# Motion intensity tests (require ffmpeg)
# ---------------------------------------------------------------------------

class TestComputeMotionIntensity:
    @requires_ffmpeg
    def test_static_video(self, tmp_path: Path):
        """A static video (all frames identical) has ~0 motion."""
        out = tmp_path / "static.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=c=blue:size=64x64:rate=16:d=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out),
        ], capture_output=True, timeout=30)
        motion = compute_motion_intensity(out)
        assert motion < 1.0  # near-zero for static

    @requires_ffmpeg
    def test_dynamic_video(self, tmp_path: Path):
        """A video with changing patterns has measurable motion."""
        out = tmp_path / "dynamic.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc2=size=64x64:rate=16:d=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out),
        ], capture_output=True, timeout=30)
        motion = compute_motion_intensity(out)
        assert motion > 0.5  # should have visible motion

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            compute_motion_intensity(tmp_path / "nonexistent.mp4")

    @requires_ffmpeg
    def test_returns_float(self, tmp_path: Path):
        out = tmp_path / "test.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=c=red:size=64x64:rate=16:d=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out),
        ], capture_output=True, timeout=30)
        result = compute_motion_intensity(out)
        assert isinstance(result, float)

    @requires_ffmpeg
    def test_custom_sample_count(self, tmp_path: Path):
        out = tmp_path / "test.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=c=green:size=64x64:rate=16:d=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out),
        ], capture_output=True, timeout=30)
        # Should work with different sample counts
        m3 = compute_motion_intensity(out, sample_count=3)
        m10 = compute_motion_intensity(out, sample_count=10)
        # Both should be near zero for static
        assert m3 < 1.0
        assert m10 < 1.0
