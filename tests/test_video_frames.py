"""Tests for klippbok.video.frames — frame extraction from video clips.

These tests require ffmpeg to be installed (uses the tiny_video fixture
from conftest.py). They create real video files and extract real frames.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg


@requires_ffmpeg
class TestExtractFrames:
    """Tests for extract_frames() — requires ffmpeg."""

    def test_basic_extraction(self, tiny_video: Path, tmp_path: Path) -> None:
        """Extracts frames from a video at 1 FPS."""
        from klippbok.video.frames import extract_frames

        output_dir = tmp_path / "frames"
        frames = extract_frames(tiny_video, output_dir, fps=1)

        # tiny_video is ~1 second at 16fps, so 1 FPS should give ~1 frame
        assert len(frames) >= 1
        assert all(f.exists() for f in frames)
        assert all(f.suffix == ".jpg" for f in frames)

    def test_creates_output_dir(self, tiny_video: Path, tmp_path: Path) -> None:
        """Creates the output directory if it doesn't exist."""
        from klippbok.video.frames import extract_frames

        output_dir = tmp_path / "nested" / "frames"
        assert not output_dir.exists()

        frames = extract_frames(tiny_video, output_dir, fps=1)
        assert output_dir.exists()
        assert len(frames) >= 1

    def test_png_format(self, tiny_video: Path, tmp_path: Path) -> None:
        """Can extract frames as PNG."""
        from klippbok.video.frames import extract_frames

        output_dir = tmp_path / "frames"
        frames = extract_frames(tiny_video, output_dir, fps=1, format="png")

        assert len(frames) >= 1
        assert all(f.suffix == ".png" for f in frames)

    def test_nonexistent_video(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing video."""
        from klippbok.video.frames import extract_frames

        with pytest.raises(FileNotFoundError):
            extract_frames(tmp_path / "nonexistent.mp4", tmp_path / "frames")

    def test_sorted_output(self, tiny_video: Path, tmp_path: Path) -> None:
        """Output frames are sorted by name."""
        from klippbok.video.frames import extract_frames

        output_dir = tmp_path / "frames"
        frames = extract_frames(tiny_video, output_dir, fps=16)

        # At 16fps on a 1s video, should get ~16-17 frames
        assert len(frames) >= 5
        names = [f.name for f in frames]
        assert names == sorted(names)
