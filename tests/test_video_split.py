"""Tests for klippbok.video.split — requires ffmpeg.

Tests normalization and splitting of video clips.
Skipped if ffmpeg is not available.
"""

from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg

from klippbok.config.data_schema import VideoConfig
from klippbok.video.probe import probe_video


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestNormalizeClip:
    """Test normalizing individual clips."""

    def test_copy_already_correct(self, tiny_video: Path, tmp_path: Path) -> None:
        """Clip matching config specs should be stream-copied (fast)."""
        from klippbok.video.split import normalize_clip

        output = tmp_path / "output" / "normalized.mp4"
        # Test videos are 320x240, below 480p target — will trigger downscale warning
        # but we use 480p because VideoConfig only allows 480/720
        config = VideoConfig(fps=16, resolution=480, frame_count="auto", upscale_policy="warn")

        clip_info = normalize_clip(tiny_video, output, config)

        assert output.exists()
        assert clip_info.output == output.resolve()
        assert clip_info.frame_count > 0

    def test_reencode_fps_change(self, tiny_video_30fps: Path, tmp_path: Path) -> None:
        """Clip with wrong FPS should be re-encoded."""
        from klippbok.video.split import normalize_clip

        output = tmp_path / "output" / "normalized.mp4"
        config = VideoConfig(fps=16, resolution=480, frame_count="auto", upscale_policy="warn")

        clip_info = normalize_clip(tiny_video_30fps, output, config)

        assert output.exists()
        assert clip_info.was_reencoded is True

        # Verify output has correct fps
        output_meta = probe_video(output)
        assert abs(output_meta.fps - 16.0) < 0.5

    def test_downscale_resolution(self, tiny_video_720p: Path, tmp_path: Path) -> None:
        """720p clip should be downscaled to 480p."""
        from klippbok.video.split import normalize_clip

        output = tmp_path / "output" / "normalized.mp4"
        config = VideoConfig(fps=16, resolution=480, frame_count="auto")

        clip_info = normalize_clip(tiny_video_720p, output, config)

        assert output.exists()
        assert clip_info.was_reencoded is True

        output_meta = probe_video(output)
        assert output_meta.height == 480

    def test_trim_frame_count(self, tiny_video_18frames: Path, tmp_path: Path) -> None:
        """18-frame clip should be trimmed to 17 (nearest 4n+1)."""
        from klippbok.video.split import normalize_clip

        output = tmp_path / "output" / "normalized.mp4"
        config = VideoConfig(fps=16, resolution=480, frame_count="auto", upscale_policy="warn")

        clip_info = normalize_clip(tiny_video_18frames, output, config)

        assert output.exists()
        # Frame count should be valid 4n+1
        assert (clip_info.frame_count - 1) % 4 == 0 or clip_info.frame_count == 1

    def test_output_dir_created(self, tiny_video: Path, tmp_path: Path) -> None:
        """Output directory is created if it doesn't exist."""
        from klippbok.video.split import normalize_clip

        output = tmp_path / "deep" / "nested" / "dir" / "clip.mp4"
        config = VideoConfig(fps=16, resolution=480, frame_count="auto", upscale_policy="warn")

        normalize_clip(tiny_video, output, config)
        assert output.exists()


@requires_ffmpeg
class TestNormalizeDirectory:
    """Test batch normalization of a directory."""

    def test_normalize_directory(self, tiny_video: Path, tmp_path: Path) -> None:
        """Normalize all clips in a directory."""
        from klippbok.video.split import normalize_directory
        import shutil

        # Set up source directory with multiple clips
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        shutil.copy(tiny_video, source_dir / "clip_001.mp4")
        shutil.copy(tiny_video, source_dir / "clip_002.mp4")

        output_dir = tmp_path / "output"
        config = VideoConfig(fps=16, resolution=480, frame_count="auto", upscale_policy="warn")

        results = normalize_directory(source_dir, output_dir, config)

        assert len(results) == 2
        assert (output_dir / "clip_001.mp4").exists()
        assert (output_dir / "clip_002.mp4").exists()

    def test_normalize_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory produces no results."""
        from klippbok.video.split import normalize_directory

        source_dir = tmp_path / "empty"
        source_dir.mkdir()
        output_dir = tmp_path / "output"
        config = VideoConfig(fps=16, resolution=480, frame_count="auto", upscale_policy="warn")

        results = normalize_directory(source_dir, output_dir, config)
        assert results == []
