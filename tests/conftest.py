"""Shared test fixtures and markers for Klippbok test suite.

Provides:
- @requires_ffmpeg: skip tests if ffmpeg/ffprobe not in PATH
- @requires_scenedetect: skip tests if PySceneDetect not installed
- Fixtures for creating tiny test videos via ffmpeg's testsrc filter
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# PATH setup — ensure ffmpeg is found if installed via winget
# ---------------------------------------------------------------------------

_WINGET_FFMPEG_PATTERN = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft", "WinGet", "Packages", "*ffmpeg*", "**", "bin",
)

def _add_ffmpeg_to_path() -> None:
    """Add WinGet-installed ffmpeg to PATH if not already available.

    WinGet installs ffmpeg to a deep package directory that requires
    a shell restart to appear in PATH. This finds it and adds it
    so tests can run without restarting.
    """
    import glob
    for bin_dir in glob.glob(_WINGET_FFMPEG_PATTERN, recursive=True):
        if os.path.isfile(os.path.join(bin_dir, "ffmpeg.exe")):
            if bin_dir not in os.environ["PATH"]:
                os.environ["PATH"] = bin_dir + ";" + os.environ["PATH"]
            return

_add_ffmpeg_to_path()


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def _has_ffmpeg() -> bool:
    """Check if ffmpeg is available in PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _has_ffprobe() -> bool:
    """Check if ffprobe is available in PATH."""
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _has_scenedetect() -> bool:
    """Check if PySceneDetect is importable."""
    try:
        import scenedetect  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_ffmpeg = pytest.mark.skipif(
    not (_has_ffmpeg() and _has_ffprobe()),
    reason="ffmpeg/ffprobe not found in PATH — install with: winget install ffmpeg",
)

requires_scenedetect = pytest.mark.skipif(
    not _has_scenedetect(),
    reason="PySceneDetect not installed — install with: pip install scenedetect[opencv]",
)


# ---------------------------------------------------------------------------
# Test video fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    """Create a tiny test video: 16fps, 320x240, 17 frames (~1s), H.264.

    Uses ffmpeg's testsrc2 filter to generate colored frames.
    Skipped automatically if ffmpeg is not available.

    Returns:
        Path to the generated .mp4 file.
    """
    if not _has_ffmpeg():
        pytest.skip("ffmpeg not available")

    output = tmp_path / "test_clip.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "testsrc2=size=320x240:rate=16:duration=1.0625",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-frames:v", "17",  # 4*4+1 = valid frame count
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        pytest.skip(f"Failed to create test video: {result.stderr.decode()[-200:]}")

    return output


@pytest.fixture
def tiny_video_30fps(tmp_path: Path) -> Path:
    """Create a test video at 30fps (wrong for Wan training).

    17 frames at 30fps — fps mismatch should be detected.
    """
    if not _has_ffmpeg():
        pytest.skip("ffmpeg not available")

    output = tmp_path / "test_30fps.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "testsrc2=size=320x240:rate=30:duration=0.567",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-frames:v", "17",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        pytest.skip(f"Failed to create test video: {result.stderr.decode()[-200:]}")

    return output


@pytest.fixture
def tiny_video_720p(tmp_path: Path) -> Path:
    """Create a test video at 720p resolution.

    16fps, 1280x720, 17 frames. Above 480p target.
    """
    if not _has_ffmpeg():
        pytest.skip("ffmpeg not available")

    output = tmp_path / "test_720p.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "testsrc2=size=1280x720:rate=16:duration=1.0625",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-frames:v", "17",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        pytest.skip(f"Failed to create test video: {result.stderr.decode()[-200:]}")

    return output


@pytest.fixture
def tiny_video_18frames(tmp_path: Path) -> Path:
    """Create a test video with invalid frame count (18 = not 4n+1).

    Should be trimmed to 17 during normalization.
    """
    if not _has_ffmpeg():
        pytest.skip("ffmpeg not available")

    output = tmp_path / "test_18frames.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "testsrc2=size=320x240:rate=16:duration=1.125",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-frames:v", "18",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        pytest.skip(f"Failed to create test video: {result.stderr.decode()[-200:]}")

    return output


@pytest.fixture
def two_scene_video(tmp_path: Path) -> Path:
    """Create a video with two distinct scenes (for scene detection tests).

    First half: blue test pattern. Second half: red pattern.
    The abrupt color change should trigger scene detection.
    """
    if not _has_ffmpeg():
        pytest.skip("ffmpeg not available")

    # Create two short clips with different colors
    blue = tmp_path / "blue.mp4"
    red = tmp_path / "red.mp4"
    combined = tmp_path / "two_scenes.mp4"

    # Blue clip (1 second)
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "color=c=blue:size=320x240:rate=16:d=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(blue),
    ], capture_output=True, timeout=30)

    # Red clip (1 second)
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "color=c=red:size=320x240:rate=16:d=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(red),
    ], capture_output=True, timeout=30)

    # Concatenate them
    filelist = tmp_path / "filelist.txt"
    filelist.write_text(f"file '{blue}'\nfile '{red}'\n")

    result = subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(filelist),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(combined),
    ], capture_output=True, timeout=30)

    if result.returncode != 0:
        pytest.skip("Failed to create two-scene test video")

    return combined
