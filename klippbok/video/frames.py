"""Frame extraction from video clips using ffmpeg.

Standalone tool that extracts frames at a configurable FPS rate.
Useful for feeding video content to image-only VLMs (local models
that can't process video natively) and for visual inspection.

All VLMs ultimately work from frames — even Gemini extracts frames
server-side at 1 FPS by default. This module gives you explicit
control over frame sampling rate for local backends.

Usage:
    from klippbok.video.frames import extract_frames

    frames = extract_frames("clip.mp4", output_dir, fps=2)
    # Returns list of Path objects: [frame_001.jpg, frame_002.jpg, ...]
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: int = 1,
    format: str = "jpg",
) -> list[Path]:
    """Extract frames from a video at a specified FPS rate.

    Uses ffmpeg's fps filter to sample frames at a fixed rate.
    A 5-second clip at fps=2 gives ~10 frames. The number of frames
    scales with video duration, not with a fixed count.

    WHY FPS-based (not fixed count): For captioning, you want
    consistent temporal sampling regardless of clip length. A 2s clip
    and a 5s clip at fps=2 both give proportional coverage.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to write extracted frame images.
            Created if it doesn't exist.
        fps: Frames per second to extract (default: 1).
            Higher = more frames = better motion capture.
            1 FPS is standard for most captioning.
            2-4 FPS recommended for motion-focused datasets.
        format: Output image format — 'jpg' or 'png' (default: jpg).
            JPG is smaller and faster. PNG is lossless.

    Returns:
        Sorted list of extracted frame file paths.

    Raises:
        FileNotFoundError: if video_path doesn't exist.
        RuntimeError: if ffmpeg fails.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command
    # -q:v 2 gives good JPEG quality without huge file sizes
    ext = format.lower()
    quality_args = ["-q:v", "2"] if ext == "jpg" else []

    output_pattern = str(output_dir / f"frame_%04d.{ext}")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        *quality_args,
        output_pattern,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=120,
    )

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-300:]
        raise RuntimeError(
            f"ffmpeg frame extraction failed (exit code {result.returncode}): "
            f"{stderr}"
        )

    # Collect and sort the extracted frames
    frames = sorted(output_dir.glob(f"frame_*.{ext}"))
    return frames
