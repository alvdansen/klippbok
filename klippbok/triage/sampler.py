"""Frame sampling from video clips for triage matching.

Extracts a small number of evenly-spaced frames from each clip for
CLIP embedding comparison. Unlike the captioning frame extractor
(which samples at a fixed FPS), this pulls a fixed COUNT of frames
regardless of clip duration — we want consistent coverage for matching.

Frames are extracted as temporary PNGs and cleaned up after embedding.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def _get_duration(video_path: Path) -> float:
    """Get video duration in seconds via ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in seconds.

    Raises:
        RuntimeError: if ffprobe fails or duration can't be determined.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        raise RuntimeError(f"Could not determine duration of {video_path}: {e}")


def sample_clip_frames(
    clip_path: str | Path,
    count: int = 5,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Extract evenly-spaced sample frames from a video clip.

    Pulls exactly `count` frames distributed across the clip duration.
    For a 5-second clip with count=5, extracts frames at approximately
    0.5s, 1.5s, 2.5s, 3.5s, 4.5s.

    WHY fixed count (not FPS): For triage matching, we need consistent
    coverage regardless of clip length. A 2s clip and a 10s clip both
    get the same number of frames to compare against references.

    Args:
        clip_path: Path to the video clip.
        count: Number of frames to extract (default: 5).
        output_dir: Directory for extracted frames. If None, uses a
            temporary directory (caller must clean up).

    Returns:
        Sorted list of extracted frame paths (PNG format).

    Raises:
        FileNotFoundError: if clip doesn't exist.
        RuntimeError: if ffmpeg fails.
    """
    clip_path = Path(clip_path).resolve()

    if not clip_path.exists():
        raise FileNotFoundError(f"Clip not found: {clip_path}")

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="klippbok_triage_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get duration to calculate sampling rate
    duration = _get_duration(clip_path)

    if duration <= 0:
        raise RuntimeError(f"Clip has zero or negative duration: {clip_path}")

    # Calculate FPS to get approximately `count` frames
    sample_fps = count / duration

    output_pattern = str(output_dir / "triage_%04d.png")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(clip_path),
        "-vf", f"fps={sample_fps:.6f}",
        "-frames:v", str(count),
        output_pattern,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=60,
    )

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-300:]
        raise RuntimeError(
            f"ffmpeg frame sampling failed for {clip_path.name}: {stderr}"
        )

    frames = sorted(output_dir.glob("triage_*.png"))
    return frames


def sample_scene_frames(
    video_path: str | Path,
    start_time: float,
    end_time: float,
    count: int = 2,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Extract frames from a specific scene within a long video.

    Uses ffmpeg -ss seeking to jump directly to scene timestamps,
    avoiding full-file decode. This is the key to fast triage of
    raw footage — we only read the few seconds we need.

    Calculates N evenly-spaced timestamps within [start_time, end_time]
    and extracts one frame at each. For a scene from 10.0s to 20.0s
    with count=2, extracts frames at ~13.3s and ~16.7s.

    WHY not just use sample_clip_frames: That function processes
    entire files. For a 2-hour movie, we need to seek to specific
    scenes without decoding the whole thing. -ss before -i gives
    us fast input seeking.

    Args:
        video_path: Path to the source video file.
        start_time: Scene start time in seconds.
        end_time: Scene end time in seconds.
        count: Number of frames to extract (default: 2).
        output_dir: Directory for extracted frames. If None, uses a
            temporary directory (caller must clean up).

    Returns:
        Sorted list of extracted frame paths (PNG format).

    Raises:
        FileNotFoundError: if video doesn't exist.
        ValueError: if start_time >= end_time or count < 1.
        RuntimeError: if ffmpeg fails.
    """
    video_path = Path(video_path).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if start_time >= end_time:
        raise ValueError(
            f"start_time ({start_time}) must be less than end_time ({end_time})"
        )

    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="klippbok_scene_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate evenly-spaced timestamps within the scene.
    # For count=2 in a 10s scene: positions at 1/3 and 2/3 through.
    # This avoids the very start (which may be a cut transition)
    # and the very end (ditto).
    duration = end_time - start_time
    timestamps: list[float] = []
    for i in range(count):
        # Distribute N points evenly, avoiding edges
        t = start_time + duration * (i + 1) / (count + 1)
        timestamps.append(t)

    frames: list[Path] = []
    for i, timestamp in enumerate(timestamps):
        output_file = output_dir / f"scene_{i:04d}.png"

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{timestamp:.3f}",  # seek BEFORE input = fast
            "-i", str(video_path),
            "-frames:v", "1",            # extract exactly one frame
            "-q:v", "2",                  # good quality PNG
            str(output_file),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[-300:]
            raise RuntimeError(
                f"ffmpeg scene frame extraction failed at {timestamp:.1f}s "
                f"in {video_path.name}: {stderr}"
            )

        if output_file.exists():
            frames.append(output_file)

    return sorted(frames)


def cleanup_frames(frame_paths: list[Path]) -> None:
    """Remove temporary frame files and their parent directory.

    Safe to call even if some files are already gone.

    Args:
        frame_paths: List of frame file paths to remove.
    """
    if not frame_paths:
        return

    parent = frame_paths[0].parent

    for frame in frame_paths:
        try:
            frame.unlink(missing_ok=True)
        except OSError:
            pass

    # Remove the parent directory if it's a temp dir and now empty
    try:
        is_temp = (
            parent.name.startswith("klippbok_triage_")
            or parent.name.startswith("klippbok_scene_")
        )
        if is_temp and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass
