"""Video probing via ffprobe.

Extracts metadata from video files: resolution, fps, frame count, codec,
SAR, duration, etc. This is the first step in any video pipeline —
you need to know what you have before you can validate or process it.

ffprobe is part of the ffmpeg suite and must be installed separately.
On Windows: winget install ffmpeg
"""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

from klippbok.video.errors import FFmpegNotFoundError, ProbeError
from klippbok.video.models import VideoMetadata


def _check_ffprobe() -> str:
    """Verify ffprobe is available and return its path.

    Raises FFmpegNotFoundError with install instructions if not found.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return "ffprobe"
    except FileNotFoundError:
        raise FFmpegNotFoundError("ffprobe")
    except subprocess.TimeoutExpired:
        raise FFmpegNotFoundError("ffprobe")

    raise FFmpegNotFoundError("ffprobe")


def _parse_frame_rate(rate_str: str) -> float:
    """Parse ffprobe's frame rate string into a float.

    ffprobe reports frame rates as fractions like '24000/1001' (23.976fps)
    or '30/1' (30fps). We convert to float for easier comparison.

    Args:
        rate_str: Frame rate string from ffprobe (e.g. '24000/1001', '30/1')

    Returns:
        Frame rate as a float, rounded to 3 decimal places.
    """
    try:
        frac = Fraction(rate_str)
        return round(float(frac), 3)
    except (ValueError, ZeroDivisionError):
        # Fallback: try parsing as plain float
        try:
            return round(float(rate_str), 3)
        except ValueError:
            return 0.0


def _parse_sar(sar_str: str | None) -> str:
    """Normalize SAR string to 'W:H' format.

    ffprobe may report SAR as '1:1', '1/1', 'N/A', or None.
    We normalize to colon-separated format for consistency.

    Args:
        sar_str: Raw SAR string from ffprobe.

    Returns:
        Normalized SAR like '1:1' or '4:3'.
    """
    if not sar_str or sar_str in ("N/A", "0:0", "0/0"):
        return "1:1"
    # Normalize slash to colon
    return sar_str.replace("/", ":")


def probe_video(path: str | Path) -> VideoMetadata:
    """Extract metadata from a video file using ffprobe.

    Calls ffprobe to get stream information (resolution, fps, codec, etc.)
    and format information (duration, file size, container). Returns a
    VideoMetadata model with all available fields.

    Special handling:
    - Frame count: some containers don't store nb_frames. In that case,
      we estimate from duration × fps.
    - SAR: normalized to 'W:H' format. Missing/N/A treated as '1:1'.
    - Duration: pulled from format (container) level, more reliable than stream.

    Args:
        path: Path to the video file.

    Returns:
        VideoMetadata with all probed fields.

    Raises:
        FFmpegNotFoundError: if ffprobe is not in PATH.
        ProbeError: if the file can't be probed (corrupted, not a video, etc).
    """
    _check_ffprobe()
    path = Path(path).resolve()

    if not path.exists():
        raise ProbeError(str(path), "File does not exist")

    if not path.is_file():
        raise ProbeError(str(path), "Path is not a file")

    # Run ffprobe with JSON output for reliable parsing.
    # -show_streams gets per-stream info (resolution, codec, fps)
    # -show_format gets container info (duration, file size)
    cmd = [
        "ffprobe",
        "-v", "quiet",           # suppress banner and warnings
        "-print_format", "json", # structured output
        "-show_streams",         # stream-level info
        "-show_format",          # format/container info
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise ProbeError(str(path), "ffprobe timed out after 30 seconds")
    except FileNotFoundError:
        raise FFmpegNotFoundError("ffprobe")

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "unknown error"
        raise ProbeError(str(path), stderr)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise ProbeError(str(path), "ffprobe returned invalid JSON")

    # Find the first video stream
    video_stream = None
    has_audio = False
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        if stream.get("codec_type") == "audio":
            has_audio = True

    if video_stream is None:
        raise ProbeError(str(path), "No video stream found in file")

    fmt = data.get("format", {})

    # Parse frame rate — prefer r_frame_rate (real/average frame rate)
    # over avg_frame_rate which can be unreliable for VFR content
    fps_str = video_stream.get("r_frame_rate", video_stream.get("avg_frame_rate", "0/1"))
    fps = _parse_frame_rate(fps_str)

    # Parse frame count — nb_frames is ideal but not always available.
    # Fallback: duration × fps (less accurate but usually close enough).
    nb_frames_str = video_stream.get("nb_frames", "N/A")
    if nb_frames_str not in ("N/A", "", None):
        try:
            frame_count = int(nb_frames_str)
        except ValueError:
            frame_count = 0
    else:
        frame_count = 0

    # Get duration from format level (more reliable than stream level)
    duration = 0.0
    duration_str = fmt.get("duration", video_stream.get("duration", "0"))
    if duration_str and duration_str != "N/A":
        try:
            duration = float(duration_str)
        except ValueError:
            duration = 0.0

    # Estimate frame count from duration if not available directly
    if frame_count == 0 and duration > 0 and fps > 0:
        frame_count = round(duration * fps)

    # Parse bit rate
    bit_rate = None
    br_str = video_stream.get("bit_rate")
    if br_str and br_str != "N/A":
        try:
            bit_rate = int(br_str)
        except ValueError:
            pass

    # Parse file size
    file_size = None
    size_str = fmt.get("size")
    if size_str:
        try:
            file_size = int(size_str)
        except ValueError:
            pass

    return VideoMetadata(
        path=path,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        codec=video_stream.get("codec_name", "unknown"),
        pix_fmt=video_stream.get("pix_fmt"),
        sar=_parse_sar(video_stream.get("sample_aspect_ratio")),
        bit_rate=bit_rate,
        file_size=file_size,
        has_audio=has_audio,
        container=fmt.get("format_name"),
    )


def probe_directory(
    directory: str | Path,
    extensions: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".webm"),
) -> list[VideoMetadata]:
    """Probe all video files in a directory.

    Scans for files with known video extensions, probes each one,
    and returns metadata sorted by filename.

    Files that fail to probe are skipped with a warning printed to console.
    This is intentional — a bad file shouldn't stop you from scanning
    the rest of the directory.

    Args:
        directory: Path to scan for video files.
        extensions: File extensions to include (case-insensitive).

    Returns:
        List of VideoMetadata, sorted by filename.

    Raises:
        FFmpegNotFoundError: if ffprobe is not in PATH.
        FileNotFoundError: if the directory doesn't exist.
    """
    directory = Path(directory).resolve()

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    # Check ffprobe once before scanning (fail fast)
    _check_ffprobe()

    # Find all video files, sorted by name for deterministic order
    video_files = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )

    results: list[VideoMetadata] = []
    for video_file in video_files:
        try:
            meta = probe_video(video_file)
            results.append(meta)
        except ProbeError as e:
            # Skip bad files but warn the user
            print(f"  WARNING: Skipping {video_file.name}: {e}")

    return results
