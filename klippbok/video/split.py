"""Video splitting and normalization via ffmpeg.

Takes source clips and produces normalized training clips:
- Correct frame rate (re-encode if needed)
- Correct resolution (downscale, never upscale)
- Square pixel aspect ratio (SAR correction)
- Valid 4n+1 frame count (trim from end)
- Consistent codec and container (H.264/mov)

Conservative first draft: any change needed = full re-encode.
No change needed = stream copy (lossless, fast).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from klippbok.config.data_schema import VideoConfig
from klippbok.video.errors import FFmpegNotFoundError, SplitError
from klippbok.video.models import ClipInfo, SceneBoundary, VideoMetadata
from klippbok.video.probe import probe_video
from klippbok.video.validate import nearest_valid_frame_count, validate_clip


def _check_ffmpeg() -> None:
    """Verify ffmpeg is available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        raise FFmpegNotFoundError("ffmpeg")


def _build_encode_cmd(
    source: Path,
    output: Path,
    video_config: VideoConfig,
    target_frame_count: int,
    start_time: float | None = None,
) -> list[str]:
    """Build ffmpeg command for re-encoding a clip.

    Encoding settings:
    - H.264 codec (universal compatibility)
    - CRF 18 (visually lossless — conservative for training data)
    - yuv420p pixel format (compatibility with all players/tools)
    - Scale to target height, preserving aspect ratio (-2 ensures even width)
    - Strip audio (-an) — not needed for video training

    Args:
        source: Input video path.
        output: Output video path.
        video_config: Target video specs.
        target_frame_count: Exact number of frames to output.
        start_time: Optional start time in seconds (for scene splitting).

    Returns:
        ffmpeg command as a list of strings.
    """
    cmd = ["ffmpeg", "-y"]  # -y = overwrite output

    # Input with optional start time
    if start_time is not None:
        cmd.extend(["-ss", f"{start_time:.3f}"])
    cmd.extend(["-i", str(source)])

    # Video filters: scale to target height, preserve aspect ratio
    # -2 ensures width is even (required by H.264)
    # flags= sets the scaling algorithm (lanczos, bicubic, bilinear, area)
    target_height = video_config.resolution
    vf = f"scale=-2:{target_height}:flags={video_config.downscale_method}"

    # Output encoding
    cmd.extend([
        "-c:v", "libx264",         # H.264 codec
        "-crf", "18",              # visually lossless quality
        "-preset", "medium",       # balance speed/compression
        "-pix_fmt", "yuv420p",     # universal pixel format
        "-r", str(video_config.fps),  # target frame rate
        "-vf", vf,                 # scale filter
        "-frames:v", str(target_frame_count),  # exact frame count
        "-an",                     # strip audio
        "-movflags", "+faststart", # web-friendly mp4
        str(output),
    ])

    return cmd


def _build_copy_cmd(
    source: Path,
    output: Path,
    target_frame_count: int,
) -> list[str]:
    """Build ffmpeg command for stream copying (no re-encode).

    Used when the clip already meets all specs — just copy the streams
    into a new container. This is lossless and very fast.

    Args:
        source: Input video path.
        output: Output video path.
        target_frame_count: Exact number of frames to output.

    Returns:
        ffmpeg command as a list of strings.
    """
    return [
        "ffmpeg", "-y",
        "-i", str(source),
        "-c:v", "copy",
        "-frames:v", str(target_frame_count),
        "-an",
        "-movflags", "+faststart",
        str(output),
    ]


def normalize_clip(
    source: str | Path,
    output: str | Path,
    video_config: VideoConfig,
    metadata: VideoMetadata | None = None,
) -> ClipInfo:
    """Normalize a single video clip to match the data config specs.

    Probes the source, validates it, and either stream-copies (if already
    correct) or re-encodes to match the target specs.

    Frame count is always trimmed DOWN to the nearest valid 4n+1 count.
    Frames are trimmed from the end of the clip (preserving the start).

    Args:
        source: Path to the source video file.
        output: Path for the normalized output file.
        video_config: Target video specs.
        metadata: Pre-probed metadata (optional, saves a probe call).

    Returns:
        ClipInfo with processing details.

    Raises:
        FFmpegNotFoundError: if ffmpeg is not in PATH.
        SplitError: if ffmpeg fails to process the clip.
    """
    _check_ffmpeg()
    source = Path(source).resolve()
    output = Path(output).resolve()

    # Probe if metadata not provided
    if metadata is None:
        metadata = probe_video(source)

    # Validate to determine what needs changing
    validation = validate_clip(metadata, video_config)

    # Determine target frame count.
    # Key: if fps is changing, calculate from the POST-conversion frame count
    # (duration * target_fps), not the source frame count. Otherwise we'd ask
    # ffmpeg for more frames than exist at the new rate.
    fps_changing = abs(metadata.fps - video_config.fps) > 0.5
    if fps_changing:
        # How many frames will exist at the target fps?
        post_conversion_fc = int(metadata.duration * video_config.fps)
        target_fc = nearest_valid_frame_count(post_conversion_fc, "down")
    else:
        target_fc = validation.recommended_frame_count or metadata.frame_count
        # Safety check: ensure it's a valid 4n+1 count
        if target_fc not in {4 * n + 1 for n in range(target_fc // 4 + 2)}:
            target_fc = nearest_valid_frame_count(target_fc, "down")

    trimmed = metadata.frame_count - target_fc

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Build and run ffmpeg command
    if validation.needs_reencode:
        cmd = _build_encode_cmd(source, output, video_config, target_fc)
        was_reencoded = True
    else:
        cmd = _build_copy_cmd(source, output, target_fc)
        was_reencoded = False

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes max per clip
        )
    except subprocess.TimeoutExpired:
        raise SplitError(str(source), "ffmpeg timed out after 5 minutes")
    except FileNotFoundError:
        raise FFmpegNotFoundError("ffmpeg")

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "unknown error"
        # Extract the most useful line from ffmpeg's verbose output
        error_lines = [
            line for line in stderr.split("\n")
            if any(kw in line.lower() for kw in ("error", "invalid", "no such"))
        ]
        detail = error_lines[-1] if error_lines else stderr[-200:]
        raise SplitError(str(source), detail)

    # Probe the output to get actual specs
    try:
        output_meta = probe_video(output)
    except Exception:
        # Output was created but can't be probed — use expected values
        output_meta = None

    if output_meta:
        return ClipInfo(
            source=source,
            output=output,
            frame_count=output_meta.frame_count,
            duration=output_meta.duration,
            width=output_meta.width,
            height=output_meta.height,
            fps=output_meta.fps,
            was_reencoded=was_reencoded,
            trimmed_frames=trimmed,
        )
    else:
        # Fallback to expected values
        return ClipInfo(
            source=source,
            output=output,
            frame_count=target_fc,
            duration=target_fc / video_config.fps,
            width=0,  # unknown
            height=video_config.resolution,
            fps=float(video_config.fps),
            was_reencoded=was_reencoded,
            trimmed_frames=trimmed,
        )


def normalize_directory(
    source_dir: str | Path,
    output_dir: str | Path,
    video_config: VideoConfig,
    extensions: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".webm"),
    output_format: str | None = None,
) -> list[ClipInfo]:
    """Normalize all video clips in a directory.

    Scans the source directory for video files, normalizes each one,
    and writes results to the output directory. By default, output files
    keep the same container format as the source (.mov stays .mov, etc.).

    Args:
        source_dir: Directory containing source clips.
        output_dir: Directory for normalized output clips.
        video_config: Target video specs.
        extensions: File extensions to include.
        output_format: Force a specific output extension (e.g. ".mp4", ".mov").
            If None, matches each source file's extension.

    Returns:
        List of ClipInfo for successfully processed clips.

    Raises:
        FFmpegNotFoundError: if ffmpeg is not in PATH.
    """
    _check_ffmpeg()
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find video files
    video_files = sorted(
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )

    if not video_files:
        print(f"No video files found in {source_dir}")
        return []

    results: list[ClipInfo] = []
    total = len(video_files)

    # Sidecar extensions to copy alongside videos (captions, etc.)
    sidecar_extensions = {".txt", ".json", ".jsonl"}
    sidecars_copied = 0

    for i, source_file in enumerate(video_files, 1):
        # Match source format unless override specified
        ext = output_format if output_format else source_file.suffix.lower()
        output_file = output_dir / (source_file.stem + ext)
        print(f"  [{i}/{total}] {source_file.name} ->{output_file.name}")

        try:
            clip_info = normalize_clip(source_file, output_file, video_config)
            results.append(clip_info)
            status = "re-encoded" if clip_info.was_reencoded else "copied"
            trim_msg = f", trimmed {clip_info.trimmed_frames}f" if clip_info.trimmed_frames else ""
            print(f"           {status}{trim_msg} ->{clip_info.frame_count} frames")

            # Copy sidecar files (e.g. clip_001.txt caption for clip_001.mov)
            for ext in sidecar_extensions:
                sidecar = source_file.with_suffix(ext)
                if sidecar.is_file():
                    import shutil
                    dest = output_dir / sidecar.name
                    if not dest.exists():
                        shutil.copy2(sidecar, dest)
                        sidecars_copied += 1

        except Exception as e:
            print(f"           FAILED: {e}")

    print(f"\nDone: {len(results)}/{total} clips processed successfully.")
    if sidecars_copied:
        print(f"Copied {sidecars_copied} sidecar file(s) (.txt, .json)")
    return results


def _subdivide_segments(
    segments: list[tuple[float, float]],
    max_duration: float,
) -> list[tuple[float, float, int, int]]:
    """Subdivide long segments into shorter chunks.

    When a scene is longer than max_duration, splits it into sub-clips
    of at most max_duration. Each chunk gets a scene index and sub-index
    for naming (scene000_a, scene000_b, etc.).

    Args:
        segments: List of (start_time, end_time) tuples.
        max_duration: Maximum duration per output clip in seconds.

    Returns:
        List of (start_time, end_time, scene_index, sub_index) tuples.
        sub_index is 0 for the first chunk, 1 for the second, etc.
        If no subdivision was needed, sub_index is -1.
    """
    result: list[tuple[float, float, int, int]] = []

    for scene_idx, (start, end) in enumerate(segments):
        duration = end - start
        if duration <= max_duration:
            # Fits in one clip — sub_index -1 means "no subdivide"
            result.append((start, end, scene_idx, -1))
        else:
            # Split into chunks of max_duration
            t = start
            sub_idx = 0
            while t < end - 0.05:  # 50ms tolerance to avoid tiny tail clips
                chunk_end = min(t + max_duration, end)
                result.append((t, chunk_end, scene_idx, sub_idx))
                t = chunk_end
                sub_idx += 1

    return result


def _encode_segments(
    path: Path,
    segments: list[tuple[float, float]],
    output_dir: Path,
    video_config: VideoConfig,
    label: str = "scenes",
) -> list[ClipInfo]:
    """Encode a list of (start, end) segments from a source video.

    Shared logic used by both split_video_at_scenes (boundary-based)
    and split_video_segments (direct timestamps from triage manifest).

    Handles max_frames subdivision, 4n+1 frame trimming, and naming.

    Args:
        path: Resolved path to the source video.
        segments: List of (start_time, end_time) tuples.
        output_dir: Resolved output directory.
        video_config: Target video specs.
        label: Label for summary print (e.g. "scenes" or "triage segments").

    Returns:
        List of ClipInfo for produced clips.
    """
    scene_count = len(segments)

    # Subdivide long segments if max_frames is set
    if video_config.max_frames:
        max_duration = video_config.max_frames / video_config.fps
        chunks = _subdivide_segments(segments, max_duration)
        if len(chunks) > scene_count:
            print(f"  {scene_count} {label} -> {len(chunks)} clips "
                  f"(max {video_config.max_frames} frames per clip)")
    else:
        chunks = [(s, e, i, -1) for i, (s, e) in enumerate(segments)]

    stem = path.stem
    results: list[ClipInfo] = []
    total = len(chunks)
    clip_idx = 0

    for start, end, scene_idx, sub_idx in chunks:
        segment_duration = end - start
        estimated_frames = int(segment_duration * video_config.fps)

        # Skip segments too short for training
        if estimated_frames < 5:
            continue

        target_fc = nearest_valid_frame_count(estimated_frames, "down")
        if target_fc < 5:
            continue

        # Name the output clip
        ext = path.suffix.lower() if path.suffix else ".mov"
        if sub_idx >= 0:
            # Subdivided scene: scene000a, scene000b, ...
            sub_letter = chr(ord('a') + sub_idx) if sub_idx < 26 else f"_{sub_idx}"
            output_file = output_dir / f"{stem}_scene{scene_idx:03d}{sub_letter}{ext}"
        else:
            output_file = output_dir / f"{stem}_scene{scene_idx:03d}{ext}"

        clip_idx += 1
        print(f"  [{clip_idx}/{total}] {start:.1f}s-{end:.1f}s -> {output_file.name}")

        cmd = _build_encode_cmd(
            source=path,
            output=output_file,
            video_config=video_config,
            target_frame_count=target_fc,
            start_time=start,
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            print(f"           FAILED: ffmpeg timed out")
            continue
        except FileNotFoundError:
            raise FFmpegNotFoundError("ffmpeg")

        if result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else "unknown"
            print(f"           FAILED: {stderr[-100:]}")
            continue

        # Probe output
        try:
            out_meta = probe_video(output_file)
            clip_info = ClipInfo(
                source=path,
                output=output_file,
                frame_count=out_meta.frame_count,
                duration=out_meta.duration,
                width=out_meta.width,
                height=out_meta.height,
                fps=out_meta.fps,
                was_reencoded=True,
                trimmed_frames=max(0, estimated_frames - target_fc),
                scene_index=scene_idx,
            )
            results.append(clip_info)
            print(f"           -> {clip_info.frame_count} frames, "
                  f"{clip_info.duration:.1f}s")
        except Exception as e:
            print(f"           FAILED to verify output: {e}")

    print(f"\nDone: {len(results)} clips from {scene_count} {label}.")
    return results


def split_video_at_scenes(
    path: str | Path,
    scenes: list[SceneBoundary],
    output_dir: str | Path,
    video_config: VideoConfig,
) -> list[ClipInfo]:
    """Split a long video at scene boundaries and normalize each segment.

    Takes a video with detected scene cuts and splits it into individual
    clips, one per scene. Each clip is normalized to match the video config.

    When max_frames is set (default: 81), scenes longer than that are
    subdivided into shorter clips. A 300-frame scene becomes four clips
    of 81+81+81+57 frames.

    Clips that are too short after normalization (< 5 frames) are skipped.

    Args:
        path: Path to the source video.
        scenes: Detected scene boundaries (from detect_scenes).
        output_dir: Directory for output clips.
        video_config: Target video specs.

    Returns:
        List of ClipInfo for produced clips.

    Raises:
        FFmpegNotFoundError: if ffmpeg is not in PATH.
        SplitError: if ffmpeg fails.
    """
    _check_ffmpeg()
    path = Path(path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get source metadata for video duration
    source_meta = probe_video(path)

    # Build list of scene segments: (start_time, end_time)
    # First scene starts at 0, last scene goes to end of video
    boundaries_sec = [0.0] + [s.timecode for s in scenes] + [source_meta.duration]
    segments = [
        (boundaries_sec[i], boundaries_sec[i + 1])
        for i in range(len(boundaries_sec) - 1)
    ]

    return _encode_segments(path, segments, output_dir, video_config)


def split_video_segments(
    path: str | Path,
    segments: list[tuple[float, float]],
    output_dir: str | Path,
    video_config: VideoConfig,
) -> list[ClipInfo]:
    """Split a video at pre-computed timestamp segments.

    Used by filtered ingest (--triage) to split only the scenes
    identified by triage, using timestamps read directly from the
    manifest. No scene detection needed — the timestamps come from
    the triage step.

    Args:
        path: Path to the source video.
        segments: List of (start_time, end_time) tuples to extract.
        output_dir: Directory for output clips.
        video_config: Target video specs.

    Returns:
        List of ClipInfo for produced clips.

    Raises:
        FFmpegNotFoundError: if ffmpeg is not in PATH.
        SplitError: if ffmpeg fails.
    """
    _check_ffmpeg()
    path = Path(path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    return _encode_segments(
        path, segments, output_dir, video_config, label="triage segments"
    )
