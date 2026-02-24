"""Reference image extraction from video clips.

Standalone tool that extracts one reference image per video clip for
I2V training. Reference images are VAE-encoded and concatenated with
noisy latents during training, so lossless PNG output is critical —
JPEG artifacts would corrupt the latent representation.

Three extraction strategies:
  - first_frame: frame 0 (standard I2V reference, fast, deterministic)
  - best_frame: sample N frames, pick the sharpest by Laplacian variance
  - user_selected: read frame numbers from a JSON manifest

Also handles mixed datasets containing both video clips and still images.
For image files, the image is copied/converted to PNG as-is — a still
image IS its own reference.

Usage:
    from klippbok.video.extract import extract_directory, ExtractionConfig

    config = ExtractionConfig(strategy="first_frame")
    report = extract_directory("clips/", "refs/", config)

CLI:
    python -m klippbok.video extract clips/ --output refs/
    python -m klippbok.video extract clips/ --output refs/ --strategy best_frame
    python -m klippbok.video extract clips/ --template selections.json
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2

from klippbok.video.errors import ExtractionError, FFmpegNotFoundError
from klippbok.video.extract_models import (
    ExtractionConfig,
    ExtractionReport,
    ExtractionResult,
    ExtractionStrategy,
)
from klippbok.video.image_quality import compute_sharpness, is_blank

# File extensions recognized as video or image
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def _check_ffmpeg() -> None:
    """Verify ffmpeg is available in PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        raise FFmpegNotFoundError("ffmpeg")


def _run_ffmpeg(cmd: list[str], source: str) -> None:
    """Run an ffmpeg command and raise ExtractionError on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise ExtractionError(source, "ffmpeg timed out after 60 seconds")
    except FileNotFoundError:
        raise FFmpegNotFoundError("ffmpeg")

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "unknown error"
        # Extract useful error lines from ffmpeg verbose output
        error_lines = [
            line for line in stderr.split("\n")
            if any(kw in line.lower() for kw in ("error", "invalid", "no such"))
        ]
        detail = error_lines[-1] if error_lines else stderr[-200:]
        raise ExtractionError(source, detail)


def extract_first_frame(
    video_path: str | Path,
    output_path: str | Path,
) -> ExtractionResult:
    """Extract frame 0 from a video as a lossless PNG.

    The first frame is the standard reference image for I2V training.
    This is deterministic, fast (reads only the first frame), and
    matches how I2V models are typically conditioned.

    Args:
        video_path: Path to the source video file.
        output_path: Path for the output PNG file.

    Returns:
        ExtractionResult with metadata about what was extracted.

    Raises:
        FFmpegNotFoundError: if ffmpeg is not in PATH.
        ExtractionError: if ffmpeg fails to extract the frame.
    """
    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-frames:v", "1",      # extract exactly one frame
        "-update", "1",        # overwrite single output file
        str(output_path),
    ]

    _run_ffmpeg(cmd, str(video_path))

    # Compute sharpness of the extracted frame
    sharpness = None
    try:
        sharpness = compute_sharpness(output_path)
    except (ValueError, FileNotFoundError):
        pass

    return ExtractionResult(
        source=video_path,
        output=output_path,
        frame_number=0,
        strategy=ExtractionStrategy.FIRST_FRAME,
        sharpness=sharpness,
        source_type="video",
    )


def extract_frame_at(
    video_path: str | Path,
    output_path: str | Path,
    frame_number: int | None = None,
    timestamp: float | None = None,
) -> ExtractionResult:
    """Extract a specific frame from a video by frame number or timestamp.

    Exactly one of frame_number or timestamp must be provided.
    Frame numbers are 0-based. Timestamps are in seconds.

    WHY both frame_number and timestamp: frame_number is more precise
    (exact frame), but timestamp is more user-friendly (seconds into
    the clip). The JSON manifest uses frame numbers; interactive use
    might prefer timestamps.

    Args:
        video_path: Path to the source video file.
        output_path: Path for the output PNG file.
        frame_number: 0-based frame index to extract.
        timestamp: Time in seconds to extract frame from.

    Returns:
        ExtractionResult with metadata.

    Raises:
        ValueError: if neither or both of frame_number/timestamp are given.
        FFmpegNotFoundError: if ffmpeg is not in PATH.
        ExtractionError: if ffmpeg fails.
    """
    if (frame_number is None) == (timestamp is None):
        raise ValueError(
            "Provide exactly one of frame_number or timestamp, not both."
        )

    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frame_number is not None:
        # Use the select filter to pick an exact frame by index
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"select=eq(n\\,{frame_number})",
            "-frames:v", "1",
            "-update", "1",
            str(output_path),
        ]
    else:
        # Seek to timestamp, then grab one frame
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{timestamp:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-update", "1",
            str(output_path),
        ]

    _run_ffmpeg(cmd, str(video_path))

    sharpness = None
    try:
        sharpness = compute_sharpness(output_path)
    except (ValueError, FileNotFoundError):
        pass

    actual_frame = frame_number if frame_number is not None else None

    return ExtractionResult(
        source=video_path,
        output=output_path,
        frame_number=actual_frame,
        strategy=ExtractionStrategy.USER_SELECTED,
        sharpness=sharpness,
        source_type="video",
    )


def extract_best_frame(
    video_path: str | Path,
    output_path: str | Path,
    sample_count: int = 10,
) -> ExtractionResult:
    """Sample N frames from a video and keep the sharpest one.

    Extracts frames evenly distributed across the video duration,
    measures each one's Laplacian variance (sharpness), and keeps
    the sharpest. On ties, earlier frames are preferred (more likely
    to be a clean starting point for I2V).

    WHY this matters: some clips start with a fade-in, motion blur,
    or a partially obscured frame. The "best frame" strategy finds
    the clearest frame for VAE encoding.

    Args:
        video_path: Path to the source video file.
        output_path: Path for the output PNG file.
        sample_count: Number of frames to sample. More = better pick
            but slower. Default 10 is a good balance.

    Returns:
        ExtractionResult with the frame number and sharpness of the pick.

    Raises:
        FFmpegNotFoundError: if ffmpeg is not in PATH.
        ExtractionError: if ffmpeg fails.
    """
    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract sample_count frames to a temp directory
    # Use the select filter to pick evenly-spaced frames.
    # The expression: not(mod(n, total/count)) picks every (total/count)th frame
    # But we don't know total frames here without probing first.
    # Simpler approach: use fps filter to sample at a rate that gives ~N frames.

    # We need the duration to compute the sampling rate
    try:
        duration_result = subprocess.run(
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
        duration = float(duration_result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        raise ExtractionError(str(video_path), "Could not determine video duration")

    if duration <= 0:
        raise ExtractionError(str(video_path), "Video has zero or negative duration")

    # Calculate fps that gives approximately sample_count frames
    # Add a small buffer to ensure we get enough
    sample_fps = max(sample_count / duration, 1.0)

    # Extract candidate frames to temp directory
    candidates_dir = output_path.parent / f"_candidates_{output_path.stem}"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={sample_fps:.4f}",
        "-frames:v", str(sample_count),
        str(candidates_dir / "frame_%04d.png"),
    ]

    _run_ffmpeg(cmd, str(video_path))

    # Step 2: Score each candidate by sharpness
    candidates = sorted(candidates_dir.glob("frame_*.png"))

    if not candidates:
        # Clean up and fall back to first frame
        _cleanup_dir(candidates_dir)
        return extract_first_frame(video_path, output_path)

    best_path = candidates[0]
    best_sharpness = -1.0
    best_index = 0

    for i, candidate in enumerate(candidates):
        try:
            sharpness = compute_sharpness(candidate)
            # On tie, prefer earlier frame (already established by iteration order)
            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_path = candidate
                best_index = i
        except (ValueError, FileNotFoundError):
            continue

    # Step 3: Copy the winner to the output path and clean up
    import shutil
    shutil.copy2(str(best_path), str(output_path))
    _cleanup_dir(candidates_dir)

    # Estimate the actual frame number based on sampling position
    # This is approximate — the exact frame depends on fps filter behavior
    estimated_frame = int(best_index * (duration * 16) / max(len(candidates), 1))

    return ExtractionResult(
        source=video_path,
        output=output_path,
        frame_number=estimated_frame,
        strategy=ExtractionStrategy.BEST_FRAME,
        sharpness=best_sharpness if best_sharpness >= 0 else None,
        source_type="video",
    )


def copy_image_as_reference(
    image_path: str | Path,
    output_path: str | Path,
) -> ExtractionResult:
    """Copy/convert a still image to PNG for use as a reference.

    For mixed datasets that contain both video clips and still images.
    A still image IS its own reference — for T2V it's a single-frame
    training target, for I2V it's both target and reference.

    If the source is already PNG, it's copied directly. Other formats
    (JPG, BMP, TIFF, WebP) are converted to PNG via OpenCV to ensure
    lossless output.

    Args:
        image_path: Path to the source image file.
        output_path: Path for the output PNG file.

    Returns:
        ExtractionResult with source_type="image".

    Raises:
        FileNotFoundError: if the source image doesn't exist.
        ValueError: if the file can't be read as an image.
    """
    image_path = Path(image_path).resolve()
    output_path = Path(output_path).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and re-save as PNG (handles format conversion)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(
            f"Cannot read image '{image_path}'. "
            f"File may be corrupted or not a supported format."
        )

    cv2.imwrite(str(output_path), img)

    # Score the image
    sharpness = None
    try:
        sharpness = compute_sharpness(output_path)
    except (ValueError, FileNotFoundError):
        pass

    return ExtractionResult(
        source=image_path,
        output=output_path,
        frame_number=None,
        strategy=None,
        sharpness=sharpness,
        source_type="image",
    )


def extract_reference_image(
    source_path: str | Path,
    output_path: str | Path,
    config: ExtractionConfig | None = None,
) -> ExtractionResult:
    """Extract or copy a reference image from a source file.

    Unified dispatch: detects whether the source is a video or image,
    then routes to the appropriate extraction or copy function.
    Handles skip-if-exists logic.

    Args:
        source_path: Path to the source file (video or image).
        output_path: Path for the output PNG.
        config: Extraction configuration. Defaults to first_frame strategy.

    Returns:
        ExtractionResult with extraction metadata.
    """
    if config is None:
        config = ExtractionConfig()

    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()

    # Skip if output exists and overwrite is False
    if output_path.exists() and not config.overwrite:
        return ExtractionResult(
            source=source_path,
            output=output_path,
            skipped=True,
            source_type=_classify_file(source_path),
        )

    # Route based on file type
    suffix = source_path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return copy_image_as_reference(source_path, output_path)
    elif suffix in VIDEO_EXTENSIONS:
        if config.strategy == ExtractionStrategy.FIRST_FRAME:
            return extract_first_frame(source_path, output_path)
        elif config.strategy == ExtractionStrategy.BEST_FRAME:
            return extract_best_frame(
                source_path, output_path, sample_count=config.sample_count
            )
        else:
            # USER_SELECTED without a manifest — fall back to first frame
            return extract_first_frame(source_path, output_path)
    else:
        return ExtractionResult(
            source=source_path,
            success=False,
            error=(
                f"Unsupported file type '{suffix}'. "
                f"Expected video ({', '.join(sorted(VIDEO_EXTENSIONS))}) "
                f"or image ({', '.join(sorted(IMAGE_EXTENSIONS))})."
            ),
        )


def extract_directory(
    source_dir: str | Path,
    output_dir: str | Path,
    config: ExtractionConfig | None = None,
) -> ExtractionReport:
    """Extract reference images for all video/image files in a directory.

    Processes every video and image file in the source directory,
    producing one stem-matched PNG per source file in the output directory.
    Progress is printed to console. Failed files are skipped with warnings.

    Output naming: clip_001.mp4 -> clip_001.png, still_007.jpg -> still_007.png

    Also writes a reference_images.json manifest in the output directory.

    Args:
        source_dir: Directory containing video clips and/or images.
        output_dir: Directory for extracted reference PNGs.
        config: Extraction configuration.

    Returns:
        ExtractionReport with per-file results and summary.
    """
    if config is None:
        config = ExtractionConfig()

    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video and image files
    source_files = sorted(
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (VIDEO_EXTENSIONS | IMAGE_EXTENSIONS)
    )

    if not source_files:
        print(f"No video or image files found in {source_dir}")
        return ExtractionReport()

    results: list[ExtractionResult] = []
    total = len(source_files)

    for i, source_file in enumerate(source_files, 1):
        output_file = output_dir / (source_file.stem + ".png")
        file_type = _classify_file(source_file)
        print(f"  [{i}/{total}] {source_file.name} ({file_type})")

        try:
            result = extract_reference_image(source_file, output_file, config)
            results.append(result)

            if result.skipped:
                print(f"           skipped (already exists)")
            elif result.success:
                sharpness_str = f", sharpness={result.sharpness:.1f}" if result.sharpness else ""
                blank_warn = ""
                if result.output and result.sharpness is not None and result.sharpness < 5.0:
                    blank_warn = " [WARNING: blank/uniform frame]"
                print(f"           -> {output_file.name}{sharpness_str}{blank_warn}")
            else:
                print(f"           FAILED: {result.error}")
        except Exception as e:
            results.append(ExtractionResult(
                source=source_file,
                success=False,
                error=str(e),
                source_type=file_type,
            ))
            print(f"           FAILED: {e}")

    report = ExtractionReport(results=results)

    # Print summary
    print(f"\nDone: {report.succeeded} extracted, "
          f"{report.skipped} skipped, "
          f"{report.failed} failed "
          f"(of {report.total} files)")
    if report.videos > 0:
        print(f"  Videos: {report.videos}")
    if report.images > 0:
        print(f"  Images: {report.images} (pass-through)")

    # Write manifest
    _write_manifest(output_dir, results)

    return report


def generate_selection_template(
    source_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Generate a JSON template for user-selected frame extraction.

    Scans the source directory for video files and writes a JSON file
    with a default frame number (0) for each clip. The user edits this
    file to specify which frame to extract from each clip, then passes
    it to extract_from_selections().

    Image files are listed with "auto" (they'll be copied as-is).

    Args:
        source_dir: Directory containing video clips.
        output_path: Path to write the JSON template.

    Returns:
        Path to the written template file.
    """
    source_dir = Path(source_dir).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selections: dict[str, dict] = {}

    source_files = sorted(
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (VIDEO_EXTENSIONS | IMAGE_EXTENSIONS)
    )

    for source_file in source_files:
        if source_file.suffix.lower() in VIDEO_EXTENSIONS:
            selections[source_file.name] = {"frame": 0}
        else:
            # Image files get auto-copied, no frame selection needed
            selections[source_file.name] = {"auto": True}

    output_path.write_text(json.dumps(selections, indent=2))
    print(f"Selection template written: {output_path}")
    print(f"  {len(selections)} entries. Edit frame numbers, then run:")
    print(f"  python -m klippbok.video extract {source_dir} --output <dir> --selections {output_path}")

    return output_path


def extract_from_selections(
    source_dir: str | Path,
    output_dir: str | Path,
    selections_path: str | Path,
) -> ExtractionReport:
    """Extract reference images using a user-edited selections manifest.

    Reads a JSON file mapping filenames to frame numbers, then extracts
    the specified frame from each video. Image files marked "auto" are
    copied as-is.

    The JSON format (produced by generate_selection_template):
        {
          "clip_001.mp4": {"frame": 42},
          "clip_002.mp4": {"frame": 0},
          "still_003.jpg": {"auto": true}
        }

    Args:
        source_dir: Directory containing the source files.
        output_dir: Directory for extracted reference PNGs.
        selections_path: Path to the JSON selections manifest.

    Returns:
        ExtractionReport with per-file results.
    """
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    selections_path = Path(selections_path).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load selections
    selections = json.loads(selections_path.read_text())

    results: list[ExtractionResult] = []
    total = len(selections)

    for i, (filename, spec) in enumerate(selections.items(), 1):
        source_file = source_dir / filename
        stem = Path(filename).stem
        output_file = output_dir / (stem + ".png")

        print(f"  [{i}/{total}] {filename}")

        if not source_file.exists():
            results.append(ExtractionResult(
                source=source_file,
                success=False,
                error=f"Source file not found: {source_file}",
            ))
            print(f"           FAILED: source file not found")
            continue

        try:
            if spec.get("auto"):
                # Image pass-through
                result = copy_image_as_reference(source_file, output_file)
            else:
                # Video: extract at specified frame
                frame_num = spec.get("frame", 0)
                result = extract_frame_at(
                    source_file, output_file, frame_number=frame_num
                )
            results.append(result)

            if result.success:
                frame_str = f" frame {result.frame_number}" if result.frame_number is not None else ""
                sharpness_str = f", sharpness={result.sharpness:.1f}" if result.sharpness else ""
                print(f"           -> {output_file.name}{frame_str}{sharpness_str}")
            else:
                print(f"           FAILED: {result.error}")
        except Exception as e:
            results.append(ExtractionResult(
                source=source_file,
                success=False,
                error=str(e),
                source_type=_classify_file(source_file),
            ))
            print(f"           FAILED: {e}")

    report = ExtractionReport(results=results)
    print(f"\nDone: {report.succeeded} extracted, {report.failed} failed "
          f"(of {report.total} files)")

    _write_manifest(output_dir, results)
    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_file(path: Path) -> str:
    """Classify a file as 'video' or 'image' by extension."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    return "video"


def _write_manifest(output_dir: Path, results: list[ExtractionResult]) -> None:
    """Write reference_images.json manifest to the output directory."""
    successful = [r for r in results if r.success and not r.skipped and r.output]
    if not successful:
        return

    manifest = []
    for r in successful:
        entry: dict = {
            "source": str(r.source),
            "output": str(r.output),
            "source_type": r.source_type,
        }
        if r.frame_number is not None:
            entry["frame_number"] = r.frame_number
        if r.strategy is not None:
            entry["strategy"] = r.strategy.value
        if r.sharpness is not None:
            entry["sharpness"] = round(r.sharpness, 2)
        manifest.append(entry)

    manifest_path = output_dir / "reference_images.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written: {manifest_path}")


def _cleanup_dir(path: Path) -> None:
    """Remove a directory and all its contents."""
    import shutil
    try:
        shutil.rmtree(str(path))
    except OSError:
        pass
