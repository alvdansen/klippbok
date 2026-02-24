"""CLI for Klippbok video tools.

Usage:
    python -m klippbok.video scan <dir>                  — probe + validate + report
    python -m klippbok.video ingest <video|dir> --output <dir>  — scene detect + split + normalize
    python -m klippbok.video normalize <dir> --output <dir> — batch normalize pre-cut clips
    python -m klippbok.video caption <dir> --provider gemini — caption clips via VLM
    python -m klippbok.video audit <dir> --provider gemini   — audit existing captions
    python -m klippbok.video score <dir>                 — score caption quality (no API needed)
    python -m klippbok.video extract <dir> --output <dir>   — extract reference images from clips
    python -m klippbok.video triage <dir> --concepts <dir>  — match clips against reference images

All commands accept --config <path> to load a klippbok_data.yaml config file.
Without --config, sensible defaults are used (16fps, 480p, auto frame count).

This is a thin wrapper around the library modules — all logic lives in
probe.py, validate.py, scene.py, split.py, extract.py, and the caption package.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_scan(args: argparse.Namespace) -> None:
    """Scan a directory of clips: probe + validate + report."""
    from klippbok.video.probe import probe_directory
    from klippbok.video.validate import (
        format_scan_report,
        format_scan_report_verbose,
        validate_directory,
    )

    directory = Path(args.directory)
    video_config = _load_video_config(args.config, getattr(args, "fps", None))

    print(f"Scanning: {directory}")
    print()

    # Probe all videos first
    metadata_list = probe_directory(directory)
    if not metadata_list:
        print(f"No video files found in {directory}")
        return

    # Validate against config
    report = validate_directory(directory, video_config, metadata_list=metadata_list)

    # Print report — verbose (-v) shows every clip, default groups by pattern
    if getattr(args, "verbose", False):
        print(format_scan_report_verbose(report, video_config))
    else:
        print(format_scan_report(report, video_config))


def _ingest_single_video(
    video_path: Path,
    output_dir: Path,
    video_config,
    threshold: float,
    triage_segments: list[tuple[float, float]] | None = None,
) -> list:
    """Ingest a single video: scene detect + split + normalize.

    Args:
        video_path: Path to the source video.
        output_dir: Output directory for clips.
        video_config: VideoConfig with target specs.
        threshold: Scene detection threshold.
        triage_segments: If provided, skip scene detection and split
            at these (start_time, end_time) timestamps directly.
            Used by --triage filtered ingest.

    Returns:
        List of ClipInfo from split functions.
    """
    from klippbok.video.probe import probe_video

    # Probe source
    meta = probe_video(video_path)
    print(f"Source: {meta.display_resolution}, {meta.fps}fps, "
          f"{meta.frame_count} frames, {meta.duration:.1f}s")
    print()

    if triage_segments is not None:
        # Filtered ingest — timestamps from triage manifest, no scene detection
        from klippbok.video.split import split_video_segments

        print(f"Triage: splitting {len(triage_segments)} pre-identified scene(s)")
        print()
        print("Splitting and normalizing...")
        return split_video_segments(
            video_path, triage_segments, output_dir, video_config
        )
    else:
        # Normal ingest — detect scenes, split everything
        from klippbok.video.scene import detect_scenes
        from klippbok.video.split import split_video_at_scenes

        print("Detecting scenes...")
        scenes = detect_scenes(video_path, threshold=threshold)
        total_segments = len(scenes) + 1
        print(f"Found {len(scenes)} scene cut(s) -> {total_segments} segments")
        print()

        print("Splitting and normalizing...")
        return split_video_at_scenes(
            video_path, scenes, output_dir, video_config
        )


def _load_triage_manifest(
    manifest_path: str | Path,
) -> dict[str, list[tuple[float, float]]]:
    """Load a scene triage manifest and extract included scene timestamps.

    Reads the JSON manifest produced by triage_videos() and returns a
    mapping from video path to a list of (start_time, end_time) segments
    marked as 'include: true'. These timestamps are passed directly to
    split_video_segments() — no scene detection needed.

    Args:
        manifest_path: Path to scene_triage_manifest.json.

    Returns:
        Dict mapping video path strings to lists of (start, end) tuples.

    Raises:
        FileNotFoundError: if manifest doesn't exist.
        ValueError: if manifest format is invalid.
    """
    manifest_path = Path(manifest_path).resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Triage manifest not found: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    if data.get("triage_mode") != "scene":
        raise ValueError(
            f"Expected triage_mode='scene' in manifest, "
            f"got '{data.get('triage_mode')}'. "
            f"This doesn't look like a scene triage manifest."
        )

    result: dict[str, list[tuple[float, float]]] = {}
    for video in data.get("videos", []):
        video_path = video["path"]
        segments: list[tuple[float, float]] = []
        for scene in video.get("scenes", []):
            if scene.get("include", False):
                segments.append((scene["start_time"], scene["end_time"]))
        if segments:
            result[video_path] = segments

    return result


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest one or more videos: scene detect + split + normalize.

    Accepts either a single video file or a directory of videos.
    When given a directory, all video files are ingested into the
    same output directory.
    """
    input_path = Path(args.video)
    output_dir = Path(args.output)
    video_config = _load_video_config(args.config)
    threshold = args.threshold
    triage_manifest_path = getattr(args, "triage", None)

    # Apply --max-frames override
    max_frames_override = getattr(args, "max_frames", None)
    if max_frames_override is not None:
        if max_frames_override == 0:
            # 0 means no limit
            video_config = video_config.model_copy(update={"max_frames": None})
        else:
            video_config = video_config.model_copy(update={"max_frames": max_frames_override})

    max_label = video_config.max_frames or "unlimited"
    print(f"Output: {output_dir}")
    print(f"Config: {video_config.fps}fps, {video_config.resolution}p, "
          f"max {max_label} frames/clip")
    print(f"Scene detection threshold: {threshold}")

    # Load triage manifest if provided
    triage_filter: dict[str, list[tuple[float, float]]] | None = None
    if triage_manifest_path:
        print(f"Triage manifest: {triage_manifest_path}")
        triage_filter = _load_triage_manifest(triage_manifest_path)
        total_scenes = sum(len(s) for s in triage_filter.values())
        print(f"  {len(triage_filter)} video(s), {total_scenes} scene(s) to extract")

    print()

    # Validate input exists
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Single file or directory?
    video_extensions = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

    if input_path.is_dir():
        # Batch mode: ingest all videos in directory
        video_files = sorted(
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        )
        if not video_files:
            print(f"No video files found in {input_path}")
            return

        # If triage filter active, only process videos that have matching scenes
        if triage_filter is not None:
            # Match by resolving paths or by filename
            filtered_files = []
            for vf in video_files:
                resolved = str(vf.resolve())
                if resolved in triage_filter:
                    filtered_files.append((vf, triage_filter[resolved]))
                else:
                    # Try matching by filename (manifest might have different path)
                    for manifest_path_str, segments in triage_filter.items():
                        if Path(manifest_path_str).name == vf.name:
                            filtered_files.append((vf, segments))
                            break

            if not filtered_files:
                print("No videos in this directory match the triage manifest.")
                return

            print(f"Filtered ingest: {len(filtered_files)}/{len(video_files)} "
                  f"video(s) have matching scenes")
            print()

            all_clips: list = []
            for i, (video_path, segments) in enumerate(filtered_files, 1):
                print(f"=== [{i}/{len(filtered_files)}] {video_path.name} "
                      f"({len(segments)} scene(s)) ===")
                try:
                    clips = _ingest_single_video(
                        video_path, output_dir, video_config, threshold,
                        triage_segments=segments,
                    )
                    all_clips.extend(clips)
                except Exception as e:
                    print(f"  FAILED: {e}")
                print()

            print(f"\nFiltered ingest complete: {len(all_clips)} clips")
        else:
            print(f"Batch ingest: {len(video_files)} video(s) in {input_path}")
            print()

            all_clips = []
            for i, video_path in enumerate(video_files, 1):
                print(f"=== [{i}/{len(video_files)}] {video_path.name} ===")
                try:
                    clips = _ingest_single_video(
                        video_path, output_dir, video_config, threshold
                    )
                    all_clips.extend(clips)
                except Exception as e:
                    print(f"  FAILED: {e}")
                print()

            print(f"\nBatch complete: {len(all_clips)} clips from {len(video_files)} video(s)")

        # Caption if requested
        if args.caption and all_clips:
            _caption_clips(output_dir, args)

        # Write manifest
        _write_manifest(output_dir, all_clips)
    else:
        # Single file mode
        triage_segments = None
        if triage_filter is not None:
            resolved = str(input_path.resolve())
            if resolved in triage_filter:
                triage_segments = triage_filter[resolved]
            else:
                # Try by filename
                for manifest_path_str, segments in triage_filter.items():
                    if Path(manifest_path_str).name == input_path.name:
                        triage_segments = segments
                        break
            if triage_segments:
                print(f"Filtered ingest: {len(triage_segments)} matching scene(s)")

        print(f"Ingesting: {input_path}")
        clips = _ingest_single_video(
            input_path, output_dir, video_config, threshold,
            triage_segments=triage_segments,
        )

        # Caption if requested
        if args.caption and clips:
            _caption_clips(output_dir, args)

        # Write manifest
        _write_manifest(output_dir, clips)


def cmd_normalize(args: argparse.Namespace) -> None:
    """Normalize pre-cut clips in a directory."""
    from klippbok.video.split import normalize_directory

    source_dir = Path(args.directory)
    output_dir = Path(args.output)
    video_config = _load_video_config(args.config, getattr(args, "fps", None))

    print(f"Normalizing: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Config: {video_config.fps}fps, {video_config.resolution}p, "
          f"frame_count={video_config.frame_count}")
    print()

    output_format = getattr(args, "format", None)
    clips = normalize_directory(source_dir, output_dir, video_config, output_format=output_format)

    # Write manifest
    _write_manifest(output_dir, clips)


def cmd_caption(args: argparse.Namespace) -> None:
    """Caption clips in a directory using a VLM backend."""
    _caption_clips(Path(args.directory), args)


def cmd_score(args: argparse.Namespace) -> None:
    """Score caption quality in a directory (no API calls needed)."""
    from klippbok.caption.scoring import ScoringConfig, format_score_report, score_directory

    directory = Path(args.directory)
    config = ScoringConfig()

    print(f"Scoring captions in: {directory}")
    print()

    scores = score_directory(directory, config)

    if not scores:
        print("No .txt caption files found.")
        return

    print(format_score_report(scores))


def cmd_extract(args: argparse.Namespace) -> None:
    """Extract reference images from video clips (and copy still images)."""
    from klippbok.video.extract import (
        extract_directory,
        extract_from_selections,
        generate_selection_template,
    )
    from klippbok.video.extract_models import ExtractionConfig, ExtractionStrategy

    directory = Path(args.directory)

    # Mode 1: Generate selection template
    if args.template:
        generate_selection_template(directory, args.template)
        return

    # Mode 2: Extract from user-edited selections
    if args.selections:
        if not args.output:
            print("Error: --output is required when using --selections", file=sys.stderr)
            sys.exit(1)
        output_dir = Path(args.output)
        print(f"Extracting from selections: {args.selections}")
        print(f"Source: {directory}")
        print(f"Output: {output_dir}")
        print()
        extract_from_selections(directory, output_dir, args.selections)
        return

    # Mode 3: Automatic extraction (first_frame or best_frame)
    if not args.output:
        print("Error: --output is required for extraction", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    strategy = ExtractionStrategy(args.strategy)
    config = ExtractionConfig(
        strategy=strategy,
        sample_count=args.samples,
        overwrite=args.overwrite,
    )

    print(f"Extracting reference images")
    print(f"Source: {directory}")
    print(f"Output: {output_dir}")
    print(f"Strategy: {strategy.value}")
    if strategy == ExtractionStrategy.BEST_FRAME:
        print(f"Samples: {config.sample_count}")
    print()

    extract_directory(directory, output_dir, config)


def cmd_triage(args: argparse.Namespace) -> None:
    """Match clips against concept reference images using CLIP."""
    from klippbok.triage import organize_clips, triage_clips
    from klippbok.triage.models import VideoTriageReport

    directory = Path(args.directory)
    concepts_dir = Path(args.concepts)

    print(f"Triage: matching clips against references")
    print(f"Clips: {directory}")
    print(f"Concepts: {concepts_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Frames per clip: {args.frames}")
    print(f"Frames per scene: {args.frames_per_scene}")
    print(f"Scene threshold: {args.scene_threshold}")
    if args.organize:
        print(f"Organize: clips will be {'copied' if not args.move else 'moved'} into concept folders")
    print()

    report = triage_clips(
        clips_dir=directory,
        concepts_dir=concepts_dir,
        threshold=args.threshold,
        frames_per_clip=args.frames,
        model_name=args.clip_model,
        output_path=args.output,
        frames_per_scene=args.frames_per_scene,
        scene_threshold=args.scene_threshold,
    )

    # Organize clips into concept folders if requested
    if args.organize:
        if isinstance(report, VideoTriageReport):
            print("\nError: --organize is not supported for raw video triage.")
            print("Scenes must be split first. Run filtered ingest:")
            print(f"  python -m klippbok.video ingest {directory} -o <output> "
                  f"--triage scene_triage_manifest.json")
            sys.exit(1)
        organize_dir = Path(args.organize)
        organize_clips(report, organize_dir, copy=not args.move)


def cmd_audit(args: argparse.Namespace) -> None:
    """Audit existing captions against VLM output."""
    from klippbok.caption import audit_captions
    from klippbok.caption.models import CaptionConfig

    directory = Path(args.directory)
    config = CaptionConfig(
        provider=args.provider,
        use_case=args.use_case,
        audit_mode=args.mode,
    )

    print(f"Auditing captions in: {directory}")
    print(f"Provider: {config.provider}")
    print(f"Mode: {config.audit_mode}")
    print()

    results = audit_captions(directory, config)

    print(f"\nAudit complete: {len(results)} captions reviewed")
    matches = sum(1 for r in results if r.recommendation == "keep")
    print(f"  Keep as-is: {matches}")
    print(f"  Review suggested: {len(results) - matches}")


def _caption_clips(directory: Path, args: argparse.Namespace) -> None:
    """Shared captioning logic for caption and ingest --caption."""
    from klippbok.caption import caption_clips
    from klippbok.caption.models import CaptionConfig

    # Build config kwargs — only include openai fields when relevant
    tags = getattr(args, "tags", None)
    kwargs: dict = dict(
        provider=args.provider,
        use_case=getattr(args, "use_case", None),
        anchor_word=getattr(args, "anchor_word", None),
        secondary_anchors=tags if tags else None,
        overwrite=getattr(args, "overwrite", False),
        caption_fps=getattr(args, "caption_fps", 1),
    )

    if args.provider == "openai":
        kwargs["openai_base_url"] = getattr(args, "base_url", "http://localhost:11434/v1")
        if getattr(args, "model", None):
            kwargs["openai_model"] = args.model

    config = CaptionConfig(**kwargs)

    print(f"Captioning clips in: {directory}")
    print(f"Provider: {config.provider}")
    if config.provider == "openai":
        print(f"Endpoint: {config.openai_base_url}")
        print(f"Model: {config.openai_model}")
    if config.anchor_word:
        print(f"Anchor word: {config.anchor_word}")
    if config.secondary_anchors:
        print(f"Tags: {', '.join(config.secondary_anchors)}")
    if config.caption_fps != 1:
        print(f"Caption FPS: {config.caption_fps}")
    print()

    caption_clips(directory, config)


def _load_video_config(config_path: str | None, fps_override: int | None = None):
    """Load VideoConfig from a klippbok_data.yaml, or use defaults.

    Args:
        config_path: Optional path to klippbok_data.yaml.
        fps_override: If provided, overrides the FPS from config/defaults.
    """
    from klippbok.config.data_schema import VideoConfig

    if config_path:
        from klippbok.config.loader import load_data_config
        data_config = load_data_config(config_path)
        video_config = data_config.video
    else:
        # Sensible defaults (matches Wan training settings)
        video_config = VideoConfig(fps=16, resolution=720, frame_count="auto")

    # Apply FPS override if given
    if fps_override is not None:
        video_config = video_config.model_copy(update={"fps": fps_override})

    # Warn about non-standard FPS
    if video_config.fps != 16:
        print(f"NOTE: Using {video_config.fps} fps (Wan models were trained at 16 fps).")
        print(f"      Non-standard frame rates may produce lower quality results.")
        print()

    return video_config


def _write_manifest(output_dir: Path, clips: list) -> None:
    """Write a JSON manifest of processed clips."""
    if not clips:
        return

    manifest_path = output_dir / "manifest.json"
    manifest = []
    for clip in clips:
        entry = {
            "source": str(clip.source),
            "output": str(clip.output),
            "frame_count": clip.frame_count,
            "duration": round(clip.duration, 3),
            "width": clip.width,
            "height": clip.height,
            "fps": clip.fps,
            "was_reencoded": clip.was_reencoded,
            "trimmed_frames": clip.trimmed_frames,
        }
        if clip.scene_index is not None:
            entry["scene_index"] = clip.scene_index
        manifest.append(entry)

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written: {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="klippbok.video",
        description="Klippbok video ingestion and scene detection tools",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── scan ──
    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan a directory of clips: probe + validate + report",
    )
    scan_parser.add_argument("directory", help="Directory of video clips to scan")
    scan_parser.add_argument(
        "--config", "-c",
        help="Path to klippbok_data.yaml (default: use standard Wan defaults)",
    )
    scan_parser.add_argument(
        "--fps", type=int,
        help="Target frame rate (default: 16 for Wan models)",
    )
    scan_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show full per-clip details instead of grouped summary",
    )

    # ── ingest ──
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest video(s): scene detect + split + normalize",
    )
    ingest_parser.add_argument(
        "video",
        help="Path to a video file or directory of videos",
    )
    ingest_parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for clips",
    )
    ingest_parser.add_argument(
        "--config", "-c",
        help="Path to klippbok_data.yaml",
    )
    ingest_parser.add_argument(
        "--threshold", "-t", type=float, default=27.0,
        help="Scene detection threshold (default: 27.0)",
    )
    ingest_parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Max frames per clip (default: 81 = ~5s at 16fps). Use 0 for no limit.",
    )
    ingest_parser.add_argument(
        "--triage",
        metavar="MANIFEST",
        help="Path to scene_triage_manifest.json — only split scenes marked 'include: true'",
    )
    ingest_parser.add_argument(
        "--caption", action="store_true",
        help="Auto-caption clips after splitting",
    )
    ingest_parser.add_argument(
        "--provider", default="gemini",
        choices=["gemini", "replicate", "openai"],
        help="VLM provider for captioning (default: gemini)",
    )

    # ── normalize ──
    norm_parser = subparsers.add_parser(
        "normalize",
        help="Normalize pre-cut clips in a directory",
    )
    norm_parser.add_argument("directory", help="Directory of clips to normalize")
    norm_parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for normalized clips",
    )
    norm_parser.add_argument(
        "--config", "-c",
        help="Path to klippbok_data.yaml",
    )
    norm_parser.add_argument(
        "--fps", type=int,
        help="Target frame rate (default: 16 for Wan models)",
    )
    norm_parser.add_argument(
        "--format", "-f",
        choices=[".mp4", ".mov", ".mkv"],
        help="Force output format (default: match source file)",
    )

    # ── caption ──
    cap_parser = subparsers.add_parser(
        "caption",
        help="Caption clips using a VLM backend",
    )
    cap_parser.add_argument("directory", help="Directory of video clips")
    cap_parser.add_argument(
        "--provider", "-p", default="gemini",
        choices=["gemini", "replicate", "openai"],
        help="VLM provider (default: gemini)",
    )
    cap_parser.add_argument(
        "--use-case", "-u",
        choices=["character", "style", "motion", "object"],
        help="Use case for prompt selection",
    )
    cap_parser.add_argument(
        "--anchor-word", "-a",
        help="Anchor word to prepend to captions",
    )
    cap_parser.add_argument(
        "--tags", "-t", nargs="+",
        help="Secondary anchor tags the model should mention when relevant (e.g. -t vintage retro)",
    )
    cap_parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing .txt captions",
    )
    cap_parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="Base URL for OpenAI-compatible API (default: Ollama)",
    )
    cap_parser.add_argument(
        "--model",
        help="Model name for OpenAI-compatible backend (default: llama3.2-vision)",
    )
    cap_parser.add_argument(
        "--caption-fps", type=int, default=1,
        help="Frame sampling rate for captioning (default: 1 FPS)",
    )

    # ── score ──
    score_parser = subparsers.add_parser(
        "score",
        help="Score caption quality (no API calls needed)",
    )
    score_parser.add_argument("directory", help="Directory of .txt caption files")

    # ── extract ──
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract reference images from video clips",
    )
    extract_parser.add_argument("directory", help="Directory of video clips / images")
    extract_parser.add_argument(
        "--output", "-o",
        help="Output directory for extracted PNG reference images",
    )
    extract_parser.add_argument(
        "--strategy", "-s",
        default="first_frame",
        choices=["first_frame", "best_frame"],
        help="Frame selection strategy (default: first_frame)",
    )
    extract_parser.add_argument(
        "--samples", type=int, default=10,
        help="Number of frames to sample for best_frame strategy (default: 10)",
    )
    extract_parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing reference images",
    )
    extract_parser.add_argument(
        "--selections",
        help="Path to JSON selections manifest (from --template)",
    )
    extract_parser.add_argument(
        "--template",
        help="Generate a selection template JSON at this path (no extraction)",
    )

    # ── triage ──
    triage_parser = subparsers.add_parser(
        "triage",
        help="Match clips against concept reference images (requires torch + transformers)",
    )
    triage_parser.add_argument("directory", help="Directory of video clips to triage")
    triage_parser.add_argument(
        "--concepts", "-s", required=True,
        help="Path to concepts/ directory with reference images in type subfolders",
    )
    triage_parser.add_argument(
        "--threshold", type=float, default=0.70,
        help="Similarity threshold for matching (0.0-1.0, default: 0.70)",
    )
    triage_parser.add_argument(
        "--frames", type=int, default=5,
        help="Number of frames to sample per clip (default: 5)",
    )
    triage_parser.add_argument(
        "--output", "-o",
        help="Path for triage manifest JSON (default: triage_manifest.json in clips dir)",
    )
    triage_parser.add_argument(
        "--organize",
        metavar="DIR",
        help="Organize clips into concept-named folders under DIR after matching",
    )
    triage_parser.add_argument(
        "--move", action="store_true",
        help="Move files instead of copying when using --organize",
    )
    triage_parser.add_argument(
        "--frames-per-scene", type=int, default=2,
        help="Frames to sample per scene for long videos (default: 2)",
    )
    triage_parser.add_argument(
        "--scene-threshold", type=float, default=27.0,
        help="Scene detection threshold for long videos (default: 27.0)",
    )
    triage_parser.add_argument(
        "--clip-model", default="openai/clip-vit-base-patch32",
        help="CLIP model to use (default: openai/clip-vit-base-patch32)",
    )

    # ── audit ──
    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit existing captions against VLM output",
    )
    audit_parser.add_argument("directory", help="Directory of captioned clips")
    audit_parser.add_argument(
        "--provider", default="gemini",
        choices=["gemini", "replicate", "openai"],
        help="VLM provider (default: gemini)",
    )
    audit_parser.add_argument(
        "--use-case",
        choices=["character", "style", "motion", "object"],
        help="Use case for prompt selection",
    )
    audit_parser.add_argument(
        "--mode", default="report_only",
        choices=["report_only", "save_audit"],
        help="Audit mode: report_only or save_audit (default: report_only)",
    )

    return parser


def main() -> None:
    """Entry point for python -m klippbok.video."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "scan": cmd_scan,
        "ingest": cmd_ingest,
        "normalize": cmd_normalize,
        "caption": cmd_caption,
        "score": cmd_score,
        "audit": cmd_audit,
        "extract": cmd_extract,
        "triage": cmd_triage,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
