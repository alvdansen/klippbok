"""Main triage orchestrator — match clips against concept references.

Wires together concept discovery, frame sampling, CLIP embeddings,
and cross-reference matching into a single pipeline.

Flow:
    1. Discover concept references from concepts/ folder
    2. Load CLIP model and encode all reference images
    3. For each clip:
       a. Sample N frames via ffmpeg
       b. Encode frames with CLIP
       c. Compare against all references
       d. Record matches above threshold
    4. Write triage manifest (JSON)
    5. Print summary

Usage:
    from klippbok.triage import triage_clips

    report = triage_clips(
        clips_dir="path/to/clips",
        concepts_dir="path/to/concepts",
    )
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path

from klippbok.triage.embeddings import CLIPEmbedder, check_clip_available
from klippbok.triage.filters import build_prompt_cache, detect_text_overlays
from klippbok.triage.models import (
    VIDEO_EXTENSIONS,
    ClipMatch,
    ClipTriage,
    ConceptReference,
    SceneTriage,
    TriageReport,
    VideoTriageReport,
)
from klippbok.triage.sampler import (
    _get_duration,
    cleanup_frames,
    sample_clip_frames,
    sample_scene_frames,
)
from klippbok.triage.concepts import discover_concepts, print_concept_summary

# Duration threshold: videos shorter than this are treated as pre-cut clips
# (sample frames directly). Videos this long or longer get scene detection
# first, then per-scene frame sampling. 30s is well above typical training
# clip length (3-5s) and well below raw footage length (minutes+).
LONG_VIDEO_THRESHOLD = 30.0


def _find_clips(directory: Path) -> list[Path]:
    """Find all video clips in a directory, sorted by name."""
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    )


def triage_clips(
    clips_dir: str | Path,
    concepts_dir: str | Path,
    threshold: float = 0.70,
    frames_per_clip: int = 5,
    model_name: str = "openai/clip-vit-base-patch32",
    output_path: str | Path | None = None,
    frames_per_scene: int = 2,
    scene_threshold: float = 27.0,
) -> TriageReport | VideoTriageReport:
    """Run reference-guided triage on all clips in a directory.

    Duration-adaptive: automatically detects whether the directory
    contains short clips (post-ingest, < 30s) or long raw videos
    (>= 30s). Short clips get per-clip frame sampling (existing
    behavior). Long videos get scene detection first, then per-scene
    frame sampling — delegated to triage_videos().

    Args:
        clips_dir: Directory containing video clips to triage.
        concepts_dir: Directory with concept reference images organized
            in type subfolders (e.g. concepts/character/holly.jpg).
        threshold: Minimum cosine similarity for a match (0.0-1.0).
            Default 0.70 is a reasonable starting point — tune based
            on your data. Lower = more matches (more false positives).
        frames_per_clip: Number of frames to sample per clip for
            matching (default: 5). More = better coverage but slower.
        model_name: CLIP model to use (default: openai/clip-vit-base-patch32).
        output_path: Path for the triage manifest JSON. Defaults to
            triage_manifest.json in the clips directory.
        frames_per_scene: Frames per scene for long videos (default: 2).
        scene_threshold: Scene detection threshold for long videos (default: 27.0).

    Returns:
        TriageReport for short clips, or VideoTriageReport for long videos.

    Raises:
        ImportError: if torch or transformers aren't installed.
        FileNotFoundError: if clips_dir or concepts_dir don't exist.
    """
    # Validate dependencies first
    check_clip_available()

    clips_dir = Path(clips_dir).resolve()
    concepts_dir = Path(concepts_dir).resolve()

    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

    # Step 1: Find video files
    clips = _find_clips(clips_dir)
    if not clips:
        print(f"No video files found in {clips_dir}")
        return TriageReport(
            concepts=[],
            threshold=threshold,
            model_name=model_name,
        )

    # Step 1b: Probe durations to decide short-clip vs long-video mode
    print(f"Found {len(clips)} video file(s). Probing durations...")
    clip_durations: list[float] = []
    for clip_path in clips:
        try:
            dur = _get_duration(clip_path)
            clip_durations.append(dur)
        except RuntimeError:
            # Can't probe duration — assume short clip
            clip_durations.append(0.0)

    long_count = sum(1 for d in clip_durations if d >= LONG_VIDEO_THRESHOLD)

    # If majority of videos are long, switch to scene-aware mode
    if long_count > len(clips) / 2:
        print(f"\nDetected raw video input — {long_count}/{len(clips)} videos "
              f"are >= {LONG_VIDEO_THRESHOLD}s")
        print("Running scene-aware triage (detect scenes first, then match)\n")
        return triage_videos(
            video_paths=clips,
            concepts_dir=concepts_dir,
            threshold=threshold,
            frames_per_scene=frames_per_scene,
            scene_threshold=scene_threshold,
            model_name=model_name,
            output_path=output_path,
        )

    # --- Short clip mode (existing behavior) ---

    # Step 2: Discover concepts
    print("Discovering concept references...")
    concepts = discover_concepts(concepts_dir)

    if not concepts:
        print("  No reference images found. Nothing to match against.")
        return TriageReport(threshold=threshold, model_name=model_name)

    print_concept_summary(concepts)
    print()

    print(f"Found {len(clips)} clips and {len(concepts)} reference(s)")
    print()

    # Step 3: Load CLIP model and encode references
    print(f"Loading CLIP model ({model_name})...")
    embedder = CLIPEmbedder(model_name=model_name)

    print("Encoding reference images...")
    ref_embeddings = {}
    for ref in concepts:
        try:
            ref_embeddings[ref] = embedder.encode_image(ref.image_path)
            print(f"  {ref.folder_name}/{ref.name} - OK")
        except Exception as e:
            print(f"  {ref.folder_name}/{ref.name} - FAILED: {e}")
    print()

    if not ref_embeddings:
        print("No reference images could be encoded. Cannot proceed.")
        return TriageReport(
            concepts=concepts,
            threshold=threshold,
            model_name=model_name,
        )

    # Pre-compute text overlay prompt embeddings (once for all clips)
    print("Building text overlay filter...")
    text_prompt_cache = build_prompt_cache(embedder)

    # Step 4: Process each clip
    print(f"Matching clips (threshold: {threshold:.2f}, {frames_per_clip} frames/clip)...")
    print()

    results: list[ClipTriage] = []
    total = len(clips)
    durations: list[float] = []

    for i, clip_path in enumerate(clips, 1):
        start = time.time()

        # Progress with ETA
        if durations:
            avg = sum(durations) / len(durations)
            remaining = (total - i + 1) * avg
            eta = f" (ETA: {remaining:.0f}s)"
        else:
            eta = ""

        print(f"  [{i}/{total}] {clip_path.name}{eta}")

        clip_triage = ClipTriage(clip_path=clip_path)

        try:
            # Sample frames
            frame_paths = sample_clip_frames(clip_path, count=frames_per_clip)

            if not frame_paths:
                print(f"           No frames extracted")
                results.append(clip_triage)
                continue

            # Encode frames
            frame_embeddings = embedder.encode_images(frame_paths)

            # Check for text overlays / title cards
            is_text, text_score = detect_text_overlays(
                embedder, frame_embeddings,
                _prompt_embeddings=text_prompt_cache,
            )
            clip_triage.has_text_overlay = is_text
            clip_triage.text_overlay_score = text_score

            # Compare against each reference
            matches: list[ClipMatch] = []
            match_labels: list[str] = []

            for ref, ref_emb in ref_embeddings.items():
                best_score, best_idx = embedder.best_match_score(
                    frame_embeddings, ref_emb
                )
                if best_score >= threshold:
                    matches.append(ClipMatch(
                        concept=ref,
                        similarity=best_score,
                        best_frame_index=best_idx,
                    ))
                    match_labels.append(
                        f"{ref.name} ({best_score:.2f})"
                    )

            # Sort matches by similarity (highest first)
            matches.sort(key=lambda m: m.similarity, reverse=True)
            clip_triage.matches = matches

            # Print result
            elapsed = time.time() - start
            durations.append(elapsed)

            if is_text:
                print(f"           TEXT OVERLAY (score: {text_score:.2f}) ({elapsed:.1f}s)")
            elif matches:
                print(f"           MATCH: {', '.join(match_labels)} ({elapsed:.1f}s)")
            else:
                # Show the closest near-miss for debugging
                best_overall = 0.0
                best_ref_name = ""
                for ref, ref_emb in ref_embeddings.items():
                    score, _ = embedder.best_match_score(frame_embeddings, ref_emb)
                    if score > best_overall:
                        best_overall = score
                        best_ref_name = ref.name
                print(f"           no match (closest: {best_ref_name} at {best_overall:.2f}) ({elapsed:.1f}s)")

            # Clean up temp frames
            cleanup_frames(frame_paths)

        except Exception as e:
            elapsed = time.time() - start
            durations.append(elapsed)
            print(f"           FAILED: {e} ({elapsed:.1f}s)")
            # Show full traceback for first failure to aid debugging
            if len(durations) <= 1:
                traceback.print_exc()

        results.append(clip_triage)

    # Build report
    report = TriageReport(
        clips=results,
        concepts=concepts,
        threshold=threshold,
        model_name=model_name,
    )

    # Print summary
    print()
    text_msg = f", {report.text_overlay_count} text overlays" if report.text_overlay_count else ""
    print(f"Triage complete: {report.matched_count}/{report.total} matched, "
          f"{report.unmatched_count} unmatched{text_msg}")

    if report.unmatched_count > 0:
        print(f"\n{report.unmatched_count} clip(s) didn't match any reference. "
              f"You can:")
        print(f"  - Lower the threshold (current: {threshold:.2f})")
        print(f"  - Add more reference images to concepts/")
        print(f"  - Sort unmatched clips manually")

    # Write manifest
    if output_path is None:
        output_path = clips_dir / "triage_manifest.json"
    else:
        output_path = Path(output_path)

    _write_manifest(report, output_path)

    return report


def _write_manifest(report: TriageReport, output_path: Path) -> None:
    """Write triage results as a JSON manifest.

    The manifest is designed to be human-readable and editable —
    users can review matches and adjust before proceeding to captioning.

    Args:
        report: The complete triage report.
        output_path: Where to write the JSON file.
    """
    manifest = {
        "triage": {
            "model": report.model_name,
            "threshold": report.threshold,
            "total_clips": report.total,
            "matched": report.matched_count,
            "unmatched": report.unmatched_count,
            "text_overlays": report.text_overlay_count,
        },
        "concepts": [
            {
                "name": ref.name,
                "type": ref.concept_type.value if ref.concept_type else None,
                "folder": ref.folder_name,
                "image": str(ref.image_path),
            }
            for ref in report.concepts
        ],
        "clips": [],
    }

    for clip in report.clips:
        clip_entry: dict = {
            "file": clip.clip_path.name,
            "path": str(clip.clip_path),
        }

        if clip.has_text_overlay:
            clip_entry["text_overlay"] = True
            clip_entry["text_overlay_score"] = round(clip.text_overlay_score, 3)

        if clip.is_matched:
            clip_entry["matches"] = [
                {
                    "concept": m.concept.name,
                    "type": m.concept.concept_type.value if m.concept.concept_type else None,
                    "similarity": round(m.similarity, 3),
                    "best_frame": m.best_frame_index,
                }
                for m in clip.matches
            ]
            # Primary match determines the captioning use case
            best = clip.best_match
            if best and best.concept.concept_type:
                clip_entry["use_case"] = best.concept.concept_type.value
                clip_entry["anchor_word"] = best.concept.name
        else:
            clip_entry["matches"] = []
            clip_entry["use_case"] = None
            clip_entry["anchor_word"] = None

        manifest["clips"].append(clip_entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest written: {output_path}")
    print("Review the manifest and adjust matches before captioning.")


def triage_videos(
    video_paths: list[Path],
    concepts_dir: str | Path,
    threshold: float = 0.70,
    frames_per_scene: int = 2,
    scene_threshold: float = 27.0,
    model_name: str = "openai/clip-vit-base-patch32",
    output_path: str | Path | None = None,
) -> VideoTriageReport:
    """Run scene-aware triage on long/raw videos.

    For each video: detect scenes -> sample frames per scene -> CLIP match.
    Produces a scene triage manifest that filtered ingest can read to only
    split matching scenes.

    This is the "triage first" workflow — users identify which scenes
    contain their target concept BEFORE spending time splitting everything.

    Args:
        video_paths: List of video file paths to triage.
        concepts_dir: Directory with concept reference images.
        threshold: Minimum cosine similarity for a match (0.0-1.0).
        frames_per_scene: Number of frames to sample per scene (default: 2).
        scene_threshold: Scene detection sensitivity (default: 27.0).
        model_name: CLIP model to use.
        output_path: Path for scene triage manifest JSON. Defaults to
            scene_triage_manifest.json in the first video's parent directory.

    Returns:
        VideoTriageReport with per-scene matches grouped by video.

    Raises:
        ImportError: if torch or transformers aren't installed.
        FileNotFoundError: if concepts_dir doesn't exist.
    """
    from klippbok.video.scene import detect_scenes

    # Validate dependencies first
    check_clip_available()

    concepts_dir = Path(concepts_dir).resolve()

    # Step 1: Discover concepts
    print("Discovering concept references...")
    concepts = discover_concepts(concepts_dir)

    if not concepts:
        print("  No reference images found. Nothing to match against.")
        return VideoTriageReport(
            threshold=threshold,
            model_name=model_name,
            scene_detection_threshold=scene_threshold,
            frames_per_scene=frames_per_scene,
        )

    print_concept_summary(concepts)
    print()

    print(f"Found {len(video_paths)} video(s) and {len(concepts)} reference(s)")
    print()

    # Step 2: Load CLIP model and encode references
    print(f"Loading CLIP model ({model_name})...")
    embedder = CLIPEmbedder(model_name=model_name)

    print("Encoding reference images...")
    ref_embeddings = {}
    for ref in concepts:
        try:
            ref_embeddings[ref] = embedder.encode_image(ref.image_path)
            print(f"  {ref.folder_name}/{ref.name} - OK")
        except Exception as e:
            print(f"  {ref.folder_name}/{ref.name} - FAILED: {e}")
    print()

    if not ref_embeddings:
        print("No reference images could be encoded. Cannot proceed.")
        return VideoTriageReport(
            concepts=concepts,
            threshold=threshold,
            model_name=model_name,
            scene_detection_threshold=scene_threshold,
            frames_per_scene=frames_per_scene,
        )

    # Pre-compute text overlay prompt embeddings
    print("Building text overlay filter...")
    text_prompt_cache = build_prompt_cache(embedder)

    # Step 3: Process each video
    all_scenes: list[SceneTriage] = []
    total_scene_count = 0
    durations: list[float] = []

    for vid_idx, video_path in enumerate(video_paths, 1):
        print(f"\n=== [{vid_idx}/{len(video_paths)}] {video_path.name} ===")

        try:
            # Detect scenes in this video
            print("  Detecting scenes...")
            boundaries = detect_scenes(video_path, threshold=scene_threshold)

            # Build segment list from boundaries
            vid_duration = _get_duration(video_path)
            boundary_times = [0.0] + [b.timecode for b in boundaries] + [vid_duration]
            segments = [
                (boundary_times[i], boundary_times[i + 1])
                for i in range(len(boundary_times) - 1)
            ]

            scene_count = len(segments)
            total_scene_count += scene_count
            print(f"  Found {len(boundaries)} cut(s) -> {scene_count} scene(s)")
            print(f"  Matching scenes (threshold: {threshold:.2f}, "
                  f"{frames_per_scene} frames/scene)...")

            # Process each scene
            for scene_idx, (start, end) in enumerate(segments):
                start_t = time.time()

                # Progress with ETA
                if durations:
                    avg = sum(durations) / len(durations)
                    remaining_scenes = scene_count - scene_idx
                    remaining_vids = len(video_paths) - vid_idx
                    # rough estimate — assume same scene count for remaining videos
                    total_remaining = remaining_scenes + remaining_vids * scene_count
                    eta = f" (ETA: {total_remaining * avg:.0f}s)"
                else:
                    eta = ""

                print(f"  [{scene_idx + 1}/{scene_count}] "
                      f"{start:.1f}s-{end:.1f}s{eta}", end="")

                scene_triage = SceneTriage(
                    source_video=video_path,
                    scene_index=scene_idx,
                    start_time=start,
                    end_time=end,
                )

                try:
                    # Sample frames from this scene
                    frame_paths = sample_scene_frames(
                        video_path, start, end, count=frames_per_scene,
                    )

                    if not frame_paths:
                        print(" — no frames extracted")
                        all_scenes.append(scene_triage)
                        continue

                    # Encode frames with CLIP
                    frame_embeddings = embedder.encode_images(frame_paths)

                    # Check for text overlays
                    is_text, text_score = detect_text_overlays(
                        embedder, frame_embeddings,
                        _prompt_embeddings=text_prompt_cache,
                    )
                    scene_triage.has_text_overlay = is_text
                    scene_triage.text_overlay_score = text_score

                    # Compare against each reference
                    matches: list[ClipMatch] = []
                    match_labels: list[str] = []

                    for ref, ref_emb in ref_embeddings.items():
                        best_score, best_idx = embedder.best_match_score(
                            frame_embeddings, ref_emb
                        )
                        if best_score >= threshold:
                            matches.append(ClipMatch(
                                concept=ref,
                                similarity=best_score,
                                best_frame_index=best_idx,
                            ))
                            match_labels.append(f"{ref.name} ({best_score:.2f})")

                    matches.sort(key=lambda m: m.similarity, reverse=True)
                    scene_triage.matches = matches

                    # Print result
                    elapsed = time.time() - start_t
                    durations.append(elapsed)

                    if is_text:
                        print(f" — TEXT OVERLAY ({text_score:.2f}) ({elapsed:.1f}s)")
                    elif matches:
                        print(f" — MATCH: {', '.join(match_labels)} ({elapsed:.1f}s)")
                    else:
                        print(f" — no match ({elapsed:.1f}s)")

                    # Clean up temp frames
                    cleanup_frames(frame_paths)

                except Exception as e:
                    elapsed = time.time() - start_t
                    durations.append(elapsed)
                    print(f" — FAILED: {e} ({elapsed:.1f}s)")

                all_scenes.append(scene_triage)

        except Exception as e:
            print(f"  FAILED to process video: {e}")

    # Build report
    report = VideoTriageReport(
        scenes=all_scenes,
        concepts=concepts,
        threshold=threshold,
        model_name=model_name,
        scene_detection_threshold=scene_threshold,
        frames_per_scene=frames_per_scene,
    )

    # Print summary
    print()
    text_msg = (f", {report.text_overlay_count} text overlays"
                if report.text_overlay_count else "")
    print(f"Scene triage complete: {report.matched_count}/{report.total} matched, "
          f"{report.unmatched_count} unmatched{text_msg}")

    # Write manifest
    if output_path is None:
        output_path = video_paths[0].parent / "scene_triage_manifest.json"
    else:
        output_path = Path(output_path)

    _write_scene_manifest(report, output_path)

    return report


def _write_scene_manifest(report: VideoTriageReport, output_path: Path) -> None:
    """Write scene-level triage results as a JSON manifest.

    The manifest groups scenes by source video and includes an 'include'
    field that users can edit before running filtered ingest. This is the
    key handoff point: triage writes it, user reviews it, ingest reads it.

    Args:
        report: The complete scene triage report.
        output_path: Where to write the JSON file.
    """
    manifest: dict = {
        "triage_mode": "scene",
        "triage": {
            "model": report.model_name,
            "threshold": report.threshold,
            "scene_detection_threshold": report.scene_detection_threshold,
            "frames_per_scene": report.frames_per_scene,
            "total_scenes": report.total,
            "matched": report.matched_count,
            "unmatched": report.unmatched_count,
            "text_overlays": report.text_overlay_count,
        },
        "concepts": [
            {
                "name": ref.name,
                "type": ref.concept_type.value if ref.concept_type else None,
                "folder": ref.folder_name,
                "image": str(ref.image_path),
            }
            for ref in report.concepts
        ],
        "videos": [],
    }

    # Group scenes by source video, preserving order
    by_video: dict[str, list[SceneTriage]] = {}
    video_order: list[str] = []
    for scene in report.scenes:
        key = str(scene.source_video)
        if key not in by_video:
            by_video[key] = []
            video_order.append(key)
        by_video[key].append(scene)

    for video_path_str in video_order:
        scenes = by_video[video_path_str]
        video_entry: dict = {
            "file": Path(video_path_str).name,
            "path": video_path_str,
            "total_scenes": len(scenes),
            "scenes": [],
        }

        for scene in scenes:
            scene_entry: dict = {
                "scene_index": scene.scene_index,
                "start_time": round(scene.start_time, 3),
                "end_time": round(scene.end_time, 3),
                "text_overlay": scene.has_text_overlay,
                "include": scene.is_matched and not scene.has_text_overlay,
            }

            if scene.matches:
                scene_entry["matches"] = [
                    {
                        "concept": m.concept.name,
                        "similarity": round(m.similarity, 3),
                        "best_frame": m.best_frame_index,
                    }
                    for m in scene.matches
                ]
            else:
                scene_entry["matches"] = []

            video_entry["scenes"].append(scene_entry)

        manifest["videos"].append(video_entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nScene manifest written: {output_path}")
    print("Review the manifest — flip 'include' on/off before running filtered ingest.")
    print(f"  python -m klippbok.video ingest <video_dir> -o <output_dir> "
          f"--triage {output_path}")


def organize_clips(
    report: TriageReport,
    output_dir: str | Path,
    copy: bool = True,
) -> dict[str, list[Path]]:
    """Organize triaged clips into concept-named folders.

    After triage, this function physically sorts clips into folders
    based on their best match. Each matched concept gets a folder
    named after the concept (e.g. "hollygolightly/"). Unmatched
    clips go into an "unmatched/" folder.

    Sidecar files (.txt, .json) are also copied/moved alongside
    their video clip.

    Args:
        report: Completed triage report with per-clip matches.
        output_dir: Root directory where concept folders will be created.
        copy: If True, copy files. If False, move files.

    Returns:
        Dict mapping folder names to lists of clip paths in that folder.
        e.g. {"hollygolightly": [Path(...), ...], "unmatched": [Path(...)]}
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transfer = shutil.copy2 if copy else shutil.move
    action = "Copying" if copy else "Moving"

    organized: dict[str, list[Path]] = {}
    sidecar_extensions = {".txt", ".json", ".jsonl"}

    for clip in report.clips:
        # Determine target folder name
        if clip.has_text_overlay:
            folder_name = "text_overlay"
        elif clip.is_matched and clip.best_match:
            folder_name = clip.best_match.concept.name
        else:
            folder_name = "unmatched"

        # Create folder
        target_dir = output_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # Transfer the clip
        source = clip.clip_path
        dest = target_dir / source.name

        if dest.exists() and dest.resolve() == source.resolve():
            # Already in the right place
            organized.setdefault(folder_name, []).append(dest)
            continue

        try:
            transfer(str(source), str(dest))
            organized.setdefault(folder_name, []).append(dest)

            # Transfer sidecars (captions, metadata) if they exist
            for ext in sidecar_extensions:
                sidecar = source.with_suffix(ext)
                if sidecar.is_file():
                    sidecar_dest = target_dir / sidecar.name
                    if not sidecar_dest.exists():
                        transfer(str(sidecar), str(sidecar_dest))

        except Exception as e:
            print(f"  WARNING: Failed to organize {source.name}: {e}")

    # Print summary
    print(f"\n{action} clips into {len(organized)} folder(s):")
    for folder_name in sorted(organized.keys()):
        count = len(organized[folder_name])
        label = f"  {folder_name}/" if folder_name != "unmatched" else "  unmatched/"
        print(f"{label} ({count} clip{'s' if count != 1 else ''})")

    return organized
