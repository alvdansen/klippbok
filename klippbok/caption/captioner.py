"""Caption orchestrator -batch captioning and auditing.

Coordinates the VLM backends to caption directories of video clips.
Handles progress tracking, error recovery, and output file writing.

The orchestrator is backend-agnostic -it works with any VLMBackend
implementation through the abstract base class.
"""

from __future__ import annotations

import time
from pathlib import Path

from klippbok.caption.base import VLMBackend
from klippbok.caption.models import AuditResult, CaptionConfig, CaptionResult
from klippbok.caption.prompts import get_video_prompt


# Video file extensions to process
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _create_backend(config: CaptionConfig) -> VLMBackend:
    """Create a VLM backend from config.

    Args:
        config: Caption configuration with provider and API key.

    Returns:
        An initialized VLMBackend instance.

    Raises:
        ValueError: if the provider is not recognized.
    """
    if config.provider == "gemini":
        from klippbok.caption.gemini import GeminiBackend
        return GeminiBackend(
            api_key=config.api_key,
            model=config.gemini_model,
            timeout=config.timeout,
            max_retries=config.max_retries,
            caption_fps=config.caption_fps,
        )
    elif config.provider == "replicate":
        from klippbok.caption.replicate import ReplicateBackend
        return ReplicateBackend(
            api_token=config.api_key,
            model=config.replicate_model,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
    elif config.provider == "openai":
        from klippbok.caption.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(
            base_url=config.openai_base_url,
            model=config.openai_model,
            api_key=config.api_key,
            timeout=config.timeout,
            caption_fps=config.caption_fps,
        )
    else:
        raise ValueError(
            f"Unknown caption provider: {config.provider!r}. "
            f"Valid options: 'gemini', 'replicate', 'openai'"
        )


def _find_video_files(directory: Path) -> list[Path]:
    """Find all video files in a directory, sorted by name.

    Args:
        directory: Directory to scan.

    Returns:
        Sorted list of video file paths.
    """
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    )


def _prepend_anchor(caption: str, anchor_word: str) -> str:
    """Prepend anchor word to caption if not already present.

    Checks if the caption already starts with the anchor word
    (case-insensitive). If not, prepends it naturally.

    Args:
        caption: The generated caption text.
        anchor_word: The anchor word to prepend.

    Returns:
        Caption with anchor word at the start.

    Examples:
        >>> _prepend_anchor("A girl walks", "Luna")
        "Luna, a girl walks"
        >>> _prepend_anchor("Luna is walking", "Luna")
        "Luna is walking"
    """
    if caption.lower().startswith(anchor_word.lower()):
        return caption

    # Lowercase the first character of the caption for natural flow
    if caption and caption[0].isupper():
        caption = caption[0].lower() + caption[1:]

    return f"{anchor_word}, {caption}"


def caption_clips(
    directory: str | Path,
    config: CaptionConfig,
) -> list[CaptionResult]:
    """Caption all video clips in a directory.

    Scans for video files, generates captions using the configured VLM
    backend, and writes .txt sidecar files alongside each clip.

    Progress is printed to console with estimated time remaining.
    Individual failures are logged but don't stop the batch.

    Args:
        directory: Directory containing video clips.
        config: Caption configuration.

    Returns:
        List of CaptionResult for all clips (including skipped/failed).
    """
    directory = Path(directory).resolve()

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find video files
    video_files = _find_video_files(directory)
    if not video_files:
        print(f"No video files found in {directory}")
        return []

    # Create backend
    backend = _create_backend(config)

    # Get the appropriate prompt — custom_prompt overrides use-case selection.
    # Anchor word is baked INTO the prompt so the VLM uses it as a name.
    if config.custom_prompt:
        prompt = config.custom_prompt
    else:
        prompt = get_video_prompt(
            config.use_case,
            anchor_word=config.anchor_word,
            secondary_anchors=config.secondary_anchors,
        )

    results: list[CaptionResult] = []
    total = len(video_files)
    durations: list[float] = []

    for i, video_path in enumerate(video_files, 1):
        # Check for existing caption
        caption_path = video_path.with_suffix(".txt")
        if caption_path.exists() and not config.overwrite:
            print(f"  [{i}/{total}] {video_path.name} -skipped (caption exists)")
            results.append(CaptionResult(
                path=video_path,
                skipped=True,
                provider=config.provider,
            ))
            continue

        # Estimate time remaining
        if durations:
            avg_duration = sum(durations) / len(durations)
            remaining = (total - i) * (avg_duration + config.between_request_delay)
            eta = f" (ETA: {remaining:.0f}s)"
        else:
            eta = ""

        print(f"  [{i}/{total}] {video_path.name}{eta}")

        # Generate caption
        start_time = time.time()
        try:
            caption = backend.caption_video(video_path, prompt)
            elapsed = time.time() - start_time
            durations.append(elapsed)

            # Write caption file
            caption_path.write_text(caption, encoding="utf-8")
            print(f"           OK {elapsed:.1f}s -{len(caption)} chars")

            results.append(CaptionResult(
                path=video_path,
                caption=caption,
                provider=config.provider,
                duration=elapsed,
                success=True,
            ))

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"           FAIL FAILED ({elapsed:.1f}s): {e}")
            results.append(CaptionResult(
                path=video_path,
                provider=config.provider,
                duration=elapsed,
                success=False,
                error=str(e),
            ))

        # Rate limiting delay (skip after last clip)
        if i < total:
            time.sleep(config.between_request_delay)

    # Print summary
    succeeded = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)

    print()
    parts = [f"Captioning complete: {succeeded}/{total} succeeded"]
    if skipped:
        parts.append(f"{skipped} skipped")
    if failed:
        parts.append(f"{failed} failed")
    print(", ".join(parts))

    if failed:
        print("To retry failed clips, run the same command again — "
              "existing captions are skipped automatically.")

    print()
    print("Tip: Review your captions before training. Auto-generated captions "
          "are a starting point — a quick manual pass catches mistakes and "
          "improves training quality.")

    return results


def audit_captions(
    directory: str | Path,
    config: CaptionConfig,
) -> list[AuditResult]:
    """Audit existing captions against fresh VLM output.

    For each clip that has a .txt caption, sends the clip to the VLM
    and compares the result with the existing caption. This helps
    identify captions that may need updating.

    Two modes:
    - report_only: prints comparison to console
    - save_audit: saves VLM suggestion as {stem}.audit.txt

    Args:
        directory: Directory containing captioned video clips.
        config: Caption configuration with audit_mode set.

    Returns:
        List of AuditResult for all audited clips.
    """
    directory = Path(directory).resolve()

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find video files that have existing captions
    video_files = _find_video_files(directory)
    captioned = [
        f for f in video_files
        if f.with_suffix(".txt").exists()
    ]

    if not captioned:
        print(f"No captioned video files found in {directory}")
        return []

    # Create backend
    backend = _create_backend(config)
    prompt = config.custom_prompt or get_video_prompt(config.use_case)

    results: list[AuditResult] = []
    total = len(captioned)

    for i, video_path in enumerate(captioned, 1):
        caption_path = video_path.with_suffix(".txt")
        existing = caption_path.read_text(encoding="utf-8").strip()

        print(f"  [{i}/{total}] {video_path.name}")

        try:
            vlm_caption = backend.caption_video(video_path, prompt)

            # Simple similarity check: word overlap ratio
            existing_words = set(existing.lower().split())
            vlm_words = set(vlm_caption.lower().split())
            if existing_words and vlm_words:
                overlap = len(existing_words & vlm_words)
                total_words = len(existing_words | vlm_words)
                similarity = overlap / total_words
            else:
                similarity = 0.0

            recommendation = "keep" if similarity > 0.4 else "review"

            result = AuditResult(
                path=video_path,
                existing_caption=existing,
                vlm_caption=vlm_caption,
                recommendation=recommendation,
                provider=config.provider,
            )
            results.append(result)

            # Print comparison
            status = "KEEP" if recommendation == "keep" else "REVIEW"
            print(f"           [{status}] similarity: {similarity:.0%}")

            if config.audit_mode == "report_only":
                if recommendation == "review":
                    print(f"           Existing: {existing[:80]}...")
                    print(f"           VLM says: {vlm_caption[:80]}...")
            elif config.audit_mode == "save_audit":
                audit_path = video_path.with_suffix(".audit.txt")
                audit_path.write_text(vlm_caption, encoding="utf-8")
                if recommendation == "review":
                    print(f"           Saved: {audit_path.name}")

        except Exception as e:
            print(f"           FAIL FAILED: {e}")

        # Rate limiting
        if i < total:
            time.sleep(config.between_request_delay)

    return results
