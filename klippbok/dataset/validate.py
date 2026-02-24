"""Core dataset validation engine.

Orchestrates structural checks (file pairing from discover.py),
quality checks (metrics from quality.py and image_quality.py),
and completeness checks into a unified validation pipeline.

Follows the same pattern as klippbok.video.validate:
- Accumulate ValidationIssue objects
- Check each thing independently (no early exits)
- Return structured results for reporting

Usage:
    from klippbok.config.loader import load_config
    from klippbok.dataset.validate import validate_all

    config = load_config("klippbok_data.yaml")
    report = validate_all(config, config_dir=Path("."))
"""

from __future__ import annotations

from pathlib import Path

from klippbok.config.data_schema import KlippbokDataConfig
from klippbok.dataset.discover import (
    discover_all_datasets,
    discover_dataset,
    validate_file_type,
)
from klippbok.dataset.models import (
    DatasetReport,
    DatasetValidation,
    SamplePair,
    StructureType,
)
from klippbok.video.models import IssueCode, Severity, ValidationIssue


# ---------------------------------------------------------------------------
# Per-sample validation
# ---------------------------------------------------------------------------

def validate_sample(
    sample: SamplePair,
    config: KlippbokDataConfig,
) -> SamplePair:
    """Run all validation checks on a single training sample.

    Checks:
    1. File type magic bytes (if filetype is installed)
    2. Caption content (empty, too long)
    3. Reference image quality (blank detection via sharpness)
    4. Blur threshold (if configured)
    5. Exposure range (if configured)

    Returns a new SamplePair with accumulated issues (original issues
    from discovery are preserved, new issues are appended).

    Args:
        sample: The discovered sample pair (may already have issues from pairing).
        config: The data config with quality thresholds.

    Returns:
        New SamplePair with all validation issues collected.
    """
    issues = list(sample.issues)  # preserve discovery-time issues
    width = sample.width
    height = sample.height
    frame_count = sample.frame_count
    fps = sample.fps

    # ── File type validation ──
    if sample.target.exists():
        ft_issue = validate_file_type(sample.target, "video")
        if ft_issue is not None:
            issues.append(ft_issue)

    if sample.reference is not None and sample.reference.exists():
        ft_issue = validate_file_type(sample.reference, "image")
        if ft_issue is not None:
            issues.append(ft_issue)

    # ── Caption content checks ──
    if sample.caption is not None and sample.caption.exists():
        try:
            caption_text = sample.caption.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            caption_text = ""
            issues.append(ValidationIssue(
                code=IssueCode.FILE_CORRUPTED,
                severity=Severity.ERROR,
                message=(
                    f"Cannot read caption file '{sample.caption.name}'. "
                    f"Check file encoding (should be UTF-8) and permissions."
                ),
                field="caption",
            ))

        if not caption_text:
            issues.append(ValidationIssue(
                code=IssueCode.CAPTION_EMPTY,
                severity=Severity.WARNING,
                message=(
                    f"Caption file '{sample.caption.name}' is empty. "
                    f"An empty caption means no text conditioning for this clip. "
                    f"This may be intentional (caption dropout) or an error."
                ),
                field="caption",
            ))

        # Token length is estimated by word count (rough proxy)
        max_tokens = config.controls.text.max_tokens
        # Rough estimate: 1 word ~ 1.3 tokens for English
        word_count = len(caption_text.split())
        estimated_tokens = int(word_count * 1.3)
        if estimated_tokens > max_tokens:
            issues.append(ValidationIssue(
                code=IssueCode.CAPTION_TOO_LONG,
                severity=Severity.WARNING,
                message=(
                    f"Caption for '{sample.stem}' is ~{estimated_tokens} tokens "
                    f"(estimated from {word_count} words). Max is {max_tokens}. "
                    f"The text encoder will truncate at {max_tokens} tokens. "
                    f"Consider shortening the caption."
                ),
                field="caption",
                actual=estimated_tokens,
                expected=max_tokens,
            ))

    # ── Reference image quality checks ──
    if sample.reference is not None and sample.reference.exists():
        try:
            import cv2
            img = cv2.imread(str(sample.reference), cv2.IMREAD_GRAYSCALE)
            if img is None:
                issues.append(ValidationIssue(
                    code=IssueCode.FILE_CORRUPTED,
                    severity=Severity.ERROR,
                    message=(
                        f"Cannot read reference image '{sample.reference.name}'. "
                        f"File may be corrupted or not a supported format."
                    ),
                    field="reference",
                ))
            else:
                # Blank detection via Laplacian variance
                laplacian = cv2.Laplacian(img, cv2.CV_64F)
                sharpness = float(laplacian.var())
                if sharpness < 5.0:
                    issues.append(ValidationIssue(
                        code=IssueCode.REFERENCE_BLANK,
                        severity=Severity.ERROR,
                        message=(
                            f"Reference image '{sample.reference.name}' appears blank "
                            f"(sharpness={sharpness:.1f}, threshold=5.0). "
                            f"A blank reference carries no visual information for the VAE. "
                            f"Re-extract with a different frame or strategy."
                        ),
                        field="reference",
                        actual=sharpness,
                        expected=5.0,
                    ))

                # Blur threshold check (if configured)
                blur_threshold = config.quality.blur_threshold
                if blur_threshold is not None and sharpness < blur_threshold:
                    issues.append(ValidationIssue(
                        code=IssueCode.BLUR_BELOW_THRESHOLD,
                        severity=Severity.WARNING,
                        message=(
                            f"Reference image '{sample.reference.name}' is blurry "
                            f"(sharpness={sharpness:.1f}, threshold={blur_threshold}). "
                            f"Consider re-extracting with best_frame strategy."
                        ),
                        field="quality",
                        actual=sharpness,
                        expected=blur_threshold,
                    ))

                # Exposure check (if configured)
                exposure_range = config.quality.exposure_range
                if exposure_range is not None:
                    from klippbok.dataset.quality import compute_exposure
                    mean_brightness, _ = compute_exposure(sample.reference)
                    low, high = exposure_range
                    if mean_brightness < low or mean_brightness > high:
                        issues.append(ValidationIssue(
                            code=IssueCode.EXPOSURE_OUT_OF_RANGE,
                            severity=Severity.WARNING,
                            message=(
                                f"Reference image '{sample.reference.name}' has "
                                f"brightness {mean_brightness:.2f}, outside range "
                                f"[{low:.2f}, {high:.2f}]. "
                                f"Image may be under- or over-exposed."
                            ),
                            field="quality",
                            actual=mean_brightness,
                            expected=f"[{low}, {high}]",
                        ))
        except ImportError:
            # OpenCV not installed — skip image quality checks
            pass

    return SamplePair(
        stem=sample.stem,
        target=sample.target,
        caption=sample.caption,
        reference=sample.reference,
        issues=issues,
        width=width,
        height=height,
        frame_count=frame_count,
        fps=fps,
    )


# ---------------------------------------------------------------------------
# Per-dataset validation
# ---------------------------------------------------------------------------

def validate_dataset(
    discovered: DatasetValidation,
    config: KlippbokDataConfig,
) -> DatasetValidation:
    """Validate all samples in a discovered dataset.

    Runs per-sample validation, then dataset-level checks:
    - Motion intensity (if thresholds configured, requires ffmpeg)
    - Duplicate detection (if enabled in config)

    Args:
        discovered: The discovery result from discover_dataset().
        config: The data config with quality thresholds.

    Returns:
        New DatasetValidation with all issues collected.
    """
    # Validate each sample
    validated_samples = [validate_sample(s, config) for s in discovered.samples]

    # Dataset-level issues (start with discovery-time issues)
    dataset_issues = list(discovered.dataset_issues)

    # ── Motion intensity checks (require video files + OpenCV) ──
    motion_config = config.quality.motion
    if motion_config.min_intensity is not None or motion_config.max_intensity is not None:
        try:
            from klippbok.dataset.quality import compute_motion_intensity

            new_samples = []
            for sample in validated_samples:
                extra_issues = list(sample.issues)
                if sample.target.exists():
                    try:
                        motion = compute_motion_intensity(sample.target)
                        if (motion_config.min_intensity is not None
                                and motion < motion_config.min_intensity):
                            extra_issues.append(ValidationIssue(
                                code=IssueCode.MOTION_BELOW_MIN,
                                severity=Severity.WARNING,
                                message=(
                                    f"Clip '{sample.stem}' has low motion intensity "
                                    f"({motion:.1f}%, min={motion_config.min_intensity}%). "
                                    f"Static clips teach the model very little about motion."
                                ),
                                field="quality",
                                actual=motion,
                                expected=motion_config.min_intensity,
                            ))
                        if (motion_config.max_intensity is not None
                                and motion > motion_config.max_intensity):
                            extra_issues.append(ValidationIssue(
                                code=IssueCode.MOTION_ABOVE_MAX,
                                severity=Severity.WARNING,
                                message=(
                                    f"Clip '{sample.stem}' has excessive motion "
                                    f"({motion:.1f}%, max={motion_config.max_intensity}%). "
                                    f"Chaotic motion is hard for the model to learn."
                                ),
                                field="quality",
                                actual=motion,
                                expected=motion_config.max_intensity,
                            ))
                    except (ValueError, OSError):
                        # Can't measure motion — skip (file issues caught elsewhere)
                        pass

                new_samples.append(SamplePair(
                    stem=sample.stem,
                    target=sample.target,
                    caption=sample.caption,
                    reference=sample.reference,
                    issues=extra_issues,
                    width=sample.width,
                    height=sample.height,
                    frame_count=sample.frame_count,
                    fps=sample.fps,
                ))
            validated_samples = new_samples
        except ImportError:
            pass

    # ── Duplicate detection (opt-in) ──
    if config.quality.check_duplicates:
        ref_paths = [
            s.reference for s in validated_samples
            if s.reference is not None and s.reference.exists()
        ]
        if len(ref_paths) >= 2:
            try:
                from klippbok.dataset.quality import find_duplicates
                groups = find_duplicates(ref_paths)
                for group in groups:
                    names = ", ".join(p.name for p in group)
                    dataset_issues.append(ValidationIssue(
                        code=IssueCode.DUPLICATE_DETECTED,
                        severity=Severity.WARNING,
                        message=(
                            f"Duplicate reference images detected: {names}. "
                            f"These images are perceptually very similar. "
                            f"Duplicates waste training compute and can bias the model."
                        ),
                        field="quality",
                    ))
            except ImportError:
                pass

    return DatasetValidation(
        source_path=discovered.source_path,
        structure=discovered.structure,
        samples=validated_samples,
        orphaned_files=discovered.orphaned_files,
        dataset_issues=dataset_issues,
    )


# ---------------------------------------------------------------------------
# Top-level validation
# ---------------------------------------------------------------------------

def validate_all(
    config: KlippbokDataConfig,
    config_dir: Path | None = None,
) -> DatasetReport:
    """Validate all dataset sources defined in a config.

    This is the main entry point for dataset validation. It:
    1. Discovers all files in each dataset source
    2. Validates each source (per-sample + dataset-level checks)
    3. Runs cross-dataset checks (duplicate detection across sources)

    Args:
        config: The data config.
        config_dir: Base directory for resolving relative paths.

    Returns:
        DatasetReport spanning all sources.
    """
    if config_dir is None:
        config_dir = Path.cwd()

    # Discover all datasets
    discovered_list = discover_all_datasets(config, config_dir)

    # Validate each dataset
    validated_list = [validate_dataset(d, config) for d in discovered_list]

    # Cross-dataset duplicate detection
    cross_issues: list[ValidationIssue] = []
    if config.quality.check_duplicates and len(validated_list) > 1:
        # Collect all reference images across all sources
        all_refs: list[Path] = []
        for ds in validated_list:
            for s in ds.samples:
                if s.reference is not None and s.reference.exists():
                    all_refs.append(s.reference)

        if len(all_refs) >= 2:
            try:
                from klippbok.dataset.quality import find_duplicates
                groups = find_duplicates(all_refs)
                for group in groups:
                    # Only flag groups that span multiple sources
                    sources_in_group: set[Path] = set()
                    for ref_path in group:
                        for ds in validated_list:
                            for s in ds.samples:
                                if s.reference == ref_path:
                                    sources_in_group.add(ds.source_path)
                    if len(sources_in_group) > 1:
                        names = ", ".join(p.name for p in group)
                        cross_issues.append(ValidationIssue(
                            code=IssueCode.DUPLICATE_DETECTED,
                            severity=Severity.WARNING,
                            message=(
                                f"Cross-dataset duplicates: {names}. "
                                f"Found across {len(sources_in_group)} dataset sources."
                            ),
                            field="quality",
                        ))
            except ImportError:
                pass

    return DatasetReport(
        datasets=validated_list,
        cross_dataset_issues=cross_issues,
    )
