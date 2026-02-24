"""Dataset file discovery and structure detection.

Scans directories, detects layout (flat vs klippbok), classifies files by
role (target, caption, reference), and pairs them by filename stem.

This is the foundation for dataset validation — before we can check
quality or completeness, we need to know what files exist and how
they relate to each other.

Uses `filetype` for magic-byte validation when available, falling back
to extension-based classification when it's not installed.
"""

from __future__ import annotations

from pathlib import Path

from klippbok.config.data_schema import KlippbokDataConfig
from klippbok.dataset.errors import DatasetValidationError
from klippbok.dataset.models import (
    DatasetValidation,
    SamplePair,
    StructureType,
)
from klippbok.video.models import IssueCode, Severity, ValidationIssue


# ---------------------------------------------------------------------------
# File classification constants
# ---------------------------------------------------------------------------

# Extensions recognized as video targets
VIDEO_EXTENSIONS: set[str] = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv",
    ".m4v", ".mpg", ".mpeg", ".ts", ".mts",
}

# Extensions recognized as images (references)
IMAGE_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif",
}

# Extensions recognized as captions
CAPTION_EXTENSIONS: set[str] = {".txt"}

# MIME type prefixes for magic-byte validation
_VIDEO_MIME_PREFIX = "video/"
_IMAGE_MIME_PREFIX = "image/"


# ---------------------------------------------------------------------------
# Magic-byte validation (optional filetype dependency)
# ---------------------------------------------------------------------------

def _check_magic_bytes(path: Path) -> str | None:
    """Validate file type using magic bytes via the `filetype` library.

    Returns the detected MIME type, or None if:
    - filetype is not installed (graceful degradation)
    - the file can't be read or has no recognized magic bytes

    Why magic bytes: file extensions can lie. A .mp4 file might actually
    be a .txt that was renamed. Magic bytes read the actual file header
    to determine the real type.
    """
    try:
        import filetype as ft
    except ImportError:
        return None

    kind = ft.guess(str(path))
    if kind is None:
        return None
    return kind.mime


def validate_file_type(path: Path, expected_category: str) -> ValidationIssue | None:
    """Check if a file's magic bytes match the expected category.

    Args:
        path: Path to the file.
        expected_category: 'video' or 'image'.

    Returns:
        A ValidationIssue if the magic bytes don't match, None if they do
        (or if filetype is not installed — we skip the check).
    """
    mime = _check_magic_bytes(path)
    if mime is None:
        # filetype not installed or unrecognized — skip silently
        return None

    expected_prefix = _VIDEO_MIME_PREFIX if expected_category == "video" else _IMAGE_MIME_PREFIX
    if not mime.startswith(expected_prefix):
        return ValidationIssue(
            code=IssueCode.FILE_TYPE_INVALID,
            severity=Severity.ERROR,
            message=(
                f"File '{path.name}' has extension '{path.suffix}' but magic bytes "
                f"indicate '{mime}'. The file may be mislabeled or corrupted. "
                f"Re-download or re-export this file."
            ),
            field="file_type",
            actual=mime,
            expected=expected_prefix.rstrip("/"),
        )
    return None


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------

def detect_structure(directory: Path) -> StructureType:
    """Detect whether a directory uses flat or klippbok layout.

    Klippbok layout has a `training/targets/` subdirectory.
    Everything else is treated as flat layout.

    Args:
        directory: Path to the dataset source folder.

    Returns:
        StructureType.DIMLJUS if training/targets/ exists,
        StructureType.FLAT otherwise.
    """
    targets_dir = directory / "training" / "targets"
    if targets_dir.is_dir():
        return StructureType.DIMLJUS
    return StructureType.FLAT


# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------

def _classify_extension(path: Path) -> str:
    """Classify a file by its extension into a role.

    Returns one of: 'target', 'caption', 'reference', 'other'.
    """
    ext = path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "target"
    if ext in CAPTION_EXTENSIONS:
        return "caption"
    if ext in IMAGE_EXTENSIONS:
        return "reference"
    return "other"


def discover_files(
    directory: Path,
    structure: StructureType | None = None,
) -> dict[str, list[Path]]:
    """Scan a directory and classify files by role.

    For flat layout: scans the directory directly.
    For klippbok layout: scans training/targets/ for videos,
    training/signals/captions/ for captions, training/signals/references/ for images.

    Args:
        directory: Path to the dataset source folder.
        structure: Detected structure type. Auto-detected if None.

    Returns:
        Dict with keys 'targets', 'captions', 'references', 'other'.
        Each value is a sorted list of file paths.
    """
    if structure is None:
        structure = detect_structure(directory)

    result: dict[str, list[Path]] = {
        "targets": [],
        "captions": [],
        "references": [],
        "other": [],
    }

    if structure == StructureType.DIMLJUS:
        # Klippbok layout: structured subdirectories
        targets_dir = directory / "training" / "targets"
        captions_dir = directory / "training" / "signals" / "captions"
        references_dir = directory / "training" / "signals" / "references"

        if targets_dir.is_dir():
            for f in sorted(targets_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
                    result["targets"].append(f)
                elif f.is_file():
                    result["other"].append(f)

        if captions_dir.is_dir():
            for f in sorted(captions_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in CAPTION_EXTENSIONS:
                    result["captions"].append(f)
                elif f.is_file():
                    result["other"].append(f)

        if references_dir.is_dir():
            for f in sorted(references_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    result["references"].append(f)
                elif f.is_file():
                    result["other"].append(f)
    else:
        # Flat layout: everything in one directory
        if directory.is_dir():
            for f in sorted(directory.iterdir()):
                if not f.is_file():
                    continue
                role = _classify_extension(f)
                if role == "target":
                    result["targets"].append(f)
                elif role == "caption":
                    result["captions"].append(f)
                elif role == "reference":
                    result["references"].append(f)
                else:
                    result["other"].append(f)

    return result


# ---------------------------------------------------------------------------
# Stem-based pairing
# ---------------------------------------------------------------------------

def pair_samples(
    targets: list[Path],
    captions: list[Path],
    references: list[Path],
    caption_required: bool = True,
    reference_required: bool = False,
) -> tuple[list[SamplePair], list[Path]]:
    """Match targets with captions and references by filename stem.

    Each target video becomes a SamplePair. Captions and references are
    matched by stem — clip_001.mp4 pairs with clip_001.txt and clip_001.png.

    Unmatched signal files (captions/references without a matching target)
    are returned as orphaned files.

    Args:
        targets: List of target video file paths.
        captions: List of caption file paths.
        references: List of reference image file paths.
        caption_required: If True, missing captions produce ERROR; if False, WARNING.
        reference_required: If True, missing references produce ERROR; if False, skip silently.

    Returns:
        Tuple of (list of SamplePairs, list of orphaned file paths).
    """
    # Build lookup dicts: stem -> path
    caption_map: dict[str, Path] = {c.stem: c for c in captions}
    reference_map: dict[str, Path] = {r.stem: r for r in references}

    samples: list[SamplePair] = []
    matched_caption_stems: set[str] = set()
    matched_reference_stems: set[str] = set()

    for target in targets:
        stem = target.stem
        issues: list[ValidationIssue] = []

        # Match caption
        caption_path = caption_map.get(stem)
        if caption_path is not None:
            matched_caption_stems.add(stem)
        elif caption_required:
            issues.append(ValidationIssue(
                code=IssueCode.CAPTION_MISSING,
                severity=Severity.ERROR,
                message=(
                    f"No caption file found for '{target.name}'. "
                    f"Expected: '{stem}.txt' in the same location. "
                    f"Generate captions with: python -m klippbok.video caption"
                ),
                field="caption",
            ))

        # Match reference
        reference_path = reference_map.get(stem)
        if reference_path is not None:
            matched_reference_stems.add(stem)
        elif reference_required:
            issues.append(ValidationIssue(
                code=IssueCode.REFERENCE_MISSING,
                severity=Severity.ERROR,
                message=(
                    f"No reference image found for '{target.name}'. "
                    f"Expected: '{stem}.png' in the same location. "
                    f"Extract references with: python -m klippbok.video extract"
                ),
                field="reference",
            ))

        samples.append(SamplePair(
            stem=stem,
            target=target,
            caption=caption_path,
            reference=reference_path,
            issues=issues,
        ))

    # Find orphaned signal files (no matching target)
    orphaned: list[Path] = []
    for stem, path in caption_map.items():
        if stem not in matched_caption_stems and stem not in {t.stem for t in targets}:
            orphaned.append(path)
    for stem, path in reference_map.items():
        if stem not in matched_reference_stems and stem not in {t.stem for t in targets}:
            orphaned.append(path)

    return samples, sorted(orphaned)


# ---------------------------------------------------------------------------
# High-level discovery
# ---------------------------------------------------------------------------

def discover_dataset(
    directory: str | Path,
    config: KlippbokDataConfig,
) -> DatasetValidation:
    """Discover and pair all files in a single dataset source folder.

    Detects structure, classifies files, pairs by stem, and flags
    empty datasets and orphaned files.

    Args:
        directory: Path to the dataset source folder.
        config: The data config (controls whether captions/references are required).

    Returns:
        DatasetValidation with discovered samples and any discovery-time issues.
    """
    directory = Path(directory).resolve()

    if not directory.is_dir():
        raise DatasetValidationError(
            f"Dataset path does not exist or is not a directory: {directory}"
        )

    structure = detect_structure(directory)
    files = discover_files(directory, structure)

    caption_required = config.controls.text.required
    reference_required = config.controls.images.reference.required

    samples, orphaned = pair_samples(
        targets=files["targets"],
        captions=files["captions"],
        references=files["references"],
        caption_required=caption_required,
        reference_required=reference_required,
    )

    # Dataset-level issues
    dataset_issues: list[ValidationIssue] = []

    if not samples:
        dataset_issues.append(ValidationIssue(
            code=IssueCode.DATASET_EMPTY,
            severity=Severity.ERROR,
            message=(
                f"No video files found in '{directory}'. "
                f"Expected video files ({', '.join(sorted(VIDEO_EXTENSIONS)[:5])}...) "
                f"in {'training/targets/' if structure == StructureType.DIMLJUS else 'the directory'}."
            ),
            field="dataset",
        ))

    if orphaned:
        for orphan in orphaned:
            dataset_issues.append(ValidationIssue(
                code=IssueCode.ORPHANED_FILE,
                severity=Severity.WARNING,
                message=(
                    f"File '{orphan.name}' has no matching target video. "
                    f"Expected a video file with stem '{orphan.stem}'. "
                    f"This file will be ignored during training."
                ),
                field="dataset",
            ))

    return DatasetValidation(
        source_path=directory,
        structure=structure,
        samples=samples,
        orphaned_files=orphaned,
        dataset_issues=dataset_issues,
    )


def discover_all_datasets(
    config: KlippbokDataConfig,
    config_dir: Path | None = None,
) -> list[DatasetValidation]:
    """Discover all dataset sources defined in a config.

    Resolves relative paths from config_dir (the directory containing
    the config file).

    Args:
        config: The data config with datasets[] entries.
        config_dir: Base directory for resolving relative paths.
            Defaults to current working directory.

    Returns:
        List of DatasetValidation, one per source folder.
    """
    if config_dir is None:
        config_dir = Path.cwd()
    config_dir = Path(config_dir).resolve()

    results: list[DatasetValidation] = []
    for ds in config.datasets:
        ds_path = Path(ds.path)
        if not ds_path.is_absolute():
            ds_path = config_dir / ds_path
        ds_path = ds_path.resolve()
        results.append(discover_dataset(ds_path, config))

    return results
