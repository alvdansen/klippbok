"""Core dataset organize engine.

Takes validated data (messy output from ingest/triage/caption pipeline)
and produces a clean, trainer-ready directory. Optionally generates
trainer config files so the user can start training immediately.

This is the last data-preparation step before training. Without it,
users have to manually arrange files and write trainer configs by hand.

Usage:
    from klippbok.dataset.organize import organize_dataset

    result = organize_dataset(
        source_dir=Path("./clips"),
        output_dir=Path("./clean"),
        trainers=["musubi"],
    )

Flow:
    1. Validate source (reuses existing discovery + validation)
    2. Filter samples: keep valid ones (no errors)
    3. Build output layout (flat or klippbok)
    4. Copy/move files to destination
    5. Generate trainer configs if requested
    6. Return OrganizeResult
"""

from __future__ import annotations

import shutil
from pathlib import Path

from klippbok.config.data_schema import KlippbokDataConfig
from klippbok.dataset.errors import OrganizeError
from klippbok.dataset.models import (
    OrganizeLayout,
    OrganizedSample,
    OrganizeResult,
    SamplePair,
)
from klippbok.dataset.validate import validate_all
from klippbok.video.models import Severity


# ---------------------------------------------------------------------------
# File transfer
# ---------------------------------------------------------------------------

def _transfer_file(source: Path, dest: Path, *, copy: bool = True) -> None:
    """Copy or move a single file, creating parent dirs as needed.

    Skips silently if source == dest (idempotent re-runs).
    Overwrites existing files at dest (idempotent).

    Args:
        source: Path to the source file.
        dest: Where to place the file.
        copy: True to copy, False to move.
    """
    if source.resolve() == dest.resolve():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    if copy:
        shutil.copy2(source, dest)
    else:
        shutil.move(str(source), str(dest))


# ---------------------------------------------------------------------------
# Name collision handling
# ---------------------------------------------------------------------------

def _resolve_collision(dest: Path, used_stems: set[str]) -> Path:
    """If dest stem is already used, append a numeric suffix.

    Rare case — only happens when multiple source folders have
    samples with the same stem. Appends _2, _3, etc.

    Args:
        dest: Proposed destination path.
        used_stems: Set of stems already claimed in this organize run.

    Returns:
        A unique destination path (may be unchanged if no collision).
    """
    stem = dest.stem
    if stem not in used_stems:
        used_stems.add(stem)
        return dest

    # Find next available suffix
    counter = 2
    while f"{stem}_{counter}" in used_stems:
        counter += 1
    new_stem = f"{stem}_{counter}"
    used_stems.add(new_stem)
    return dest.with_stem(new_stem)


# ---------------------------------------------------------------------------
# Layout builders
# ---------------------------------------------------------------------------

def _build_flat_paths(
    sample: SamplePair,
    output_dir: Path,
    used_stems: set[str],
) -> tuple[Path, Path | None, Path | None]:
    """Compute destination paths for flat layout.

    All files go in output_dir root, stem-matched.

    Returns:
        (target_dest, caption_dest, reference_dest)
    """
    target_dest = _resolve_collision(
        output_dir / f"{sample.stem}{sample.target.suffix}",
        used_stems,
    )
    actual_stem = target_dest.stem

    caption_dest = None
    if sample.caption is not None:
        caption_dest = output_dir / f"{actual_stem}{sample.caption.suffix}"

    reference_dest = None
    if sample.reference is not None:
        reference_dest = output_dir / f"{actual_stem}{sample.reference.suffix}"

    return target_dest, caption_dest, reference_dest


def _build_klippbok_paths(
    sample: SamplePair,
    output_dir: Path,
    used_stems: set[str],
) -> tuple[Path, Path | None, Path | None]:
    """Compute destination paths for klippbok hierarchical layout.

    targets -> training/targets/
    captions -> training/signals/captions/
    references -> training/signals/references/

    Returns:
        (target_dest, caption_dest, reference_dest)
    """
    targets_dir = output_dir / "training" / "targets"
    captions_dir = output_dir / "training" / "signals" / "captions"
    references_dir = output_dir / "training" / "signals" / "references"

    target_dest = _resolve_collision(
        targets_dir / f"{sample.stem}{sample.target.suffix}",
        used_stems,
    )
    actual_stem = target_dest.stem

    caption_dest = None
    if sample.caption is not None:
        caption_dest = captions_dir / f"{actual_stem}{sample.caption.suffix}"

    reference_dest = None
    if sample.reference is not None:
        reference_dest = references_dir / f"{actual_stem}{sample.reference.suffix}"

    return target_dest, caption_dest, reference_dest


# ---------------------------------------------------------------------------
# Main organize function
# ---------------------------------------------------------------------------

def organize_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    layout: OrganizeLayout = OrganizeLayout.FLAT,
    config: KlippbokDataConfig | None = None,
    copy: bool = True,
    include_warnings: bool = True,
    dry_run: bool = False,
    trainers: list[str] | None = None,
) -> OrganizeResult:
    """Organize a validated dataset into a clean, trainer-ready directory.

    This is the main entry point for the organize command. It:
    1. Validates the source directory (reuses existing pipeline)
    2. Filters samples (exclude errors, optionally exclude warnings)
    3. Copies/moves files into the chosen layout
    4. Generates trainer config files if requested
    5. Returns a complete result for reporting

    Args:
        source_dir: Path to the source dataset folder.
        output_dir: Where to place organized files.
        layout: FLAT (default) or DIMLJUS hierarchical structure.
        config: Data config for validation. If None, uses defaults.
        copy: True to copy files (default), False to move them.
        include_warnings: True (default) to include samples that have
            warnings but no errors. False to also exclude warnings (--strict).
        dry_run: True to preview without touching files.
        trainers: List of trainer names to generate configs for
            (e.g. ["musubi", "aitoolkit"]).

    Returns:
        OrganizeResult with organized/skipped samples and config paths.

    Raises:
        OrganizeError: If source doesn't exist or zero valid samples.
    """
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not source_dir.is_dir():
        raise OrganizeError(
            f"Source directory does not exist: {source_dir}\n"
            f"Check the path and try again."
        )

    # Build config if not provided
    if config is None:
        config = KlippbokDataConfig(
            datasets=[{"path": str(source_dir)}],
        )

    # Validate the source dataset
    report = validate_all(config, config_dir=source_dir)

    # Collect all samples across all sources
    all_samples: list[SamplePair] = []
    for ds in report.datasets:
        all_samples.extend(ds.samples)

    # Filter samples
    organized_samples: list[OrganizedSample] = []
    skipped_samples: list[OrganizedSample] = []
    valid_samples: list[SamplePair] = []

    for sample in all_samples:
        has_errors = any(i.severity == Severity.ERROR for i in sample.issues)
        has_warnings = any(i.severity == Severity.WARNING for i in sample.issues)

        if has_errors:
            # Always skip samples with errors
            reasons = [i.message for i in sample.issues if i.severity == Severity.ERROR]
            skipped_samples.append(OrganizedSample(
                stem=sample.stem,
                skipped=True,
                skip_reason=reasons[0] if len(reasons) == 1 else f"{len(reasons)} errors",
                frame_count=sample.frame_count,
                width=sample.width,
                height=sample.height,
            ))
        elif has_warnings and not include_warnings:
            # --strict mode: also skip warnings
            reasons = [i.message for i in sample.issues if i.severity == Severity.WARNING]
            skipped_samples.append(OrganizedSample(
                stem=sample.stem,
                skipped=True,
                skip_reason=f"strict mode: {reasons[0]}" if len(reasons) == 1 else f"strict mode: {len(reasons)} warnings",
                frame_count=sample.frame_count,
                width=sample.width,
                height=sample.height,
            ))
        else:
            valid_samples.append(sample)

    if not valid_samples:
        raise OrganizeError(
            f"No valid samples to organize in '{source_dir}'.\n"
            f"Found {len(all_samples)} sample(s), but all were excluded due to errors"
            + (" or warnings (--strict mode)" if not include_warnings else "")
            + ".\n"
            f"Run 'python -m klippbok.dataset validate {source_dir}' to see issues."
        )

    # Build paths and transfer files
    used_stems: set[str] = set()
    path_builder = _build_klippbok_paths if layout == OrganizeLayout.DIMLJUS else _build_flat_paths

    for sample in valid_samples:
        target_dest, caption_dest, reference_dest = path_builder(
            sample, output_dir, used_stems,
        )

        if not dry_run:
            _transfer_file(sample.target, target_dest, copy=copy)
            if caption_dest is not None and sample.caption is not None:
                _transfer_file(sample.caption, caption_dest, copy=copy)
            if reference_dest is not None and sample.reference is not None:
                _transfer_file(sample.reference, reference_dest, copy=copy)

        organized_samples.append(OrganizedSample(
            stem=target_dest.stem,
            target_dest=target_dest,
            caption_dest=caption_dest,
            reference_dest=reference_dest,
            frame_count=sample.frame_count,
            width=sample.width,
            height=sample.height,
        ))

    # Generate trainer configs
    trainer_config_paths: list[Path] = []
    if trainers:
        from klippbok.dataset.trainers import generate_trainer_config

        for trainer_name in trainers:
            config_path = generate_trainer_config(
                trainer_name=trainer_name,
                samples=organized_samples,
                output_dir=output_dir,
                config=config,
                layout=layout,
                dry_run=dry_run,
            )
            trainer_config_paths.append(config_path)

    return OrganizeResult(
        output_dir=output_dir,
        layout=layout,
        organized=organized_samples,
        skipped=skipped_samples,
        trainer_configs=trainer_config_paths,
        dry_run=dry_run,
    )
