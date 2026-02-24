"""Manifest generation and reading for validated datasets.

The manifest (klippbok_manifest.json) is a lightweight summary of a validated
dataset. It describes:
- What folder structure was detected
- What file extensions are present
- Per-sample issues (only errors and warnings, not info)
- Aggregate statistics

The manifest is designed to be:
- Small: folder descriptions + extensions, not individual file listings
- Portable: relative paths from manifest location
- Useful: downstream tools can read it to understand the dataset

The manifest does NOT replace the data config — it's a snapshot of validation
results at a point in time, not a source of truth for configuration.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from klippbok.config.data_schema import KlippbokDataConfig
from klippbok.dataset.models import DatasetReport, DatasetValidation, SamplePair
from klippbok.video.models import Severity


def _sample_to_dict(sample: SamplePair, base_path: Path) -> dict | None:
    """Convert a sample to a manifest dict, only if it has issues.

    Samples with no issues are not included individually — the manifest
    stays compact by only recording problems.
    """
    # Only include samples with errors or warnings
    relevant_issues = [
        i for i in sample.issues
        if i.severity in (Severity.ERROR, Severity.WARNING)
    ]
    if not relevant_issues:
        return None

    return {
        "stem": sample.stem,
        "target": str(sample.target.relative_to(base_path)) if _is_relative(sample.target, base_path) else str(sample.target),
        "issues": [
            {
                "code": issue.code.value,
                "severity": issue.severity.value,
                "message": issue.message,
            }
            for issue in relevant_issues
        ],
    }


def _is_relative(path: Path, base: Path) -> bool:
    """Check if path is under base (safe relative_to)."""
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _dataset_to_dict(ds: DatasetValidation) -> dict:
    """Convert a DatasetValidation to manifest dict."""
    base = ds.source_path

    # Collect unique extensions per role
    target_exts: set[str] = set()
    caption_exts: set[str] = set()
    reference_exts: set[str] = set()
    for s in ds.samples:
        target_exts.add(s.target.suffix.lower())
        if s.caption:
            caption_exts.add(s.caption.suffix.lower())
        if s.reference:
            reference_exts.add(s.reference.suffix.lower())

    # Per-sample issues (only problematic samples)
    sample_issues = []
    for s in ds.samples:
        entry = _sample_to_dict(s, base)
        if entry is not None:
            sample_issues.append(entry)

    # Dataset-level issues
    ds_issues = [
        {
            "code": i.code.value,
            "severity": i.severity.value,
            "message": i.message,
        }
        for i in ds.dataset_issues
        if i.severity in (Severity.ERROR, Severity.WARNING)
    ]

    return {
        "source_path": str(base),
        "structure": ds.structure.value,
        "total_samples": ds.total_samples,
        "valid_samples": ds.valid_samples,
        "invalid_samples": ds.invalid_samples,
        "extensions": {
            "targets": sorted(target_exts),
            "captions": sorted(caption_exts),
            "references": sorted(reference_exts),
        },
        "dataset_issues": ds_issues,
        "sample_issues": sample_issues,
    }


def build_manifest(report: DatasetReport, config: KlippbokDataConfig) -> dict:
    """Build a manifest dict from a validation report.

    The manifest is a JSON-serializable dict that summarizes the
    validation results. It's designed to be small and portable.

    Args:
        report: The validation report.
        config: The data config (for metadata).

    Returns:
        Dict ready for json.dump().
    """
    return {
        "klippbok_version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_sources": report.total_sources,
            "total_samples": report.total_samples,
            "valid_samples": report.valid_samples,
            "invalid_samples": report.invalid_samples,
            "errors": report.error_count,
            "warnings": report.warning_count,
            "is_valid": report.is_valid,
        },
        "dataset_name": config.dataset.name,
        "use_case": config.dataset.use_case,
        "datasets": [_dataset_to_dict(ds) for ds in report.datasets],
        "cross_dataset_issues": [
            {
                "code": i.code.value,
                "severity": i.severity.value,
                "message": i.message,
            }
            for i in report.cross_dataset_issues
        ],
    }


def write_manifest(
    report: DatasetReport,
    config: KlippbokDataConfig,
    output_path: str | Path,
) -> Path:
    """Write the manifest to a JSON file.

    Args:
        report: The validation report.
        config: The data config.
        output_path: Where to write the manifest file.

    Returns:
        Path to the written manifest file.
    """
    output_path = Path(output_path)
    manifest = build_manifest(report, config)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def read_manifest(path: str | Path) -> dict:
    """Read a manifest from a JSON file.

    Args:
        path: Path to the manifest file.

    Returns:
        Parsed manifest dict.

    Raises:
        FileNotFoundError: if the manifest file doesn't exist.
        json.JSONDecodeError: if the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
