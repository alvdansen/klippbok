"""Tests for klippbok.dataset.manifest — manifest generation and reading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from klippbok.config.data_schema import KlippbokDataConfig
from klippbok.dataset.manifest import (
    build_manifest,
    read_manifest,
    write_manifest,
)
from klippbok.dataset.models import (
    DatasetReport,
    DatasetValidation,
    SamplePair,
    StructureType,
)
from klippbok.video.models import IssueCode, Severity, ValidationIssue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(name: str | None = None, use_case: str | None = None) -> KlippbokDataConfig:
    cfg = {"datasets": [{"path": "."}]}
    if name:
        cfg["dataset"] = {"name": name}
    if use_case:
        cfg.setdefault("dataset", {})["use_case"] = use_case
    return KlippbokDataConfig(**cfg)


def _make_sample(stem: str, base: Path, issues: list[ValidationIssue] | None = None) -> SamplePair:
    return SamplePair(
        stem=stem,
        target=base / f"{stem}.mp4",
        caption=base / f"{stem}.txt",
        reference=base / f"{stem}.png",
        issues=issues or [],
    )


def _make_report(
    base: Path,
    n_valid: int = 2,
    n_invalid: int = 0,
) -> DatasetReport:
    error = ValidationIssue(
        code=IssueCode.CAPTION_MISSING,
        severity=Severity.ERROR,
        message="Missing caption",
        field="caption",
    )
    samples = []
    for i in range(n_valid):
        samples.append(_make_sample(f"good_{i}", base))
    for i in range(n_invalid):
        samples.append(_make_sample(f"bad_{i}", base, issues=[error]))

    ds = DatasetValidation(
        source_path=base,
        structure=StructureType.FLAT,
        samples=samples,
    )
    return DatasetReport(datasets=[ds])


# ---------------------------------------------------------------------------
# build_manifest tests
# ---------------------------------------------------------------------------

class TestBuildManifest:
    def test_basic_structure(self, tmp_path: Path):
        report = _make_report(tmp_path, n_valid=3)
        config = _make_config(name="test_dataset")
        manifest = build_manifest(report, config)

        assert manifest["klippbok_version"] == "0.1.0"
        assert "generated_at" in manifest
        assert manifest["dataset_name"] == "test_dataset"
        assert manifest["summary"]["total_samples"] == 3
        assert manifest["summary"]["is_valid"] is True

    def test_summary_counts(self, tmp_path: Path):
        report = _make_report(tmp_path, n_valid=2, n_invalid=1)
        manifest = build_manifest(report, _make_config())

        assert manifest["summary"]["total_samples"] == 3
        assert manifest["summary"]["valid_samples"] == 2
        assert manifest["summary"]["invalid_samples"] == 1
        assert manifest["summary"]["errors"] == 1
        assert manifest["summary"]["is_valid"] is False

    def test_only_issues_in_samples(self, tmp_path: Path):
        """Samples with no issues are NOT listed individually."""
        report = _make_report(tmp_path, n_valid=5, n_invalid=1)
        manifest = build_manifest(report, _make_config())

        ds = manifest["datasets"][0]
        # Only the 1 invalid sample should appear
        assert len(ds["sample_issues"]) == 1
        assert ds["sample_issues"][0]["stem"] == "bad_0"

    def test_extensions_collected(self, tmp_path: Path):
        report = _make_report(tmp_path, n_valid=2)
        manifest = build_manifest(report, _make_config())

        ds = manifest["datasets"][0]
        assert ".mp4" in ds["extensions"]["targets"]
        assert ".txt" in ds["extensions"]["captions"]
        assert ".png" in ds["extensions"]["references"]

    def test_use_case_included(self, tmp_path: Path):
        report = _make_report(tmp_path)
        config = _make_config(name="annika", use_case="character")
        manifest = build_manifest(report, config)
        assert manifest["use_case"] == "character"

    def test_cross_dataset_issues(self, tmp_path: Path):
        cross_issue = ValidationIssue(
            code=IssueCode.DUPLICATE_DETECTED,
            severity=Severity.WARNING,
            message="Cross-dataset duplicate",
            field="quality",
        )
        report = DatasetReport(
            datasets=[],
            cross_dataset_issues=[cross_issue],
        )
        manifest = build_manifest(report, _make_config())
        assert len(manifest["cross_dataset_issues"]) == 1
        assert manifest["cross_dataset_issues"][0]["code"] == "DUPLICATE_DETECTED"

    def test_info_issues_excluded(self, tmp_path: Path):
        """Info-severity issues are not included in the manifest."""
        info_issue = ValidationIssue(
            code=IssueCode.RESOLUTION_ABOVE_TARGET,
            severity=Severity.INFO,
            message="Above target",
            field="resolution",
        )
        sample = _make_sample("clip", tmp_path, issues=[info_issue])
        ds = DatasetValidation(
            source_path=tmp_path,
            structure=StructureType.FLAT,
            samples=[sample],
        )
        report = DatasetReport(datasets=[ds])
        manifest = build_manifest(report, _make_config())
        assert len(manifest["datasets"][0]["sample_issues"]) == 0

    def test_dataset_level_issues(self, tmp_path: Path):
        ds_issue = ValidationIssue(
            code=IssueCode.DATASET_EMPTY,
            severity=Severity.ERROR,
            message="No samples",
            field="dataset",
        )
        ds = DatasetValidation(
            source_path=tmp_path,
            structure=StructureType.FLAT,
            dataset_issues=[ds_issue],
        )
        report = DatasetReport(datasets=[ds])
        manifest = build_manifest(report, _make_config())
        assert len(manifest["datasets"][0]["dataset_issues"]) == 1

    def test_structure_type_in_manifest(self, tmp_path: Path):
        ds = DatasetValidation(
            source_path=tmp_path,
            structure=StructureType.DIMLJUS,
            samples=[_make_sample("a", tmp_path)],
        )
        report = DatasetReport(datasets=[ds])
        manifest = build_manifest(report, _make_config())
        assert manifest["datasets"][0]["structure"] == "klippbok"


# ---------------------------------------------------------------------------
# write_manifest + read_manifest round-trip
# ---------------------------------------------------------------------------

class TestManifestRoundTrip:
    def test_write_and_read(self, tmp_path: Path):
        report = _make_report(tmp_path, n_valid=3, n_invalid=1)
        config = _make_config(name="test")
        out = tmp_path / "klippbok_manifest.json"

        written = write_manifest(report, config, out)
        assert written == out
        assert out.exists()

        loaded = read_manifest(out)
        assert loaded["dataset_name"] == "test"
        assert loaded["summary"]["total_samples"] == 4

    def test_valid_json(self, tmp_path: Path):
        report = _make_report(tmp_path)
        out = tmp_path / "manifest.json"
        write_manifest(report, _make_config(), out)
        # Should parse as valid JSON
        text = out.read_text(encoding="utf-8")
        parsed = json.loads(text)
        assert isinstance(parsed, dict)

    def test_read_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_manifest(tmp_path / "nonexistent.json")

    def test_utf8_encoding(self, tmp_path: Path):
        """Manifest handles non-ASCII characters."""
        report = _make_report(tmp_path)
        config = _make_config(name="annike")
        out = tmp_path / "manifest.json"
        write_manifest(report, config, out)
        loaded = read_manifest(out)
        assert loaded["dataset_name"] == "annike"
