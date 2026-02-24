"""Tests for klippbok.dataset.models and related issue codes.

Tests pure Python data models — no I/O, no ffmpeg, no external dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from klippbok.config.data_schema import QualityConfig
from klippbok.dataset.errors import DatasetValidationError, KlippbokDatasetError
from klippbok.dataset.models import (
    DatasetReport,
    DatasetValidation,
    SamplePair,
    StructureType,
)
from klippbok.video.models import IssueCode, Severity, ValidationIssue


# ---------------------------------------------------------------------------
# StructureType enum
# ---------------------------------------------------------------------------

class TestStructureType:
    def test_flat_value(self):
        assert StructureType.FLAT == "flat"

    def test_klippbok_value(self):
        assert StructureType.DIMLJUS == "klippbok"

    def test_from_string(self):
        assert StructureType("flat") == StructureType.FLAT
        assert StructureType("klippbok") == StructureType.DIMLJUS


# ---------------------------------------------------------------------------
# New IssueCode values exist
# ---------------------------------------------------------------------------

class TestNewIssueCodes:
    """Verify all Phase 4 issue codes were added to the enum."""

    @pytest.mark.parametrize("code", [
        "CAPTION_MISSING",
        "CAPTION_EMPTY",
        "CAPTION_TOO_LONG",
        "REFERENCE_MISSING",
        "REFERENCE_BLANK",
        "FILE_TYPE_INVALID",
        "FILE_CORRUPTED",
        "DUPLICATE_DETECTED",
        "ORPHANED_FILE",
        "EXPOSURE_OUT_OF_RANGE",
        "BLUR_BELOW_THRESHOLD",
        "MOTION_BELOW_MIN",
        "MOTION_ABOVE_MAX",
        "DATASET_EMPTY",
        "BUCKET_UNDERSIZED",
    ])
    def test_issue_code_exists(self, code: str):
        """Each Phase 4 issue code must be a valid IssueCode member."""
        assert hasattr(IssueCode, code)
        assert IssueCode[code].value == code


# ---------------------------------------------------------------------------
# SamplePair
# ---------------------------------------------------------------------------

class TestSamplePair:
    def test_minimal_creation(self):
        """A sample only needs a stem and a target path."""
        sample = SamplePair(
            stem="clip_001",
            target=Path("/data/clip_001.mp4"),
        )
        assert sample.stem == "clip_001"
        assert sample.target == Path("/data/clip_001.mp4")
        assert sample.caption is None
        assert sample.reference is None
        assert sample.issues == []

    def test_full_creation(self):
        """A sample with all optional fields."""
        sample = SamplePair(
            stem="clip_001",
            target=Path("/data/clip_001.mp4"),
            caption=Path("/data/clip_001.txt"),
            reference=Path("/data/clip_001.png"),
            width=1280,
            height=720,
            frame_count=17,
            fps=16.0,
        )
        assert sample.has_caption is True
        assert sample.has_reference is True
        assert sample.width == 1280
        assert sample.height == 720
        assert sample.frame_count == 17
        assert sample.fps == 16.0

    def test_is_valid_no_issues(self):
        sample = SamplePair(stem="a", target=Path("a.mp4"))
        assert sample.is_valid is True

    def test_is_valid_with_warning(self):
        """Warnings don't make a sample invalid."""
        issue = ValidationIssue(
            code=IssueCode.CAPTION_TOO_LONG,
            severity=Severity.WARNING,
            message="Caption exceeds max tokens",
            field="caption",
        )
        sample = SamplePair(stem="a", target=Path("a.mp4"), issues=[issue])
        assert sample.is_valid is True
        assert len(sample.warnings) == 1
        assert len(sample.errors) == 0

    def test_is_valid_with_error(self):
        """Errors make a sample invalid."""
        issue = ValidationIssue(
            code=IssueCode.CAPTION_MISSING,
            severity=Severity.ERROR,
            message="No caption found for clip_001",
            field="caption",
        )
        sample = SamplePair(stem="a", target=Path("a.mp4"), issues=[issue])
        assert sample.is_valid is False
        assert len(sample.errors) == 1

    def test_has_caption_false(self):
        sample = SamplePair(stem="a", target=Path("a.mp4"), caption=None)
        assert sample.has_caption is False

    def test_has_reference_false(self):
        sample = SamplePair(stem="a", target=Path("a.mp4"), reference=None)
        assert sample.has_reference is False

    def test_frozen(self):
        """SamplePair is immutable."""
        sample = SamplePair(stem="a", target=Path("a.mp4"))
        with pytest.raises(ValidationError):
            sample.stem = "b"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DatasetValidation
# ---------------------------------------------------------------------------

class TestDatasetValidation:
    def _make_sample(
        self,
        stem: str = "clip",
        issues: list[ValidationIssue] | None = None,
    ) -> SamplePair:
        return SamplePair(
            stem=stem,
            target=Path(f"/data/{stem}.mp4"),
            issues=issues or [],
        )

    def test_empty_dataset(self):
        dv = DatasetValidation(
            source_path=Path("/data"),
            structure=StructureType.FLAT,
        )
        assert dv.total_samples == 0
        assert dv.valid_samples == 0
        assert dv.invalid_samples == 0
        assert dv.is_valid is True

    def test_all_valid(self):
        dv = DatasetValidation(
            source_path=Path("/data"),
            structure=StructureType.FLAT,
            samples=[self._make_sample("a"), self._make_sample("b")],
        )
        assert dv.total_samples == 2
        assert dv.valid_samples == 2
        assert dv.invalid_samples == 0
        assert dv.is_valid is True

    def test_some_invalid(self):
        error = ValidationIssue(
            code=IssueCode.CAPTION_MISSING,
            severity=Severity.ERROR,
            message="Missing caption",
            field="caption",
        )
        dv = DatasetValidation(
            source_path=Path("/data"),
            structure=StructureType.DIMLJUS,
            samples=[
                self._make_sample("good"),
                self._make_sample("bad", issues=[error]),
            ],
        )
        assert dv.total_samples == 2
        assert dv.valid_samples == 1
        assert dv.invalid_samples == 1
        assert dv.is_valid is False
        assert dv.error_count == 1
        assert dv.warning_count == 0

    def test_dataset_level_issues(self):
        """Dataset-level issues (like DATASET_EMPTY) also count."""
        issue = ValidationIssue(
            code=IssueCode.DATASET_EMPTY,
            severity=Severity.ERROR,
            message="No training samples found",
            field="dataset",
        )
        dv = DatasetValidation(
            source_path=Path("/data"),
            structure=StructureType.FLAT,
            dataset_issues=[issue],
        )
        assert dv.is_valid is False
        assert dv.error_count == 1

    def test_orphaned_files(self):
        dv = DatasetValidation(
            source_path=Path("/data"),
            structure=StructureType.FLAT,
            orphaned_files=[Path("/data/stray.log"), Path("/data/notes.md")],
        )
        assert len(dv.orphaned_files) == 2

    def test_issue_summary(self):
        issues = [
            ValidationIssue(
                code=IssueCode.CAPTION_MISSING,
                severity=Severity.ERROR,
                message="missing",
                field="caption",
            ),
            ValidationIssue(
                code=IssueCode.CAPTION_MISSING,
                severity=Severity.ERROR,
                message="missing",
                field="caption",
            ),
            ValidationIssue(
                code=IssueCode.BLUR_BELOW_THRESHOLD,
                severity=Severity.WARNING,
                message="blurry",
                field="quality",
            ),
        ]
        dv = DatasetValidation(
            source_path=Path("/data"),
            structure=StructureType.FLAT,
            samples=[
                SamplePair(stem="a", target=Path("a.mp4"), issues=issues[:2]),
                SamplePair(stem="b", target=Path("b.mp4"), issues=issues[2:]),
            ],
        )
        summary = dv.issue_summary
        assert summary[IssueCode.CAPTION_MISSING] == 2
        assert summary[IssueCode.BLUR_BELOW_THRESHOLD] == 1

    def test_all_issues_includes_dataset_and_sample(self):
        ds_issue = ValidationIssue(
            code=IssueCode.DATASET_EMPTY,
            severity=Severity.ERROR,
            message="empty",
            field="dataset",
        )
        sample_issue = ValidationIssue(
            code=IssueCode.CAPTION_MISSING,
            severity=Severity.ERROR,
            message="no caption",
            field="caption",
        )
        dv = DatasetValidation(
            source_path=Path("/data"),
            structure=StructureType.FLAT,
            dataset_issues=[ds_issue],
            samples=[SamplePair(stem="a", target=Path("a.mp4"), issues=[sample_issue])],
        )
        assert len(dv.all_issues) == 2

    def test_frozen(self):
        dv = DatasetValidation(
            source_path=Path("/data"),
            structure=StructureType.FLAT,
        )
        with pytest.raises(ValidationError):
            dv.structure = StructureType.DIMLJUS  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DatasetReport
# ---------------------------------------------------------------------------

class TestDatasetReport:
    def _make_dataset(
        self,
        n_valid: int = 0,
        n_invalid: int = 0,
    ) -> DatasetValidation:
        error = ValidationIssue(
            code=IssueCode.CAPTION_MISSING,
            severity=Severity.ERROR,
            message="missing",
            field="caption",
        )
        samples = []
        for i in range(n_valid):
            samples.append(SamplePair(stem=f"good_{i}", target=Path(f"g{i}.mp4")))
        for i in range(n_invalid):
            samples.append(SamplePair(
                stem=f"bad_{i}",
                target=Path(f"b{i}.mp4"),
                issues=[error],
            ))
        return DatasetValidation(
            source_path=Path("/data"),
            structure=StructureType.FLAT,
            samples=samples,
        )

    def test_empty_report(self):
        report = DatasetReport()
        assert report.total_sources == 0
        assert report.total_samples == 0
        assert report.is_valid is True

    def test_single_source_all_valid(self):
        report = DatasetReport(datasets=[self._make_dataset(n_valid=5)])
        assert report.total_sources == 1
        assert report.total_samples == 5
        assert report.valid_samples == 5
        assert report.invalid_samples == 0
        assert report.is_valid is True

    def test_multi_source_mixed(self):
        report = DatasetReport(datasets=[
            self._make_dataset(n_valid=3, n_invalid=1),
            self._make_dataset(n_valid=2),
        ])
        assert report.total_sources == 2
        assert report.total_samples == 6
        assert report.valid_samples == 5
        assert report.invalid_samples == 1
        assert report.is_valid is False

    def test_cross_dataset_issues(self):
        issue = ValidationIssue(
            code=IssueCode.DUPLICATE_DETECTED,
            severity=Severity.WARNING,
            message="Duplicate across sources",
            field="quality",
        )
        report = DatasetReport(
            datasets=[self._make_dataset(n_valid=2)],
            cross_dataset_issues=[issue],
        )
        assert report.warning_count == 1
        assert len(report.all_issues) == 1

    def test_issue_summary_aggregates(self):
        report = DatasetReport(datasets=[
            self._make_dataset(n_invalid=2),
            self._make_dataset(n_invalid=1),
        ])
        summary = report.issue_summary
        assert summary[IssueCode.CAPTION_MISSING] == 3

    def test_frozen(self):
        report = DatasetReport()
        with pytest.raises(ValidationError):
            report.datasets = []  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TestErrors:
    def test_dataset_error_is_exception(self):
        assert issubclass(KlippbokDatasetError, Exception)

    def test_validation_error_inherits(self):
        assert issubclass(DatasetValidationError, KlippbokDatasetError)

    def test_validation_error_message(self):
        err = DatasetValidationError("path does not exist")
        assert "path does not exist" in str(err)
        assert err.detail == "path does not exist"


# ---------------------------------------------------------------------------
# QualityConfig.check_duplicates
# ---------------------------------------------------------------------------

class TestCheckDuplicatesConfig:
    def test_default_false(self):
        qc = QualityConfig()
        assert qc.check_duplicates is False

    def test_set_true(self):
        qc = QualityConfig(check_duplicates=True)
        assert qc.check_duplicates is True

    def test_roundtrip_yaml(self):
        """check_duplicates survives YAML serialization."""
        import yaml
        data = {"check_duplicates": True, "min_resolution": 480}
        qc = QualityConfig(**data)
        dumped = yaml.safe_dump(qc.model_dump())
        loaded = yaml.safe_load(dumped)
        assert loaded["check_duplicates"] is True
