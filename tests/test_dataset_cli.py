"""Tests for klippbok.dataset CLI and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from klippbok.dataset.__main__ import build_parser, cmd_validate
from klippbok.dataset.models import (
    DatasetReport,
    DatasetValidation,
    SamplePair,
    StructureType,
)
from klippbok.dataset.report import (
    format_bucketing_plaintext,
    format_report_plaintext,
)
from klippbok.dataset.bucketing import BucketGroup, BucketingResult
from klippbok.video.models import IssueCode, Severity, ValidationIssue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _touch(path: Path, content: bytes = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _make_textured_image(path: Path, size: int = 64) -> Path:
    rng = np.random.RandomState(42)
    img = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def _make_flat_dataset(tmp_path: Path, stems: list[str]) -> Path:
    for stem in stems:
        _touch(tmp_path / f"{stem}.mp4")
        _touch(tmp_path / f"{stem}.txt", f"Caption for {stem}.".encode())
        _make_textured_image(tmp_path / f"{stem}.png")
    return tmp_path


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParser:
    def test_validate_command(self):
        parser = build_parser()
        args = parser.parse_args(["validate", "/data"])
        assert args.command == "validate"
        assert args.path == "/data"

    def test_manifest_flag(self):
        parser = build_parser()
        args = parser.parse_args(["validate", "/data", "--manifest"])
        assert args.manifest is True

    def test_buckets_flag(self):
        parser = build_parser()
        args = parser.parse_args(["validate", "/data", "--buckets"])
        assert args.buckets is True

    def test_quality_flag(self):
        parser = build_parser()
        args = parser.parse_args(["validate", "/data", "--quality"])
        assert args.quality is True

    def test_duplicates_flag(self):
        parser = build_parser()
        args = parser.parse_args(["validate", "/data", "--duplicates"])
        assert args.duplicates is True

    def test_json_flag(self):
        parser = build_parser()
        args = parser.parse_args(["validate", "/data", "--json"])
        assert args.json_output is True

    def test_config_flag(self):
        parser = build_parser()
        args = parser.parse_args(["validate", "/data", "--config", "my_config.yaml"])
        assert args.config == "my_config.yaml"

    def test_all_flags_combined(self):
        parser = build_parser()
        args = parser.parse_args([
            "validate", "/data",
            "--manifest", "--buckets", "--quality", "--duplicates", "--json",
        ])
        assert args.manifest is True
        assert args.buckets is True
        assert args.quality is True
        assert args.duplicates is True
        assert args.json_output is True


# ---------------------------------------------------------------------------
# End-to-end CLI tests
# ---------------------------------------------------------------------------

class TestCmdValidate:
    def test_valid_dataset_returns_0(self, tmp_path: Path):
        _make_flat_dataset(tmp_path, ["a", "b"])
        parser = build_parser()
        args = parser.parse_args(["validate", str(tmp_path)])
        result = cmd_validate(args)
        assert result == 0

    def test_empty_dataset_returns_1(self, tmp_path: Path):
        parser = build_parser()
        args = parser.parse_args(["validate", str(tmp_path)])
        result = cmd_validate(args)
        assert result == 1  # DATASET_EMPTY is an error

    def test_json_output(self, tmp_path: Path, capsys):
        _make_flat_dataset(tmp_path, ["a"])
        parser = build_parser()
        args = parser.parse_args(["validate", str(tmp_path), "--json"])
        cmd_validate(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "summary" in parsed
        assert parsed["summary"]["total_samples"] == 1

    def test_manifest_written(self, tmp_path: Path):
        _make_flat_dataset(tmp_path, ["a"])
        parser = build_parser()
        args = parser.parse_args(["validate", str(tmp_path), "--manifest"])
        cmd_validate(args)
        manifest = tmp_path / "klippbok_manifest.json"
        assert manifest.exists()
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["summary"]["total_samples"] == 1

    def test_quality_flag_enables_checks(self, tmp_path: Path):
        """--quality flag should enable blur and exposure checks."""
        _make_flat_dataset(tmp_path, ["a"])
        parser = build_parser()
        args = parser.parse_args(["validate", str(tmp_path), "--quality", "--json"])
        cmd_validate(args)
        # Should not crash — quality checks run on the textured images

    def test_duplicates_flag(self, tmp_path: Path):
        """--duplicates flag enables duplicate detection."""
        # Create two clips with identical reference images
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        _touch(tmp_path / "a.mp4")
        _touch(tmp_path / "a.txt", b"Caption a")
        cv2.imwrite(str(tmp_path / "a.png"), img)
        _touch(tmp_path / "b.mp4")
        _touch(tmp_path / "b.txt", b"Caption b")
        cv2.imwrite(str(tmp_path / "b.png"), img)

        parser = build_parser()
        args = parser.parse_args(["validate", str(tmp_path), "--duplicates", "--json"])
        cmd_validate(args)
        # Should not crash


# ---------------------------------------------------------------------------
# Report formatting tests
# ---------------------------------------------------------------------------

class TestPlaintextReport:
    def _make_report(self, n_valid: int = 2, n_invalid: int = 0) -> DatasetReport:
        error = ValidationIssue(
            code=IssueCode.CAPTION_MISSING,
            severity=Severity.ERROR,
            message="Missing caption",
            field="caption",
        )
        samples = []
        for i in range(n_valid):
            samples.append(SamplePair(stem=f"good_{i}", target=Path(f"g{i}.mp4")))
        for i in range(n_invalid):
            samples.append(SamplePair(
                stem=f"bad_{i}", target=Path(f"b{i}.mp4"), issues=[error],
            ))
        ds = DatasetValidation(
            source_path=Path("/data"),
            structure=StructureType.FLAT,
            samples=samples,
        )
        return DatasetReport(datasets=[ds])

    def test_contains_header(self):
        report = self._make_report()
        text = format_report_plaintext(report)
        assert "Dataset Validation Report" in text

    def test_contains_summary(self):
        report = self._make_report(n_valid=3, n_invalid=1)
        text = format_report_plaintext(report)
        assert "Total samples:   4" in text
        assert "Valid:           3" in text
        assert "Invalid:         1" in text

    def test_pass_status(self):
        report = self._make_report(n_valid=3)
        text = format_report_plaintext(report)
        assert "PASS" in text

    def test_fail_status(self):
        report = self._make_report(n_invalid=1)
        text = format_report_plaintext(report)
        assert "FAIL" in text

    def test_issue_details(self):
        report = self._make_report(n_invalid=1)
        text = format_report_plaintext(report)
        assert "bad_0" in text
        assert "Missing caption" in text
        assert "[ERROR]" in text

    def test_all_clear_message(self):
        report = self._make_report(n_valid=2)
        text = format_report_plaintext(report)
        assert "All samples passed" in text

    def test_ascii_safe(self):
        """Output should contain no non-ASCII characters."""
        report = self._make_report(n_valid=2, n_invalid=1)
        text = format_report_plaintext(report)
        text.encode("ascii")  # raises if non-ASCII

    def test_empty_dataset_message(self):
        report = DatasetReport()
        text = format_report_plaintext(report)
        assert "No samples found" in text


class TestBucketingPlaintext:
    def test_basic_output(self):
        result = BucketingResult(
            buckets=[
                BucketGroup(bucket_key="320x240x17", count=5, samples=["a", "b", "c", "d", "e"]),
                BucketGroup(bucket_key="640x480x17", count=2, samples=["f", "g"]),
            ],
            step_size=16,
        )
        text = format_bucketing_plaintext(result)
        assert "Bucketing Preview" in text
        assert "320x240x17" in text
        assert "640x480x17" in text

    def test_truncated_samples(self):
        """When >5 samples in a bucket, show count."""
        result = BucketingResult(
            buckets=[
                BucketGroup(
                    bucket_key="320x240x17",
                    count=8,
                    samples=["a", "b", "c", "d", "e", "f", "g", "h"],
                ),
            ],
            step_size=16,
        )
        text = format_bucketing_plaintext(result)
        assert "(+3 more)" in text
