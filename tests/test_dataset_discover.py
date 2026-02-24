"""Tests for klippbok.dataset.discover — file discovery and structure detection.

All tests use tmp_path with programmatically created files. No real videos
or ffmpeg needed — just file/directory structure matters.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from klippbok.config.data_schema import (
    ControlsConfig,
    KlippbokDataConfig,
    ImagesControlConfig,
    ReferenceImageConfig,
    TextControlConfig,
)
from klippbok.dataset.discover import (
    VIDEO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    detect_structure,
    discover_all_datasets,
    discover_dataset,
    discover_files,
    pair_samples,
    validate_file_type,
)
from klippbok.dataset.errors import DatasetValidationError
from klippbok.dataset.models import StructureType
from klippbok.video.models import IssueCode, Severity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _touch(path: Path, content: bytes = b"") -> Path:
    """Create a file with optional content, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _make_flat_dataset(tmp_path: Path, stems: list[str]) -> Path:
    """Create a flat dataset with video + caption + reference per stem."""
    for stem in stems:
        _touch(tmp_path / f"{stem}.mp4")
        _touch(tmp_path / f"{stem}.txt", b"A test caption.")
        _touch(tmp_path / f"{stem}.png")
    return tmp_path


def _make_klippbok_dataset(tmp_path: Path, stems: list[str]) -> Path:
    """Create a klippbok-structured dataset."""
    for stem in stems:
        _touch(tmp_path / "training" / "targets" / f"{stem}.mp4")
        _touch(tmp_path / "training" / "signals" / "captions" / f"{stem}.txt", b"Caption")
        _touch(tmp_path / "training" / "signals" / "references" / f"{stem}.png")
    return tmp_path


def _default_config(
    caption_required: bool = True,
    reference_required: bool = False,
    path: str = ".",
) -> KlippbokDataConfig:
    """Create a config with specified required flags."""
    return KlippbokDataConfig(
        datasets=[{"path": path}],
        controls=ControlsConfig(
            text=TextControlConfig(required=caption_required),
            images=ImagesControlConfig(
                reference=ReferenceImageConfig(required=reference_required),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------

class TestDetectStructure:
    def test_flat_empty_dir(self, tmp_path: Path):
        assert detect_structure(tmp_path) == StructureType.FLAT

    def test_flat_with_files(self, tmp_path: Path):
        _touch(tmp_path / "clip.mp4")
        assert detect_structure(tmp_path) == StructureType.FLAT

    def test_klippbok_with_training_targets(self, tmp_path: Path):
        (tmp_path / "training" / "targets").mkdir(parents=True)
        assert detect_structure(tmp_path) == StructureType.DIMLJUS

    def test_klippbok_needs_both_dirs(self, tmp_path: Path):
        """Just training/ alone doesn't make it klippbok — needs training/targets/."""
        (tmp_path / "training").mkdir()
        assert detect_structure(tmp_path) == StructureType.FLAT


# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------

class TestDiscoverFiles:
    def test_flat_classify_video(self, tmp_path: Path):
        _touch(tmp_path / "clip.mp4")
        files = discover_files(tmp_path, StructureType.FLAT)
        assert len(files["targets"]) == 1
        assert files["targets"][0].name == "clip.mp4"

    def test_flat_classify_caption(self, tmp_path: Path):
        _touch(tmp_path / "clip.txt")
        files = discover_files(tmp_path, StructureType.FLAT)
        assert len(files["captions"]) == 1

    def test_flat_classify_image(self, tmp_path: Path):
        _touch(tmp_path / "clip.png")
        files = discover_files(tmp_path, StructureType.FLAT)
        assert len(files["references"]) == 1

    def test_flat_classify_unknown(self, tmp_path: Path):
        _touch(tmp_path / "notes.md")
        files = discover_files(tmp_path, StructureType.FLAT)
        assert len(files["other"]) == 1

    def test_flat_all_video_extensions(self, tmp_path: Path):
        """Test a subset of recognized video extensions."""
        for ext in [".mp4", ".mov", ".mkv", ".avi"]:
            _touch(tmp_path / f"clip{ext}")
        files = discover_files(tmp_path, StructureType.FLAT)
        assert len(files["targets"]) == 4

    def test_flat_all_image_extensions(self, tmp_path: Path):
        """Test a subset of recognized image extensions."""
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            _touch(tmp_path / f"ref{ext}")
        files = discover_files(tmp_path, StructureType.FLAT)
        assert len(files["references"]) == 4

    def test_flat_case_insensitive_extension(self, tmp_path: Path):
        _touch(tmp_path / "clip.MP4")
        files = discover_files(tmp_path, StructureType.FLAT)
        assert len(files["targets"]) == 1

    def test_flat_skips_directories(self, tmp_path: Path):
        (tmp_path / "subdir").mkdir()
        _touch(tmp_path / "clip.mp4")
        files = discover_files(tmp_path, StructureType.FLAT)
        assert len(files["targets"]) == 1
        assert len(files["other"]) == 0

    def test_flat_sorted_output(self, tmp_path: Path):
        _touch(tmp_path / "c.mp4")
        _touch(tmp_path / "a.mp4")
        _touch(tmp_path / "b.mp4")
        files = discover_files(tmp_path, StructureType.FLAT)
        names = [f.name for f in files["targets"]]
        assert names == ["a.mp4", "b.mp4", "c.mp4"]

    def test_klippbok_targets(self, tmp_path: Path):
        _make_klippbok_dataset(tmp_path, ["clip_001"])
        files = discover_files(tmp_path, StructureType.DIMLJUS)
        assert len(files["targets"]) == 1
        assert files["targets"][0].name == "clip_001.mp4"

    def test_klippbok_captions(self, tmp_path: Path):
        _make_klippbok_dataset(tmp_path, ["clip_001"])
        files = discover_files(tmp_path, StructureType.DIMLJUS)
        assert len(files["captions"]) == 1
        assert files["captions"][0].name == "clip_001.txt"

    def test_klippbok_references(self, tmp_path: Path):
        _make_klippbok_dataset(tmp_path, ["clip_001"])
        files = discover_files(tmp_path, StructureType.DIMLJUS)
        assert len(files["references"]) == 1

    def test_klippbok_other_in_targets_dir(self, tmp_path: Path):
        """Non-video files in targets/ are classified as other."""
        _make_klippbok_dataset(tmp_path, ["clip_001"])
        _touch(tmp_path / "training" / "targets" / "notes.md")
        files = discover_files(tmp_path, StructureType.DIMLJUS)
        assert len(files["other"]) == 1

    def test_auto_detect_structure(self, tmp_path: Path):
        """discover_files auto-detects structure when not provided."""
        _make_flat_dataset(tmp_path, ["clip"])
        files = discover_files(tmp_path)
        assert len(files["targets"]) == 1

    def test_empty_directory(self, tmp_path: Path):
        files = discover_files(tmp_path, StructureType.FLAT)
        assert all(len(v) == 0 for v in files.values())


# ---------------------------------------------------------------------------
# Stem pairing
# ---------------------------------------------------------------------------

class TestPairSamples:
    def test_perfect_pairing(self, tmp_path: Path):
        targets = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
        captions = [tmp_path / "a.txt", tmp_path / "b.txt"]
        references = [tmp_path / "a.png", tmp_path / "b.png"]

        samples, orphaned = pair_samples(targets, captions, references)
        assert len(samples) == 2
        assert len(orphaned) == 0
        assert samples[0].stem == "a"
        assert samples[0].has_caption is True
        assert samples[0].has_reference is True

    def test_missing_caption_required(self, tmp_path: Path):
        targets = [tmp_path / "a.mp4"]
        samples, _ = pair_samples(targets, [], [], caption_required=True)
        assert len(samples) == 1
        assert not samples[0].is_valid
        assert any(i.code == IssueCode.CAPTION_MISSING for i in samples[0].issues)

    def test_missing_caption_not_required(self, tmp_path: Path):
        targets = [tmp_path / "a.mp4"]
        samples, _ = pair_samples(targets, [], [], caption_required=False)
        assert len(samples) == 1
        assert samples[0].is_valid  # no error, just missing

    def test_missing_reference_required(self, tmp_path: Path):
        targets = [tmp_path / "a.mp4"]
        captions = [tmp_path / "a.txt"]
        samples, _ = pair_samples(
            targets, captions, [], reference_required=True,
        )
        assert any(i.code == IssueCode.REFERENCE_MISSING for i in samples[0].issues)

    def test_missing_reference_not_required(self, tmp_path: Path):
        """Missing references are silently skipped when not required."""
        targets = [tmp_path / "a.mp4"]
        captions = [tmp_path / "a.txt"]
        samples, _ = pair_samples(
            targets, captions, [], reference_required=False,
        )
        assert samples[0].is_valid
        assert samples[0].reference is None

    def test_orphaned_caption(self, tmp_path: Path):
        targets = [tmp_path / "a.mp4"]
        captions = [tmp_path / "a.txt", tmp_path / "orphan.txt"]
        samples, orphaned = pair_samples(targets, captions, [])
        assert len(samples) == 1
        assert len(orphaned) == 1
        assert orphaned[0].stem == "orphan"

    def test_orphaned_reference(self, tmp_path: Path):
        targets = [tmp_path / "a.mp4"]
        references = [tmp_path / "a.png", tmp_path / "orphan.png"]
        samples, orphaned = pair_samples(
            targets, [], references, caption_required=False,
        )
        assert len(orphaned) == 1

    def test_empty_targets(self, tmp_path: Path):
        samples, orphaned = pair_samples([], [], [])
        assert len(samples) == 0
        assert len(orphaned) == 0

    def test_stem_matching_case_sensitive(self, tmp_path: Path):
        """Stems are matched exactly (case-sensitive on case-sensitive fs)."""
        targets = [tmp_path / "Clip.mp4"]
        captions = [tmp_path / "Clip.txt"]
        samples, _ = pair_samples(targets, captions, [])
        assert samples[0].has_caption is True


# ---------------------------------------------------------------------------
# Magic byte validation
# ---------------------------------------------------------------------------

class TestValidateFileType:
    def test_returns_none_without_filetype(self, tmp_path: Path, monkeypatch):
        """When filetype is not installed, returns None (skip)."""
        import klippbok.dataset.discover as mod
        monkeypatch.setattr(mod, "_check_magic_bytes", lambda p: None)
        _touch(tmp_path / "clip.mp4")
        result = validate_file_type(tmp_path / "clip.mp4", "video")
        assert result is None

    def test_valid_magic_bytes(self, tmp_path: Path, monkeypatch):
        """When magic bytes match, no issue returned."""
        import klippbok.dataset.discover as mod
        monkeypatch.setattr(mod, "_check_magic_bytes", lambda p: "video/mp4")
        _touch(tmp_path / "clip.mp4")
        result = validate_file_type(tmp_path / "clip.mp4", "video")
        assert result is None

    def test_invalid_magic_bytes(self, tmp_path: Path, monkeypatch):
        """When magic bytes don't match, returns an issue."""
        import klippbok.dataset.discover as mod
        monkeypatch.setattr(mod, "_check_magic_bytes", lambda p: "text/plain")
        _touch(tmp_path / "clip.mp4")
        result = validate_file_type(tmp_path / "clip.mp4", "video")
        assert result is not None
        assert result.code == IssueCode.FILE_TYPE_INVALID
        assert result.severity == Severity.ERROR


# ---------------------------------------------------------------------------
# High-level discovery
# ---------------------------------------------------------------------------

class TestDiscoverDataset:
    def test_flat_dataset(self, tmp_path: Path):
        _make_flat_dataset(tmp_path, ["a", "b", "c"])
        config = _default_config()
        result = discover_dataset(tmp_path, config)
        assert result.structure == StructureType.FLAT
        assert result.total_samples == 3
        assert result.is_valid is True

    def test_klippbok_dataset(self, tmp_path: Path):
        _make_klippbok_dataset(tmp_path, ["clip_001", "clip_002"])
        config = _default_config()
        result = discover_dataset(tmp_path, config)
        assert result.structure == StructureType.DIMLJUS
        assert result.total_samples == 2

    def test_empty_dataset_error(self, tmp_path: Path):
        config = _default_config()
        result = discover_dataset(tmp_path, config)
        assert not result.is_valid
        assert any(i.code == IssueCode.DATASET_EMPTY for i in result.dataset_issues)

    def test_missing_directory_raises(self, tmp_path: Path):
        config = _default_config()
        with pytest.raises(DatasetValidationError, match="does not exist"):
            discover_dataset(tmp_path / "nonexistent", config)

    def test_orphaned_files_flagged(self, tmp_path: Path):
        _make_flat_dataset(tmp_path, ["a"])
        _touch(tmp_path / "orphan.txt", b"stray caption")
        config = _default_config()
        result = discover_dataset(tmp_path, config)
        assert len(result.orphaned_files) == 1
        assert any(i.code == IssueCode.ORPHANED_FILE for i in result.dataset_issues)

    def test_reference_required_propagated(self, tmp_path: Path):
        """Reference required flag from config reaches pairing."""
        _touch(tmp_path / "clip.mp4")
        _touch(tmp_path / "clip.txt", b"caption")
        # No .png reference
        config = _default_config(reference_required=True)
        result = discover_dataset(tmp_path, config)
        assert any(
            i.code == IssueCode.REFERENCE_MISSING
            for s in result.samples for i in s.issues
        )


class TestDiscoverAllDatasets:
    def test_single_source(self, tmp_path: Path):
        ds = tmp_path / "ds1"
        _make_flat_dataset(ds, ["a"])
        config = _default_config(path=str(ds))
        results = discover_all_datasets(config, config_dir=tmp_path)
        assert len(results) == 1
        assert results[0].total_samples == 1

    def test_relative_path_resolution(self, tmp_path: Path):
        ds = tmp_path / "ds1"
        _make_flat_dataset(ds, ["a"])
        config = _default_config(path="ds1")
        results = discover_all_datasets(config, config_dir=tmp_path)
        assert len(results) == 1
        assert results[0].total_samples == 1

    def test_multiple_sources(self, tmp_path: Path):
        ds1 = tmp_path / "ds1"
        ds2 = tmp_path / "ds2"
        _make_flat_dataset(ds1, ["a"])
        _make_flat_dataset(ds2, ["b", "c"])
        config = KlippbokDataConfig(
            datasets=[{"path": str(ds1)}, {"path": str(ds2)}],
        )
        results = discover_all_datasets(config, config_dir=tmp_path)
        assert len(results) == 2
        assert results[0].total_samples == 1
        assert results[1].total_samples == 2
