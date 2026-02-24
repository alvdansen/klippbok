"""Tests for klippbok.dataset.organize — core organize engine.

Tests cover: flat layout, klippbok layout, copy vs move, dry-run,
error filtering, strict mode, name collisions, idempotent re-runs,
and edge cases.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from klippbok.config.data_schema import (
    ControlsConfig,
    KlippbokDataConfig,
    ImagesControlConfig,
    ReferenceImageConfig,
    TextControlConfig,
)
from klippbok.dataset.errors import OrganizeError
from klippbok.dataset.models import (
    OrganizeLayout,
    OrganizedSample,
    OrganizeResult,
)
from klippbok.dataset.organize import (
    _resolve_collision,
    _transfer_file,
    organize_dataset,
)


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


def _make_flat_dataset(tmp_path: Path, stems: list[str], with_refs: bool = True) -> Path:
    """Create a flat dataset with video, caption, and optionally reference per stem."""
    for stem in stems:
        _touch(tmp_path / f"{stem}.mp4")
        _touch(tmp_path / f"{stem}.txt", f"Caption for {stem}.".encode())
        if with_refs:
            _make_textured_image(tmp_path / f"{stem}.png")
    return tmp_path


def _default_config(path: str = ".") -> KlippbokDataConfig:
    return KlippbokDataConfig(
        datasets=[{"path": path}],
        controls=ControlsConfig(
            text=TextControlConfig(required=True),
            images=ImagesControlConfig(
                reference=ReferenceImageConfig(required=False),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestOrganizeModels:
    def test_organize_layout_values(self):
        assert OrganizeLayout.FLAT.value == "flat"
        assert OrganizeLayout.DIMLJUS.value == "klippbok"

    def test_organized_sample_defaults(self):
        s = OrganizedSample(stem="clip_001")
        assert s.stem == "clip_001"
        assert s.target_dest is None
        assert s.skipped is False
        assert s.skip_reason == ""

    def test_organized_sample_skipped(self):
        s = OrganizedSample(stem="bad", skipped=True, skip_reason="missing caption")
        assert s.skipped is True
        assert s.skip_reason == "missing caption"

    def test_organize_result_counts(self):
        result = OrganizeResult(
            output_dir=Path("/out"),
            layout=OrganizeLayout.FLAT,
            organized=[
                OrganizedSample(stem="a"),
                OrganizedSample(stem="b"),
            ],
            skipped=[
                OrganizedSample(stem="c", skipped=True, skip_reason="error"),
            ],
        )
        assert result.organized_count == 2
        assert result.skipped_count == 1
        assert result.total_count == 3

    def test_organize_result_empty(self):
        result = OrganizeResult(
            output_dir=Path("/out"),
            layout=OrganizeLayout.FLAT,
        )
        assert result.organized_count == 0
        assert result.skipped_count == 0
        assert result.total_count == 0


# ---------------------------------------------------------------------------
# Error tests
# ---------------------------------------------------------------------------

class TestOrganizeErrors:
    def test_organize_error_hierarchy(self):
        from klippbok.dataset.errors import KlippbokDatasetError
        err = OrganizeError("test")
        assert isinstance(err, KlippbokDatasetError)
        assert "Organize failed" in str(err)
        assert err.detail == "test"

    def test_source_not_found(self, tmp_path: Path):
        with pytest.raises(OrganizeError, match="does not exist"):
            organize_dataset(
                source_dir=tmp_path / "nonexistent",
                output_dir=tmp_path / "out",
            )

    def test_zero_valid_samples(self, tmp_path: Path):
        """All samples have errors -> OrganizeError."""
        src = tmp_path / "src"
        src.mkdir()
        # Videos without captions -> CAPTION_MISSING errors
        _touch(src / "a.mp4")
        _touch(src / "b.mp4")

        with pytest.raises(OrganizeError, match="No valid samples"):
            organize_dataset(
                source_dir=src,
                output_dir=tmp_path / "out",
            )


# ---------------------------------------------------------------------------
# Transfer and collision helpers
# ---------------------------------------------------------------------------

class TestTransferFile:
    def test_copy_creates_file(self, tmp_path: Path):
        src = _touch(tmp_path / "src" / "a.txt", b"hello")
        dest = tmp_path / "out" / "a.txt"
        _transfer_file(src, dest, copy=True)
        assert dest.exists()
        assert dest.read_bytes() == b"hello"
        assert src.exists()  # original preserved

    def test_move_removes_source(self, tmp_path: Path):
        src = _touch(tmp_path / "src" / "a.txt", b"hello")
        dest = tmp_path / "out" / "a.txt"
        _transfer_file(src, dest, copy=False)
        assert dest.exists()
        assert not src.exists()

    def test_same_path_skips(self, tmp_path: Path):
        f = _touch(tmp_path / "a.txt", b"hello")
        _transfer_file(f, f, copy=True)
        assert f.exists()  # no error, no removal

    def test_overwrites_existing(self, tmp_path: Path):
        src = _touch(tmp_path / "src" / "a.txt", b"new")
        dest = _touch(tmp_path / "out" / "a.txt", b"old")
        _transfer_file(src, dest, copy=True)
        assert dest.read_bytes() == b"new"


class TestResolveCollision:
    def test_no_collision(self):
        used: set[str] = set()
        result = _resolve_collision(Path("/out/clip.mp4"), used)
        assert result == Path("/out/clip.mp4")
        assert "clip" in used

    def test_collision_appends_suffix(self):
        used: set[str] = {"clip"}
        result = _resolve_collision(Path("/out/clip.mp4"), used)
        assert result.stem == "clip_2"
        assert result.suffix == ".mp4"
        assert "clip_2" in used

    def test_multiple_collisions(self):
        used: set[str] = {"clip", "clip_2"}
        result = _resolve_collision(Path("/out/clip.mp4"), used)
        assert result.stem == "clip_3"


# ---------------------------------------------------------------------------
# Flat layout organize
# ---------------------------------------------------------------------------

class TestFlatOrganize:
    def test_basic_flat(self, tmp_path: Path):
        """Flat organize copies all files stem-matched to output root."""
        src = _make_flat_dataset(tmp_path / "src", ["a", "b"])
        out = tmp_path / "out"

        result = organize_dataset(src, out)

        assert result.layout == OrganizeLayout.FLAT
        assert result.organized_count == 2
        assert result.skipped_count == 0
        assert (out / "a.mp4").exists()
        assert (out / "a.txt").exists()
        assert (out / "a.png").exists()
        assert (out / "b.mp4").exists()

    def test_flat_without_refs(self, tmp_path: Path):
        """Flat organize works when no reference images present."""
        src = _make_flat_dataset(tmp_path / "src", ["a"], with_refs=False)
        out = tmp_path / "out"

        result = organize_dataset(src, out)
        assert result.organized_count == 1
        assert (out / "a.mp4").exists()
        assert (out / "a.txt").exists()
        assert not (out / "a.png").exists()

    def test_flat_preserves_extension(self, tmp_path: Path):
        """File extensions are preserved from source."""
        src = tmp_path / "src"
        _touch(src / "clip.mov")
        _touch(src / "clip.txt", b"Caption")
        out = tmp_path / "out"

        result = organize_dataset(src, out)
        assert result.organized_count == 1
        assert (out / "clip.mov").exists()

    def test_copy_preserves_originals(self, tmp_path: Path):
        """Default copy mode doesn't remove source files."""
        src = _make_flat_dataset(tmp_path / "src", ["a"])
        out = tmp_path / "out"

        organize_dataset(src, out, copy=True)
        assert (src / "a.mp4").exists()
        assert (src / "a.txt").exists()

    def test_move_removes_originals(self, tmp_path: Path):
        """Move mode removes source files."""
        src = _make_flat_dataset(tmp_path / "src", ["a"])
        out = tmp_path / "out"

        organize_dataset(src, out, copy=False)
        assert not (src / "a.mp4").exists()
        assert not (src / "a.txt").exists()
        assert (out / "a.mp4").exists()
        assert (out / "a.txt").exists()

    def test_idempotent_rerun(self, tmp_path: Path):
        """Running organize twice produces the same result."""
        src = _make_flat_dataset(tmp_path / "src", ["a", "b"])
        out = tmp_path / "out"

        r1 = organize_dataset(src, out)
        r2 = organize_dataset(src, out)

        assert r1.organized_count == r2.organized_count
        assert (out / "a.mp4").exists()
        assert (out / "b.mp4").exists()


# ---------------------------------------------------------------------------
# Klippbok layout organize
# ---------------------------------------------------------------------------

class TestKlippbokOrganize:
    def test_basic_klippbok(self, tmp_path: Path):
        """Klippbok layout creates training/targets/ and training/signals/."""
        src = _make_flat_dataset(tmp_path / "src", ["a", "b"])
        out = tmp_path / "out"

        result = organize_dataset(src, out, layout=OrganizeLayout.DIMLJUS)

        assert result.layout == OrganizeLayout.DIMLJUS
        assert (out / "training" / "targets" / "a.mp4").exists()
        assert (out / "training" / "targets" / "b.mp4").exists()
        assert (out / "training" / "signals" / "captions" / "a.txt").exists()
        assert (out / "training" / "signals" / "captions" / "b.txt").exists()
        assert (out / "training" / "signals" / "references" / "a.png").exists()
        assert (out / "training" / "signals" / "references" / "b.png").exists()

    def test_klippbok_without_refs(self, tmp_path: Path):
        """Klippbok layout works without reference images."""
        src = _make_flat_dataset(tmp_path / "src", ["a"], with_refs=False)
        out = tmp_path / "out"

        result = organize_dataset(src, out, layout=OrganizeLayout.DIMLJUS)
        assert (out / "training" / "targets" / "a.mp4").exists()
        assert (out / "training" / "signals" / "captions" / "a.txt").exists()
        # No references dir created if no refs
        assert not (out / "training" / "signals" / "references" / "a.png").exists()


# ---------------------------------------------------------------------------
# Filtering (errors, warnings, strict)
# ---------------------------------------------------------------------------

class TestFiltering:
    def test_errors_skipped(self, tmp_path: Path):
        """Samples with errors are skipped."""
        src = tmp_path / "src"
        # a has both video and caption (valid), b has no caption (error)
        _touch(src / "a.mp4")
        _touch(src / "a.txt", b"Caption for a")
        _touch(src / "b.mp4")
        out = tmp_path / "out"

        result = organize_dataset(src, out)
        assert result.organized_count == 1
        assert result.skipped_count == 1
        assert result.skipped[0].stem == "b"
        assert (out / "a.mp4").exists()
        assert not (out / "b.mp4").exists()

    def test_warnings_included_by_default(self, tmp_path: Path):
        """Samples with warnings (but no errors) are included by default."""
        src = tmp_path / "src"
        _touch(src / "a.mp4")
        _touch(src / "a.txt", b"")  # empty caption -> warning
        out = tmp_path / "out"

        # captions not required -> empty is just a warning, not error
        config = KlippbokDataConfig(
            datasets=[{"path": str(src)}],
            controls=ControlsConfig(
                text=TextControlConfig(required=False),
            ),
        )
        result = organize_dataset(src, out, config=config)
        assert result.organized_count == 1

    def test_strict_excludes_warnings(self, tmp_path: Path):
        """--strict mode excludes samples with warnings."""
        src = tmp_path / "src"
        _touch(src / "a.mp4")
        _touch(src / "a.txt", b"")  # empty caption -> warning
        _touch(src / "b.mp4")
        _touch(src / "b.txt", b"Good caption")
        out = tmp_path / "out"

        config = KlippbokDataConfig(
            datasets=[{"path": str(src)}],
            controls=ControlsConfig(
                text=TextControlConfig(required=False),
            ),
        )
        result = organize_dataset(
            src, out, config=config, include_warnings=False,
        )
        assert result.organized_count == 1
        assert result.skipped_count == 1
        assert result.skipped[0].stem == "a"

    def test_strict_all_excluded_raises(self, tmp_path: Path):
        """If strict mode excludes everything, OrganizeError raised."""
        src = tmp_path / "src"
        _touch(src / "a.mp4")
        _touch(src / "a.txt", b"")  # empty caption -> warning
        out = tmp_path / "out"

        config = KlippbokDataConfig(
            datasets=[{"path": str(src)}],
            controls=ControlsConfig(
                text=TextControlConfig(required=False),
            ),
        )
        with pytest.raises(OrganizeError, match="No valid samples"):
            organize_dataset(
                src, out, config=config, include_warnings=False,
            )


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_no_files_created(self, tmp_path: Path):
        """Dry run doesn't create any files."""
        src = _make_flat_dataset(tmp_path / "src", ["a", "b"])
        out = tmp_path / "out"

        result = organize_dataset(src, out, dry_run=True)
        assert result.dry_run is True
        assert result.organized_count == 2
        assert not out.exists()

    def test_dry_run_with_trainers(self, tmp_path: Path):
        """Dry run with trainer flag doesn't write config files."""
        src = _make_flat_dataset(tmp_path / "src", ["a"])
        out = tmp_path / "out"

        result = organize_dataset(src, out, dry_run=True, trainers=["musubi"])
        assert len(result.trainer_configs) == 1
        assert not out.exists()


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------

class TestOrganizeWithTrainers:
    def test_musubi_config_generated(self, tmp_path: Path):
        """organize with trainers=['musubi'] creates musubi_dataset.toml."""
        src = _make_flat_dataset(tmp_path / "src", ["a", "b"])
        out = tmp_path / "out"

        result = organize_dataset(src, out, trainers=["musubi"])
        assert len(result.trainer_configs) == 1
        assert result.trainer_configs[0].name == "musubi_dataset.toml"
        assert result.trainer_configs[0].exists()

    def test_multiple_trainers(self, tmp_path: Path):
        """Multiple trainer flags generate multiple config files."""
        src = _make_flat_dataset(tmp_path / "src", ["a"])
        out = tmp_path / "out"

        result = organize_dataset(src, out, trainers=["musubi", "aitoolkit"])
        assert len(result.trainer_configs) == 2

    def test_unknown_trainer_raises(self, tmp_path: Path):
        """Unknown trainer name raises OrganizeError."""
        src = _make_flat_dataset(tmp_path / "src", ["a"])
        out = tmp_path / "out"

        with pytest.raises(OrganizeError, match="Unknown trainer"):
            organize_dataset(src, out, trainers=["nonexistent"])


# ---------------------------------------------------------------------------
# Name collisions
# ---------------------------------------------------------------------------

class TestNameCollisions:
    def test_collision_resolved(self, tmp_path: Path):
        """When two sources have the same stem, second gets _2 suffix."""
        src1 = tmp_path / "src1"
        src2 = tmp_path / "src2"
        _touch(src1 / "clip.mp4")
        _touch(src1 / "clip.txt", b"Caption 1")
        _touch(src2 / "clip.mp4")
        _touch(src2 / "clip.txt", b"Caption 2")
        out = tmp_path / "out"

        config = KlippbokDataConfig(
            datasets=[{"path": str(src1)}, {"path": str(src2)}],
        )
        result = organize_dataset(src1, out, config=config)
        # Both should be organized, one with suffix
        assert result.organized_count == 2
        stems = {s.stem for s in result.organized}
        assert "clip" in stems
        assert "clip_2" in stems
