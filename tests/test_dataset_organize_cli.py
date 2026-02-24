"""Tests for dataset organize CLI integration.

Tests the argparse setup, cmd_organize handler, validate hint,
and end-to-end CLI scenarios.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from klippbok.dataset.__main__ import build_parser, cmd_organize, cmd_validate


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
# Parser tests for organize subcommand
# ---------------------------------------------------------------------------

class TestOrganizeParser:
    def test_organize_command(self):
        parser = build_parser()
        args = parser.parse_args(["organize", "/data", "-o", "/out"])
        assert args.command == "organize"
        assert args.path == "/data"
        assert args.output == "/out"

    def test_layout_flag(self):
        parser = build_parser()
        args = parser.parse_args(["organize", "/data", "-o", "/out", "-l", "klippbok"])
        assert args.layout == "klippbok"

    def test_layout_default(self):
        parser = build_parser()
        args = parser.parse_args(["organize", "/data", "-o", "/out"])
        assert args.layout == "flat"

    def test_trainer_single(self):
        parser = build_parser()
        args = parser.parse_args(["organize", "/data", "-o", "/out", "-t", "musubi"])
        assert args.trainers == ["musubi"]

    def test_trainer_multiple(self):
        parser = build_parser()
        args = parser.parse_args([
            "organize", "/data", "-o", "/out", "-t", "musubi", "-t", "aitoolkit",
        ])
        assert args.trainers == ["musubi", "aitoolkit"]

    def test_move_flag(self):
        parser = build_parser()
        args = parser.parse_args(["organize", "/data", "-o", "/out", "--move"])
        assert args.move is True

    def test_dry_run_flag(self):
        parser = build_parser()
        args = parser.parse_args(["organize", "/data", "-o", "/out", "--dry-run"])
        assert args.dry_run is True

    def test_strict_flag(self):
        parser = build_parser()
        args = parser.parse_args(["organize", "/data", "-o", "/out", "--strict"])
        assert args.strict is True

    def test_config_flag(self):
        parser = build_parser()
        args = parser.parse_args(["organize", "/data", "-o", "/out", "-c", "my.yaml"])
        assert args.config == "my.yaml"

    def test_manifest_flag(self):
        parser = build_parser()
        args = parser.parse_args(["organize", "/data", "-o", "/out", "--manifest"])
        assert args.manifest is True

    def test_concepts_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            "organize", "/data", "-o", "/out", "--concepts", "holly,cat",
        ])
        assert args.concepts == "holly,cat"

    def test_concepts_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["organize", "/data", "-o", "/out"])
        assert args.concepts is None

    def test_all_flags_combined(self):
        parser = build_parser()
        args = parser.parse_args([
            "organize", "/data", "-o", "/out",
            "-l", "klippbok", "-t", "musubi",
            "--concepts", "holly",
            "--move", "--dry-run", "--strict", "--manifest",
        ])
        assert args.layout == "klippbok"
        assert args.trainers == ["musubi"]
        assert args.concepts == "holly"
        assert args.move is True
        assert args.dry_run is True
        assert args.strict is True
        assert args.manifest is True


# ---------------------------------------------------------------------------
# End-to-end CLI organize tests
# ---------------------------------------------------------------------------

class TestCmdOrganize:
    def test_flat_organize_returns_0(self, tmp_path: Path):
        src = _make_flat_dataset(tmp_path / "src", ["a", "b"])
        out = tmp_path / "out"

        parser = build_parser()
        args = parser.parse_args(["organize", str(src), "-o", str(out)])
        result = cmd_organize(args)
        assert result == 0
        assert (out / "a.mp4").exists()
        assert (out / "b.mp4").exists()

    def test_klippbok_organize_returns_0(self, tmp_path: Path):
        src = _make_flat_dataset(tmp_path / "src", ["a"])
        out = tmp_path / "out"

        parser = build_parser()
        args = parser.parse_args(["organize", str(src), "-o", str(out), "-l", "klippbok"])
        result = cmd_organize(args)
        assert result == 0
        assert (out / "training" / "targets" / "a.mp4").exists()

    def test_musubi_config_created(self, tmp_path: Path):
        src = _make_flat_dataset(tmp_path / "src", ["a"])
        out = tmp_path / "out"

        parser = build_parser()
        args = parser.parse_args([
            "organize", str(src), "-o", str(out), "-t", "musubi",
        ])
        result = cmd_organize(args)
        assert result == 0
        assert (out / "musubi_dataset.toml").exists()

    def test_dry_run_creates_nothing(self, tmp_path: Path):
        src = _make_flat_dataset(tmp_path / "src", ["a"])
        out = tmp_path / "out"

        parser = build_parser()
        args = parser.parse_args([
            "organize", str(src), "-o", str(out), "--dry-run",
        ])
        result = cmd_organize(args)
        assert result == 0
        assert not out.exists()

    def test_nonexistent_source_returns_1(self, tmp_path: Path):
        parser = build_parser()
        args = parser.parse_args([
            "organize", str(tmp_path / "noexist"), "-o", str(tmp_path / "out"),
        ])
        result = cmd_organize(args)
        assert result == 1

    def test_empty_dataset_returns_1(self, tmp_path: Path):
        """Empty source (no videos) returns 1."""
        src = tmp_path / "src"
        src.mkdir()

        parser = build_parser()
        args = parser.parse_args([
            "organize", str(src), "-o", str(tmp_path / "out"),
        ])
        result = cmd_organize(args)
        assert result == 1


# ---------------------------------------------------------------------------
# Concepts filtering tests
# ---------------------------------------------------------------------------

class TestConcepts:
    def _make_triage_output(self, tmp_path: Path) -> Path:
        """Create a mock triage output with multiple concept folders."""
        sorted_dir = tmp_path / "sorted"
        _make_flat_dataset(sorted_dir / "holly", ["clip_a", "clip_b", "clip_c"])
        _make_flat_dataset(sorted_dir / "cat", ["clip_d", "clip_e"])
        _make_flat_dataset(sorted_dir / "tiffanys", ["clip_f"])
        (sorted_dir / "text_overlay").mkdir(parents=True)
        _touch(sorted_dir / "text_overlay" / "bad.mp4")
        return sorted_dir

    def test_single_concept(self, tmp_path: Path):
        """--concepts holly organizes only holly clips."""
        sorted_dir = self._make_triage_output(tmp_path)
        out = tmp_path / "out"

        parser = build_parser()
        args = parser.parse_args([
            "organize", str(sorted_dir), "-o", str(out),
            "--concepts", "holly",
        ])
        result = cmd_organize(args)
        assert result == 0
        assert (out / "clip_a.mp4").exists()
        assert (out / "clip_b.mp4").exists()
        assert (out / "clip_c.mp4").exists()
        # cat and tiffanys should NOT be in output
        assert not (out / "clip_d.mp4").exists()
        assert not (out / "clip_f.mp4").exists()

    def test_multiple_concepts(self, tmp_path: Path):
        """--concepts holly,cat organizes both."""
        sorted_dir = self._make_triage_output(tmp_path)
        out = tmp_path / "out"

        parser = build_parser()
        args = parser.parse_args([
            "organize", str(sorted_dir), "-o", str(out),
            "--concepts", "holly,cat",
        ])
        result = cmd_organize(args)
        assert result == 0
        # holly clips
        assert (out / "clip_a.mp4").exists()
        assert (out / "clip_b.mp4").exists()
        assert (out / "clip_c.mp4").exists()
        # cat clips
        assert (out / "clip_d.mp4").exists()
        assert (out / "clip_e.mp4").exists()
        # tiffanys NOT included
        assert not (out / "clip_f.mp4").exists()

    def test_concepts_with_trainer(self, tmp_path: Path):
        """--concepts + --trainer generates config for selected concepts only."""
        sorted_dir = self._make_triage_output(tmp_path)
        out = tmp_path / "out"

        parser = build_parser()
        args = parser.parse_args([
            "organize", str(sorted_dir), "-o", str(out),
            "--concepts", "holly", "-t", "musubi",
        ])
        result = cmd_organize(args)
        assert result == 0
        assert (out / "musubi_dataset.toml").exists()

    def test_unknown_concept_returns_1(self, tmp_path: Path):
        """Unknown concept name returns error with available list."""
        sorted_dir = self._make_triage_output(tmp_path)
        out = tmp_path / "out"

        parser = build_parser()
        args = parser.parse_args([
            "organize", str(sorted_dir), "-o", str(out),
            "--concepts", "nonexistent",
        ])
        result = cmd_organize(args)
        assert result == 1

    def test_unknown_concept_shows_available(self, tmp_path: Path, capsys):
        """Error message lists available concept folders."""
        sorted_dir = self._make_triage_output(tmp_path)
        out = tmp_path / "out"

        parser = build_parser()
        args = parser.parse_args([
            "organize", str(sorted_dir), "-o", str(out),
            "--concepts", "nonexistent",
        ])
        cmd_organize(args)

        captured = capsys.readouterr()
        assert "not found" in captured.err
        assert "holly" in captured.err
        assert "cat" in captured.err

    def test_concepts_dry_run(self, tmp_path: Path):
        """Dry-run with concepts shows what would happen without touching files."""
        sorted_dir = self._make_triage_output(tmp_path)
        out = tmp_path / "out"

        parser = build_parser()
        args = parser.parse_args([
            "organize", str(sorted_dir), "-o", str(out),
            "--concepts", "holly", "--dry-run",
        ])
        result = cmd_organize(args)
        assert result == 0
        assert not out.exists()

    def test_concepts_with_spaces(self, tmp_path: Path):
        """Concept names with spaces between commas are trimmed."""
        sorted_dir = self._make_triage_output(tmp_path)
        out = tmp_path / "out"

        parser = build_parser()
        args = parser.parse_args([
            "organize", str(sorted_dir), "-o", str(out),
            "--concepts", "holly , cat",
        ])
        result = cmd_organize(args)
        assert result == 0
        assert (out / "clip_a.mp4").exists()
        assert (out / "clip_d.mp4").exists()


# ---------------------------------------------------------------------------
# Validate hint tests
# ---------------------------------------------------------------------------

class TestValidateHint:
    def test_hint_printed_on_valid(self, tmp_path: Path, capsys):
        """Validate should print organize hint when samples are valid."""
        _make_flat_dataset(tmp_path, ["a"])
        parser = build_parser()
        args = parser.parse_args(["validate", str(tmp_path)])
        cmd_validate(args)

        captured = capsys.readouterr()
        assert "Next step: organize" in captured.out
        assert "python -m klippbok.dataset organize" in captured.out

    def test_no_hint_for_json(self, tmp_path: Path, capsys):
        """JSON output should not include the organize hint."""
        _make_flat_dataset(tmp_path, ["a"])
        parser = build_parser()
        args = parser.parse_args(["validate", str(tmp_path), "--json"])
        cmd_validate(args)

        captured = capsys.readouterr()
        assert "Next step" not in captured.out

    def test_no_hint_for_empty(self, tmp_path: Path, capsys):
        """Empty dataset (no valid samples) should not show organize hint."""
        parser = build_parser()
        args = parser.parse_args(["validate", str(tmp_path)])
        cmd_validate(args)

        captured = capsys.readouterr()
        assert "Next step" not in captured.out


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------

class TestOrganizeReport:
    def test_plaintext_format(self):
        from klippbok.dataset.models import OrganizeLayout, OrganizedSample, OrganizeResult
        from klippbok.dataset.report import format_organize_plaintext

        result = OrganizeResult(
            output_dir=Path("/out"),
            layout=OrganizeLayout.FLAT,
            organized=[
                OrganizedSample(
                    stem="a",
                    target_dest=Path("/out/a.mp4"),
                    caption_dest=Path("/out/a.txt"),
                ),
                OrganizedSample(
                    stem="b",
                    target_dest=Path("/out/b.mp4"),
                ),
            ],
            skipped=[
                OrganizedSample(stem="c", skipped=True, skip_reason="missing caption"),
            ],
        )
        text = format_organize_plaintext(result)
        assert "Organizing dataset" in text
        assert "flat" in text
        assert "2 organized" in text
        assert "1 skipped" in text
        assert "[SKIP] c: missing caption" in text

    def test_plaintext_dry_run(self):
        from klippbok.dataset.models import OrganizeLayout, OrganizedSample, OrganizeResult
        from klippbok.dataset.report import format_organize_plaintext

        result = OrganizeResult(
            output_dir=Path("/out"),
            layout=OrganizeLayout.FLAT,
            organized=[OrganizedSample(stem="a", target_dest=Path("/out/a.mp4"))],
            dry_run=True,
        )
        text = format_organize_plaintext(result)
        assert "dry-run" in text

    def test_plaintext_trainer_configs(self):
        from klippbok.dataset.models import OrganizeLayout, OrganizedSample, OrganizeResult
        from klippbok.dataset.report import format_organize_plaintext

        result = OrganizeResult(
            output_dir=Path("/out"),
            layout=OrganizeLayout.FLAT,
            organized=[OrganizedSample(stem="a", target_dest=Path("/out/a.mp4"))],
            trainer_configs=[Path("/out/musubi_dataset.toml")],
        )
        text = format_organize_plaintext(result)
        assert "Trainer config:" in text
        assert "musubi_dataset.toml" in text

    def test_ascii_safe(self):
        from klippbok.dataset.models import OrganizeLayout, OrganizedSample, OrganizeResult
        from klippbok.dataset.report import format_organize_plaintext

        result = OrganizeResult(
            output_dir=Path("/out"),
            layout=OrganizeLayout.FLAT,
            organized=[OrganizedSample(stem="a", target_dest=Path("/out/a.mp4"))],
            skipped=[OrganizedSample(stem="b", skipped=True, skip_reason="error")],
        )
        text = format_organize_plaintext(result)
        text.encode("ascii")  # raises if non-ASCII
