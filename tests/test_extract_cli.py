"""Tests for the `extract` CLI command in klippbok.video.__main__.

Tests parser configuration, argument handling, and end-to-end
extraction via the CLI. Most tests require ffmpeg.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.conftest import requires_ffmpeg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_test_image(path: Path, width: int = 320, height: int = 240) -> None:
    """Write a checkerboard test image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            if ((x // 8) + (y // 8)) % 2 == 0:
                img[y:y+8, x:x+8] = 255
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# Parser tests (no ffmpeg needed)
# ---------------------------------------------------------------------------

class TestExtractParser:
    """Tests for the extract subcommand parser configuration."""

    def test_extract_command_exists(self) -> None:
        """The parser recognizes 'extract' as a command."""
        from klippbok.video.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args(["extract", "some_dir", "--output", "out_dir"])
        assert args.command == "extract"

    def test_default_strategy(self) -> None:
        """Default strategy is first_frame."""
        from klippbok.video.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args(["extract", "some_dir", "--output", "out"])
        assert args.strategy == "first_frame"

    def test_best_frame_strategy(self) -> None:
        """Can specify best_frame strategy."""
        from klippbok.video.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "extract", "some_dir", "--output", "out",
            "--strategy", "best_frame",
        ])
        assert args.strategy == "best_frame"

    def test_samples_argument(self) -> None:
        """Can specify sample count."""
        from klippbok.video.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "extract", "some_dir", "--output", "out",
            "--samples", "20",
        ])
        assert args.samples == 20

    def test_overwrite_flag(self) -> None:
        """Can set overwrite flag."""
        from klippbok.video.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "extract", "some_dir", "--output", "out", "--overwrite",
        ])
        assert args.overwrite is True

    def test_selections_argument(self) -> None:
        """Can specify selections manifest path."""
        from klippbok.video.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "extract", "some_dir", "--output", "out",
            "--selections", "picks.json",
        ])
        assert args.selections == "picks.json"

    def test_template_argument(self) -> None:
        """Can specify template output path."""
        from klippbok.video.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "extract", "some_dir", "--template", "template.json",
        ])
        assert args.template == "template.json"


# ---------------------------------------------------------------------------
# End-to-end tests (require ffmpeg)
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestExtractCLIEndToEnd:
    """End-to-end CLI tests for extract command — requires ffmpeg."""

    def test_first_frame_extraction(self, tiny_video: Path, tmp_path: Path) -> None:
        """Full CLI round-trip: extract first frame from a clip."""
        from klippbok.video.__main__ import build_parser, cmd_extract

        src_dir = tmp_path / "clips"
        src_dir.mkdir()
        shutil.copy2(str(tiny_video), str(src_dir / "clip_001.mp4"))

        out_dir = tmp_path / "refs"

        parser = build_parser()
        args = parser.parse_args([
            "extract", str(src_dir), "--output", str(out_dir),
        ])
        cmd_extract(args)

        assert (out_dir / "clip_001.png").exists()
        assert (out_dir / "reference_images.json").exists()

    def test_best_frame_extraction(self, tiny_video: Path, tmp_path: Path) -> None:
        """Full CLI round-trip: extract best frame from a clip."""
        from klippbok.video.__main__ import build_parser, cmd_extract

        src_dir = tmp_path / "clips"
        src_dir.mkdir()
        shutil.copy2(str(tiny_video), str(src_dir / "clip_001.mp4"))

        out_dir = tmp_path / "refs"

        parser = build_parser()
        args = parser.parse_args([
            "extract", str(src_dir), "--output", str(out_dir),
            "--strategy", "best_frame", "--samples", "3",
        ])
        cmd_extract(args)

        assert (out_dir / "clip_001.png").exists()

    def test_template_generation(self, tiny_video: Path, tmp_path: Path) -> None:
        """CLI template generation mode."""
        from klippbok.video.__main__ import build_parser, cmd_extract

        src_dir = tmp_path / "clips"
        src_dir.mkdir()
        shutil.copy2(str(tiny_video), str(src_dir / "clip_001.mp4"))

        template_path = tmp_path / "selections.json"

        parser = build_parser()
        args = parser.parse_args([
            "extract", str(src_dir), "--template", str(template_path),
        ])
        cmd_extract(args)

        assert template_path.exists()
        data = json.loads(template_path.read_text())
        assert "clip_001.mp4" in data
