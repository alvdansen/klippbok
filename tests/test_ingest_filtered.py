"""Tests for filtered ingest via --triage manifest.

Tests _load_triage_manifest() and the CLI integration of --triage on
the ingest command. Uses mocks for ffmpeg/scene detection to avoid
external dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from klippbok.video.__main__ import (
    _load_triage_manifest,
    build_parser,
)


# ---------------------------------------------------------------------------
# _load_triage_manifest tests
# ---------------------------------------------------------------------------


class TestLoadTriageManifest:
    """Tests for _load_triage_manifest() — returns (start, end) segments."""

    def test_basic_load(self, tmp_path: Path) -> None:
        """Loads a valid manifest and returns correct segments."""
        manifest = {
            "triage_mode": "scene",
            "videos": [
                {
                    "file": "v1.mp4",
                    "path": "/videos/v1.mp4",
                    "scenes": [
                        {"scene_index": 0, "start_time": 0.0, "end_time": 5.2, "include": True},
                        {"scene_index": 1, "start_time": 5.2, "end_time": 10.0, "include": False},
                        {"scene_index": 2, "start_time": 10.0, "end_time": 18.0, "include": True},
                    ],
                }
            ],
        }
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")

        result = _load_triage_manifest(p)
        assert "/videos/v1.mp4" in result
        segments = result["/videos/v1.mp4"]
        assert len(segments) == 2
        assert segments[0] == (0.0, 5.2)
        assert segments[1] == (10.0, 18.0)

    def test_no_included_scenes(self, tmp_path: Path) -> None:
        """Video with no included scenes is excluded from result."""
        manifest = {
            "triage_mode": "scene",
            "videos": [
                {
                    "file": "v1.mp4",
                    "path": "/videos/v1.mp4",
                    "scenes": [
                        {"scene_index": 0, "start_time": 0.0, "end_time": 5.0, "include": False},
                    ],
                }
            ],
        }
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")

        result = _load_triage_manifest(p)
        assert "/videos/v1.mp4" not in result

    def test_missing_include_defaults_false(self, tmp_path: Path) -> None:
        """Scenes without 'include' field default to False."""
        manifest = {
            "triage_mode": "scene",
            "videos": [
                {
                    "file": "v1.mp4",
                    "path": "/videos/v1.mp4",
                    "scenes": [
                        {"scene_index": 0, "start_time": 0.0, "end_time": 5.0},
                    ],
                }
            ],
        }
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")

        result = _load_triage_manifest(p)
        assert "/videos/v1.mp4" not in result

    def test_not_found(self) -> None:
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            _load_triage_manifest(Path("/nonexistent.json"))

    def test_wrong_mode(self, tmp_path: Path) -> None:
        """Raises ValueError for non-scene manifest."""
        p = tmp_path / "manifest.json"
        p.write_text('{"triage_mode": "clip"}', encoding="utf-8")
        with pytest.raises(ValueError, match="triage_mode"):
            _load_triage_manifest(p)

    def test_multiple_videos(self, tmp_path: Path) -> None:
        """Loads segments from multiple videos."""
        manifest = {
            "triage_mode": "scene",
            "videos": [
                {
                    "file": "a.mp4",
                    "path": "/videos/a.mp4",
                    "scenes": [
                        {"scene_index": 0, "start_time": 0.0, "end_time": 5.0, "include": True},
                    ],
                },
                {
                    "file": "b.mp4",
                    "path": "/videos/b.mp4",
                    "scenes": [
                        {"scene_index": 0, "start_time": 0.0, "end_time": 3.0, "include": False},
                        {"scene_index": 1, "start_time": 3.0, "end_time": 8.0, "include": True},
                    ],
                },
            ],
        }
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")

        result = _load_triage_manifest(p)
        assert result["/videos/a.mp4"] == [(0.0, 5.0)]
        assert result["/videos/b.mp4"] == [(3.0, 8.0)]

    def test_preserves_segment_order(self, tmp_path: Path) -> None:
        """Segments are returned in manifest order."""
        manifest = {
            "triage_mode": "scene",
            "videos": [
                {
                    "file": "v.mp4",
                    "path": "/videos/v.mp4",
                    "scenes": [
                        {"scene_index": 0, "start_time": 0.0, "end_time": 5.0, "include": True},
                        {"scene_index": 1, "start_time": 5.0, "end_time": 10.0, "include": False},
                        {"scene_index": 2, "start_time": 10.0, "end_time": 15.0, "include": True},
                        {"scene_index": 3, "start_time": 15.0, "end_time": 20.0, "include": True},
                    ],
                }
            ],
        }
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")

        result = _load_triage_manifest(p)
        segments = result["/videos/v.mp4"]
        assert len(segments) == 3
        assert segments[0] == (0.0, 5.0)
        assert segments[1] == (10.0, 15.0)
        assert segments[2] == (15.0, 20.0)


# ---------------------------------------------------------------------------
# CLI flag tests
# ---------------------------------------------------------------------------


class TestIngestTriageFlag:
    """Tests for --triage flag on the ingest subcommand."""

    def test_triage_flag_exists(self) -> None:
        """--triage is accepted by the ingest parser."""
        parser = build_parser()
        args = parser.parse_args([
            "ingest", "video.mp4", "-o", "output/",
            "--triage", "manifest.json",
        ])
        assert args.triage == "manifest.json"

    def test_triage_default_none(self) -> None:
        """--triage defaults to None."""
        parser = build_parser()
        args = parser.parse_args(["ingest", "video.mp4", "-o", "output/"])
        assert args.triage is None

    def test_triage_with_directory(self) -> None:
        """--triage works with directory input."""
        parser = build_parser()
        args = parser.parse_args([
            "ingest", "/videos/", "-o", "output/",
            "--triage", "scene_triage_manifest.json",
        ])
        assert args.triage == "scene_triage_manifest.json"
        assert args.video == "/videos/"

    def test_triage_combined_with_max_frames(self) -> None:
        """--triage and --max-frames can be used together."""
        parser = build_parser()
        args = parser.parse_args([
            "ingest", "video.mp4", "-o", "output/",
            "--triage", "manifest.json",
            "--max-frames", "49",
        ])
        assert args.triage == "manifest.json"
        assert args.max_frames == 49


class TestTriageSceneFlags:
    """Tests for new triage scene-related CLI flags."""

    def test_frames_per_scene_default(self) -> None:
        """--frames-per-scene defaults to 2."""
        parser = build_parser()
        args = parser.parse_args(["triage", ".", "-s", "concepts/"])
        assert args.frames_per_scene == 2

    def test_frames_per_scene_custom(self) -> None:
        """--frames-per-scene accepts custom value."""
        parser = build_parser()
        args = parser.parse_args([
            "triage", ".", "-s", "concepts/",
            "--frames-per-scene", "4",
        ])
        assert args.frames_per_scene == 4

    def test_scene_threshold_default(self) -> None:
        """--scene-threshold defaults to 27.0."""
        parser = build_parser()
        args = parser.parse_args(["triage", ".", "-s", "concepts/"])
        assert args.scene_threshold == 27.0

    def test_scene_threshold_custom(self) -> None:
        """--scene-threshold accepts custom value."""
        parser = build_parser()
        args = parser.parse_args([
            "triage", ".", "-s", "concepts/",
            "--scene-threshold", "20.0",
        ])
        assert args.scene_threshold == 20.0

    def test_all_triage_flags_together(self) -> None:
        """All triage flags work together."""
        parser = build_parser()
        args = parser.parse_args([
            "triage", "/raw_data/", "-s", "concepts/",
            "--threshold", "0.65",
            "--frames", "3",
            "--frames-per-scene", "4",
            "--scene-threshold", "20.0",
            "--clip-model", "openai/clip-vit-large-patch14",
            "-o", "triage_out.json",
        ])
        assert args.threshold == 0.65
        assert args.frames == 3
        assert args.frames_per_scene == 4
        assert args.scene_threshold == 20.0
        assert args.clip_model == "openai/clip-vit-large-patch14"
        assert args.output == "triage_out.json"
