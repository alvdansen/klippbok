"""Tests for scene-level triage: models, sampler, orchestrator, manifest.

Tests the duration-adaptive triage feature that lets users triage raw
videos before ingesting. Covers SceneTriage/VideoTriageReport models,
sample_scene_frames(), triage_videos() orchestration, and manifest I/O.

Uses mocks for ffmpeg/ffprobe/CLIP to avoid external dependencies.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from klippbok.triage.models import (
    ClipMatch,
    ConceptReference,
    ConceptType,
    SceneTriage,
    VideoTriageReport,
)


# ---------------------------------------------------------------------------
# SceneTriage model tests
# ---------------------------------------------------------------------------


class TestSceneTriage:
    """Tests for the SceneTriage dataclass."""

    def test_creation(self) -> None:
        """SceneTriage can be created with required fields."""
        st = SceneTriage(
            source_video=Path("/videos/001.mp4"),
            scene_index=0,
            start_time=0.0,
            end_time=5.2,
        )
        assert st.source_video == Path("/videos/001.mp4")
        assert st.scene_index == 0
        assert st.start_time == 0.0
        assert st.end_time == 5.2

    def test_duration(self) -> None:
        """duration property calculates end - start."""
        st = SceneTriage(
            source_video=Path("v.mp4"),
            scene_index=0,
            start_time=10.0,
            end_time=15.5,
        )
        assert abs(st.duration - 5.5) < 0.001

    def test_no_matches(self) -> None:
        """Unmatched scene has correct properties."""
        st = SceneTriage(
            source_video=Path("v.mp4"),
            scene_index=0,
            start_time=0.0,
            end_time=5.0,
        )
        assert st.best_match is None
        assert not st.is_matched
        assert st.matches == []

    def test_with_matches(self) -> None:
        """Matched scene returns best match."""
        ref = ConceptReference("holly", ConceptType.CHARACTER, Path("h.jpg"), "character")
        match = ClipMatch(concept=ref, similarity=0.82, best_frame_index=1)
        st = SceneTriage(
            source_video=Path("v.mp4"),
            scene_index=0,
            start_time=0.0,
            end_time=5.0,
            matches=[match],
        )
        assert st.is_matched
        assert st.best_match == match
        assert st.best_match.similarity == 0.82

    def test_text_overlay_defaults(self) -> None:
        """Text overlay defaults to False."""
        st = SceneTriage(
            source_video=Path("v.mp4"),
            scene_index=0,
            start_time=0.0,
            end_time=5.0,
        )
        assert st.has_text_overlay is False
        assert st.text_overlay_score == 0.0

    def test_text_overlay_flagged(self) -> None:
        """Text overlay can be flagged."""
        st = SceneTriage(
            source_video=Path("v.mp4"),
            scene_index=0,
            start_time=0.0,
            end_time=5.0,
            has_text_overlay=True,
            text_overlay_score=0.35,
        )
        assert st.has_text_overlay is True
        assert st.text_overlay_score == 0.35


# ---------------------------------------------------------------------------
# VideoTriageReport model tests
# ---------------------------------------------------------------------------


class TestVideoTriageReport:
    """Tests for the VideoTriageReport dataclass."""

    def _make_scene(
        self,
        video: str = "v.mp4",
        index: int = 0,
        matched: bool = False,
        text: bool = False,
    ) -> SceneTriage:
        st = SceneTriage(
            source_video=Path(video),
            scene_index=index,
            start_time=index * 5.0,
            end_time=(index + 1) * 5.0,
            has_text_overlay=text,
        )
        if matched:
            ref = ConceptReference("x", ConceptType.CHARACTER, Path("x.jpg"), "char")
            st.matches = [ClipMatch(concept=ref, similarity=0.8, best_frame_index=0)]
        return st

    def test_empty_report(self) -> None:
        report = VideoTriageReport()
        assert report.total == 0
        assert report.matched_count == 0
        assert report.unmatched_count == 0
        assert report.text_overlay_count == 0
        assert report.videos == []

    def test_counts(self) -> None:
        scenes = [
            self._make_scene("a.mp4", 0, matched=True),
            self._make_scene("a.mp4", 1, matched=False),
            self._make_scene("a.mp4", 2, text=True),
            self._make_scene("b.mp4", 0, matched=True),
        ]
        report = VideoTriageReport(scenes=scenes)
        assert report.total == 4
        assert report.matched_count == 2
        # unmatched includes text overlay scenes (no concept matches)
        assert report.unmatched_count == 2
        assert report.text_overlay_count == 1

    def test_videos_property(self) -> None:
        """videos returns unique paths in order of first appearance."""
        scenes = [
            self._make_scene("a.mp4", 0),
            self._make_scene("b.mp4", 0),
            self._make_scene("a.mp4", 1),
        ]
        report = VideoTriageReport(scenes=scenes)
        assert report.videos == [Path("a.mp4"), Path("b.mp4")]

    def test_default_fields(self) -> None:
        report = VideoTriageReport()
        assert report.threshold == 0.70
        assert "clip" in report.model_name.lower()
        assert report.scene_detection_threshold == 27.0
        assert report.frames_per_scene == 2


# ---------------------------------------------------------------------------
# sample_scene_frames tests
# ---------------------------------------------------------------------------


class TestSampleSceneFrames:
    """Tests for sample_scene_frames() — uses mock ffmpeg."""

    def test_file_not_found(self) -> None:
        """Raises FileNotFoundError for missing video."""
        from klippbok.triage.sampler import sample_scene_frames
        with pytest.raises(FileNotFoundError):
            sample_scene_frames(Path("/nonexistent/video.mp4"), 0.0, 5.0)

    def test_invalid_time_range(self, tmp_path: Path) -> None:
        """Raises ValueError if start >= end."""
        from klippbok.triage.sampler import sample_scene_frames
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        with pytest.raises(ValueError, match="start_time"):
            sample_scene_frames(video, 5.0, 5.0)
        with pytest.raises(ValueError, match="start_time"):
            sample_scene_frames(video, 10.0, 5.0)

    def test_invalid_count(self, tmp_path: Path) -> None:
        """Raises ValueError if count < 1."""
        from klippbok.triage.sampler import sample_scene_frames
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        with pytest.raises(ValueError, match="count"):
            sample_scene_frames(video, 0.0, 5.0, count=0)

    def test_calculates_timestamps_correctly(self, tmp_path: Path) -> None:
        """Timestamps are evenly spaced, avoiding edges."""
        from klippbok.triage.sampler import sample_scene_frames

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        output_dir = tmp_path / "frames"

        captured_cmds: list[list[str]] = []

        def mock_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            # Create the output file
            out_path = cmd[-1]
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(b"fake png")
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("klippbok.triage.sampler.subprocess.run", side_effect=mock_run):
            frames = sample_scene_frames(
                video, 10.0, 20.0, count=2, output_dir=output_dir
            )

        assert len(frames) == 2
        assert len(captured_cmds) == 2

        # For count=2 in [10, 20]: timestamps at 10 + 10*(1/3) and 10 + 10*(2/3)
        # = 13.333 and 16.667
        ss_values = [cmd[cmd.index("-ss") + 1] for cmd in captured_cmds]
        assert abs(float(ss_values[0]) - 13.333) < 0.01
        assert abs(float(ss_values[1]) - 16.667) < 0.01

    def test_single_frame(self, tmp_path: Path) -> None:
        """count=1 extracts frame at scene midpoint."""
        from klippbok.triage.sampler import sample_scene_frames

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        captured_cmds: list[list[str]] = []

        def mock_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(cmd[-1]).write_bytes(b"fake png")
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("klippbok.triage.sampler.subprocess.run", side_effect=mock_run):
            frames = sample_scene_frames(
                video, 0.0, 10.0, count=1, output_dir=tmp_path / "out"
            )

        assert len(frames) == 1
        # Midpoint: 0 + 10 * (1/2) = 5.0
        ss = float(captured_cmds[0][captured_cmds[0].index("-ss") + 1])
        assert abs(ss - 5.0) < 0.01

    def test_ffmpeg_failure(self, tmp_path: Path) -> None:
        """Raises RuntimeError on ffmpeg failure."""
        from klippbok.triage.sampler import sample_scene_frames

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 1
            result.stderr = b"Error: something went wrong"
            return result

        with patch("klippbok.triage.sampler.subprocess.run", side_effect=mock_run):
            with pytest.raises(RuntimeError, match="scene frame extraction"):
                sample_scene_frames(
                    video, 0.0, 5.0, count=1, output_dir=tmp_path / "out"
                )

    def test_uses_temp_dir_when_no_output(self, tmp_path: Path) -> None:
        """Creates temp directory if output_dir is None."""
        from klippbok.triage.sampler import sample_scene_frames

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        def mock_run(cmd, **kwargs):
            Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(cmd[-1]).write_bytes(b"fake png")
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("klippbok.triage.sampler.subprocess.run", side_effect=mock_run):
            frames = sample_scene_frames(video, 0.0, 5.0, count=1)

        assert len(frames) == 1
        # Should be in a klippbok_scene_ prefixed temp directory
        assert "klippbok_scene_" in str(frames[0].parent)


# ---------------------------------------------------------------------------
# Scene manifest I/O tests
# ---------------------------------------------------------------------------


class TestSceneManifestIO:
    """Tests for _write_scene_manifest and _load_triage_manifest."""

    def _make_report(self) -> VideoTriageReport:
        """Create a sample VideoTriageReport."""
        ref = ConceptReference("holly", ConceptType.CHARACTER, Path("h.jpg"), "character")
        scenes = [
            SceneTriage(
                source_video=Path("C:/videos/001.mp4"),
                scene_index=0,
                start_time=0.0,
                end_time=5.2,
                matches=[ClipMatch(concept=ref, similarity=0.82, best_frame_index=1)],
            ),
            SceneTriage(
                source_video=Path("C:/videos/001.mp4"),
                scene_index=1,
                start_time=5.2,
                end_time=12.8,
            ),
            SceneTriage(
                source_video=Path("C:/videos/001.mp4"),
                scene_index=2,
                start_time=12.8,
                end_time=18.0,
                has_text_overlay=True,
                text_overlay_score=0.32,
            ),
        ]
        return VideoTriageReport(
            scenes=scenes,
            concepts=[ref],
            threshold=0.70,
            scene_detection_threshold=27.0,
            frames_per_scene=2,
        )

    def test_write_scene_manifest(self, tmp_path: Path) -> None:
        """Writes valid JSON with correct structure."""
        from klippbok.triage.triage import _write_scene_manifest

        report = self._make_report()
        output = tmp_path / "manifest.json"
        _write_scene_manifest(report, output)

        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))

        assert data["triage_mode"] == "scene"
        assert data["triage"]["total_scenes"] == 3
        assert data["triage"]["matched"] == 1
        assert data["triage"]["unmatched"] == 2  # includes text overlay
        assert data["triage"]["text_overlays"] == 1
        assert len(data["videos"]) == 1
        assert len(data["videos"][0]["scenes"]) == 3

    def test_include_field(self, tmp_path: Path) -> None:
        """include is True for matched non-text scenes, False otherwise."""
        from klippbok.triage.triage import _write_scene_manifest

        report = self._make_report()
        output = tmp_path / "manifest.json"
        _write_scene_manifest(report, output)

        data = json.loads(output.read_text(encoding="utf-8"))
        scenes = data["videos"][0]["scenes"]

        assert scenes[0]["include"] is True   # matched
        assert scenes[1]["include"] is False  # unmatched
        assert scenes[2]["include"] is False  # text overlay

    def test_load_triage_manifest(self, tmp_path: Path) -> None:
        """_load_triage_manifest reads manifest and extracts included segments."""
        from klippbok.video.__main__ import _load_triage_manifest

        manifest = {
            "triage_mode": "scene",
            "videos": [
                {
                    "file": "001.mp4",
                    "path": "C:\\videos\\001.mp4",
                    "total_scenes": 3,
                    "scenes": [
                        {"scene_index": 0, "start_time": 0.0, "end_time": 5.2, "include": True},
                        {"scene_index": 1, "start_time": 5.2, "end_time": 12.8, "include": False},
                        {"scene_index": 2, "start_time": 12.8, "end_time": 18.0, "include": True},
                    ],
                },
                {
                    "file": "002.mp4",
                    "path": "C:\\videos\\002.mp4",
                    "total_scenes": 2,
                    "scenes": [
                        {"scene_index": 0, "start_time": 0.0, "end_time": 8.0, "include": False},
                        {"scene_index": 1, "start_time": 8.0, "end_time": 15.0, "include": False},
                    ],
                },
            ],
        }
        manifest_path = tmp_path / "scene_triage_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        result = _load_triage_manifest(manifest_path)

        # 001.mp4 has scenes 0 and 2 included — as (start, end) tuples
        assert "C:\\videos\\001.mp4" in result
        assert result["C:\\videos\\001.mp4"] == [(0.0, 5.2), (12.8, 18.0)]
        # 002.mp4 has no included scenes — not in result
        assert "C:\\videos\\002.mp4" not in result

    def test_load_manifest_not_found(self) -> None:
        """Raises FileNotFoundError for missing manifest."""
        from klippbok.video.__main__ import _load_triage_manifest
        with pytest.raises(FileNotFoundError):
            _load_triage_manifest(Path("/nonexistent/manifest.json"))

    def test_load_manifest_wrong_mode(self, tmp_path: Path) -> None:
        """Raises ValueError if triage_mode is not 'scene'."""
        from klippbok.video.__main__ import _load_triage_manifest
        manifest_path = tmp_path / "bad.json"
        manifest_path.write_text('{"triage_mode": "clip"}', encoding="utf-8")
        with pytest.raises(ValueError, match="triage_mode"):
            _load_triage_manifest(manifest_path)

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Write then load produces correct segments."""
        from klippbok.triage.triage import _write_scene_manifest
        from klippbok.video.__main__ import _load_triage_manifest

        report = self._make_report()
        output = tmp_path / "manifest.json"
        _write_scene_manifest(report, output)

        result = _load_triage_manifest(output)

        # Only scene 0 is matched and not text overlay (0.0 - 5.2)
        video_path = str(Path("C:/videos/001.mp4"))
        assert video_path in result
        assert len(result[video_path]) == 1
        start, end = result[video_path][0]
        assert abs(start - 0.0) < 0.001
        assert abs(end - 5.2) < 0.001

    def test_multi_video_manifest(self, tmp_path: Path) -> None:
        """Manifest with multiple videos loads correctly."""
        from klippbok.triage.triage import _write_scene_manifest
        from klippbok.video.__main__ import _load_triage_manifest

        ref = ConceptReference("holly", ConceptType.CHARACTER, Path("h.jpg"), "character")
        scenes = [
            SceneTriage(
                source_video=Path("C:/videos/a.mp4"),
                scene_index=0, start_time=0.0, end_time=5.0,
                matches=[ClipMatch(concept=ref, similarity=0.85, best_frame_index=0)],
            ),
            SceneTriage(
                source_video=Path("C:/videos/b.mp4"),
                scene_index=0, start_time=0.0, end_time=8.0,
            ),
            SceneTriage(
                source_video=Path("C:/videos/b.mp4"),
                scene_index=1, start_time=8.0, end_time=12.0,
                matches=[ClipMatch(concept=ref, similarity=0.78, best_frame_index=0)],
            ),
        ]
        report = VideoTriageReport(scenes=scenes, concepts=[ref])
        output = tmp_path / "manifest.json"
        _write_scene_manifest(report, output)

        result = _load_triage_manifest(output)

        # a.mp4: scene 0 matched (0.0 - 5.0)
        a_path = str(Path("C:/videos/a.mp4"))
        assert a_path in result
        assert len(result[a_path]) == 1
        assert result[a_path][0] == (0.0, 5.0)

        # b.mp4: scene 1 matched (8.0 - 12.0), scene 0 unmatched
        b_path = str(Path("C:/videos/b.mp4"))
        assert b_path in result
        assert len(result[b_path]) == 1
        assert result[b_path][0] == (8.0, 12.0)


# ---------------------------------------------------------------------------
# triage_videos orchestrator tests (mocked)
# ---------------------------------------------------------------------------


class TestTriageVideosOrchestrator:
    """Tests for triage_videos() with mocked dependencies."""

    def _setup_concepts(self, tmp_path: Path) -> Path:
        """Create a concepts/ directory with a reference image."""
        concepts_dir = tmp_path / "concepts"
        char_dir = concepts_dir / "character"
        char_dir.mkdir(parents=True)
        (char_dir / "holly.jpg").write_bytes(b"\xff\xd8\xff\xe0")
        return concepts_dir

    def test_no_concepts_returns_empty(self, tmp_path: Path) -> None:
        """Returns empty report when no reference images found."""
        from klippbok.triage.triage import triage_videos

        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()

        # Mock CLIP availability
        with patch("klippbok.triage.triage.check_clip_available"):
            report = triage_videos(
                video_paths=[tmp_path / "v.mp4"],
                concepts_dir=concepts_dir,
            )

        assert report.total == 0
        assert report.concepts == []


# ---------------------------------------------------------------------------
# Duration-adaptive triage_clips tests
# ---------------------------------------------------------------------------


class TestDurationAdaptiveTriage:
    """Tests that triage_clips auto-detects long vs short videos."""

    def test_long_video_threshold_constant(self) -> None:
        """LONG_VIDEO_THRESHOLD is 30 seconds."""
        from klippbok.triage.triage import LONG_VIDEO_THRESHOLD
        assert LONG_VIDEO_THRESHOLD == 30.0

    def test_short_clips_stay_in_clip_mode(self, tmp_path: Path) -> None:
        """Videos under 30s use existing clip-level triage."""
        from klippbok.triage.triage import triage_clips

        # Create fake clips and concepts
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        (clips_dir / "clip1.mp4").write_bytes(b"fake")
        (clips_dir / "clip2.mp4").write_bytes(b"fake")

        concepts_dir = tmp_path / "concepts"
        (concepts_dir / "character").mkdir(parents=True)
        (concepts_dir / "character" / "holly.jpg").write_bytes(b"\xff\xd8\xff\xe0")

        # Mock: durations are short (5s), CLIP is available
        with patch("klippbok.triage.triage.check_clip_available"), \
             patch("klippbok.triage.triage._get_duration", return_value=5.0), \
             patch("klippbok.triage.triage.CLIPEmbedder") as mock_clip, \
             patch("klippbok.triage.triage.build_prompt_cache", return_value=[]), \
             patch("klippbok.triage.triage.sample_clip_frames", return_value=[]), \
             patch("klippbok.triage.triage.detect_text_overlays", return_value=(False, 0.0)):

            mock_embedder = MagicMock()
            mock_embedder.encode_image.return_value = np.zeros(512, dtype=np.float32)
            mock_clip.return_value = mock_embedder

            report = triage_clips(
                clips_dir=clips_dir,
                concepts_dir=concepts_dir,
            )

        # Should return a TriageReport (clip-level), not VideoTriageReport
        from klippbok.triage.models import TriageReport
        assert isinstance(report, TriageReport)

    def test_long_videos_switch_to_scene_mode(self, tmp_path: Path) -> None:
        """Videos >= 30s delegate to triage_videos (scene-aware)."""
        from klippbok.triage.triage import triage_clips

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        (clips_dir / "movie1.mp4").write_bytes(b"fake")
        (clips_dir / "movie2.mp4").write_bytes(b"fake")

        concepts_dir = tmp_path / "concepts"
        (concepts_dir / "character").mkdir(parents=True)
        (concepts_dir / "character" / "holly.jpg").write_bytes(b"\xff\xd8\xff\xe0")

        # Mock: durations are long (120s)
        with patch("klippbok.triage.triage.check_clip_available"), \
             patch("klippbok.triage.triage._get_duration", return_value=120.0), \
             patch("klippbok.triage.triage.triage_videos") as mock_tv:

            mock_tv.return_value = VideoTriageReport()
            report = triage_clips(
                clips_dir=clips_dir,
                concepts_dir=concepts_dir,
            )

        # Should have delegated to triage_videos
        mock_tv.assert_called_once()
        assert isinstance(report, VideoTriageReport)
