"""Tests for triage folder organization.

Tests organize_clips() — sorting clips into concept-named folders
after triage matching. Uses temp files, no CLIP/torch required.
"""

from pathlib import Path

import pytest

from klippbok.triage.models import (
    ClipMatch,
    ClipTriage,
    ConceptReference,
    ConceptType,
    TriageReport,
)
from klippbok.triage.triage import organize_clips


def _make_ref(name: str, concept_type: ConceptType) -> ConceptReference:
    """Create a mock ConceptReference."""
    return ConceptReference(
        name=name,
        concept_type=concept_type,
        image_path=Path(f"/fake/concepts/{concept_type.value}/{name}.jpg"),
        folder_name=concept_type.value,
    )


def _make_clip(
    tmp_path: Path,
    filename: str,
    ref: ConceptReference | None = None,
    similarity: float = 0.85,
) -> ClipTriage:
    """Create a fake video clip file and a ClipTriage for it."""
    clip_path = tmp_path / "clips" / filename
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(b"fake video data")

    triage = ClipTriage(clip_path=clip_path)
    if ref is not None:
        triage.matches = [
            ClipMatch(concept=ref, similarity=similarity, best_frame_index=0)
        ]
    return triage


class TestOrganizeClips:
    """Tests for organize_clips()."""

    def test_matched_clips_go_to_concept_folders(self, tmp_path: Path) -> None:
        """Matched clips are copied into folders named after their best match."""
        holly_ref = _make_ref("hollygolightly", ConceptType.CHARACTER)
        cat_ref = _make_ref("cat", ConceptType.CHARACTER)

        clip1 = _make_clip(tmp_path, "scene001.mp4", holly_ref)
        clip2 = _make_clip(tmp_path, "scene002.mp4", cat_ref)

        report = TriageReport(
            clips=[clip1, clip2],
            concepts=[holly_ref, cat_ref],
        )

        output_dir = tmp_path / "organized"
        result = organize_clips(report, output_dir, copy=True)

        assert "hollygolightly" in result
        assert "cat" in result
        assert (output_dir / "hollygolightly" / "scene001.mp4").exists()
        assert (output_dir / "cat" / "scene002.mp4").exists()

    def test_unmatched_clips_go_to_unmatched_folder(self, tmp_path: Path) -> None:
        """Clips with no matches go to 'unmatched/' folder."""
        clip = _make_clip(tmp_path, "unknown_scene.mp4")  # no ref = unmatched

        report = TriageReport(clips=[clip])

        output_dir = tmp_path / "organized"
        result = organize_clips(report, output_dir, copy=True)

        assert "unmatched" in result
        assert (output_dir / "unmatched" / "unknown_scene.mp4").exists()

    def test_mixed_matched_and_unmatched(self, tmp_path: Path) -> None:
        """Mix of matched and unmatched clips."""
        ref = _make_ref("holly", ConceptType.CHARACTER)

        matched = _make_clip(tmp_path, "scene001.mp4", ref)
        unmatched = _make_clip(tmp_path, "scene002.mp4")

        report = TriageReport(clips=[matched, unmatched], concepts=[ref])

        output_dir = tmp_path / "organized"
        result = organize_clips(report, output_dir, copy=True)

        assert len(result["holly"]) == 1
        assert len(result["unmatched"]) == 1

    def test_copy_mode_preserves_originals(self, tmp_path: Path) -> None:
        """Copy mode leaves original files in place."""
        ref = _make_ref("holly", ConceptType.CHARACTER)
        clip = _make_clip(tmp_path, "scene001.mp4", ref)
        original_path = clip.clip_path

        output_dir = tmp_path / "organized"
        organize_clips(
            TriageReport(clips=[clip], concepts=[ref]),
            output_dir,
            copy=True,
        )

        # Both original and copy should exist
        assert original_path.exists()
        assert (output_dir / "holly" / "scene001.mp4").exists()

    def test_move_mode_removes_originals(self, tmp_path: Path) -> None:
        """Move mode removes original files."""
        ref = _make_ref("holly", ConceptType.CHARACTER)
        clip = _make_clip(tmp_path, "scene001.mp4", ref)
        original_path = clip.clip_path

        output_dir = tmp_path / "organized"
        organize_clips(
            TriageReport(clips=[clip], concepts=[ref]),
            output_dir,
            copy=False,  # move
        )

        # Original should be gone, organized copy should exist
        assert not original_path.exists()
        assert (output_dir / "holly" / "scene001.mp4").exists()

    def test_sidecars_copied(self, tmp_path: Path) -> None:
        """Sidecar files (.txt, .json) are copied alongside videos."""
        ref = _make_ref("holly", ConceptType.CHARACTER)
        clip = _make_clip(tmp_path, "scene001.mp4", ref)

        # Create sidecar files
        caption = clip.clip_path.with_suffix(".txt")
        caption.write_text("A woman walks down the street.")
        meta = clip.clip_path.with_suffix(".json")
        meta.write_text('{"source": "test"}')

        output_dir = tmp_path / "organized"
        organize_clips(
            TriageReport(clips=[clip], concepts=[ref]),
            output_dir,
            copy=True,
        )

        assert (output_dir / "holly" / "scene001.txt").exists()
        assert (output_dir / "holly" / "scene001.json").exists()

    def test_multiple_clips_same_concept(self, tmp_path: Path) -> None:
        """Multiple clips matching the same concept go to the same folder."""
        ref = _make_ref("holly", ConceptType.CHARACTER)
        clip1 = _make_clip(tmp_path, "scene001.mp4", ref, 0.90)
        clip2 = _make_clip(tmp_path, "scene002.mp4", ref, 0.85)
        clip3 = _make_clip(tmp_path, "scene003.mp4", ref, 0.75)

        report = TriageReport(clips=[clip1, clip2, clip3], concepts=[ref])

        output_dir = tmp_path / "organized"
        result = organize_clips(report, output_dir, copy=True)

        assert len(result["holly"]) == 3
        for i in range(1, 4):
            assert (output_dir / "holly" / f"scene00{i}.mp4").exists()

    def test_output_dir_created(self, tmp_path: Path) -> None:
        """Output directory is created if it doesn't exist."""
        ref = _make_ref("holly", ConceptType.CHARACTER)
        clip = _make_clip(tmp_path, "scene001.mp4", ref)

        output_dir = tmp_path / "deep" / "nested" / "organized"
        organize_clips(
            TriageReport(clips=[clip], concepts=[ref]),
            output_dir,
            copy=True,
        )

        assert output_dir.exists()
        assert (output_dir / "holly" / "scene001.mp4").exists()

    def test_empty_report(self, tmp_path: Path) -> None:
        """Empty report produces no folders."""
        report = TriageReport()
        output_dir = tmp_path / "organized"
        result = organize_clips(report, output_dir, copy=True)
        assert result == {}

    def test_returns_organized_paths(self, tmp_path: Path) -> None:
        """Return value maps folder names to lists of destination paths."""
        ref = _make_ref("holly", ConceptType.CHARACTER)
        clip = _make_clip(tmp_path, "scene001.mp4", ref)

        output_dir = tmp_path / "organized"
        result = organize_clips(
            TriageReport(clips=[clip], concepts=[ref]),
            output_dir,
            copy=True,
        )

        assert "holly" in result
        assert len(result["holly"]) == 1
        assert result["holly"][0] == output_dir / "holly" / "scene001.mp4"


class TestOrganizeCliFlags:
    """Tests for the --organize and --move CLI flags."""

    def test_organize_flag(self) -> None:
        """--organize accepts a directory path."""
        from klippbok.video.__main__ import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "triage", ".", "-s", "concepts/",
            "--organize", "/path/to/output",
        ])
        assert args.organize == "/path/to/output"

    def test_organize_default_none(self) -> None:
        """--organize defaults to None."""
        from klippbok.video.__main__ import build_parser
        parser = build_parser()
        args = parser.parse_args(["triage", ".", "-s", "concepts/"])
        assert args.organize is None

    def test_move_flag(self) -> None:
        """--move flag defaults to False."""
        from klippbok.video.__main__ import build_parser
        parser = build_parser()
        args = parser.parse_args(["triage", ".", "-s", "concepts/"])
        assert args.move is False

    def test_move_flag_set(self) -> None:
        """--move flag can be set."""
        from klippbok.video.__main__ import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "triage", ".", "-s", "concepts/",
            "--organize", "output/", "--move",
        ])
        assert args.move is True


class TestIngestCliFlags:
    """Tests for ingest CLI changes: --max-frames, directory support."""

    def test_max_frames_flag(self) -> None:
        """--max-frames accepts an integer."""
        from klippbok.video.__main__ import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "ingest", "video.mp4", "-o", "output/",
            "--max-frames", "49",
        ])
        assert args.max_frames == 49

    def test_max_frames_default_none(self) -> None:
        """--max-frames defaults to None (use config default)."""
        from klippbok.video.__main__ import build_parser
        parser = build_parser()
        args = parser.parse_args(["ingest", "video.mp4", "-o", "output/"])
        assert args.max_frames is None

    def test_max_frames_zero_means_no_limit(self) -> None:
        """--max-frames 0 means no frame limit."""
        from klippbok.video.__main__ import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "ingest", "video.mp4", "-o", "output/",
            "--max-frames", "0",
        ])
        assert args.max_frames == 0

    def test_directory_as_input(self) -> None:
        """ingest accepts a directory path as the video argument."""
        from klippbok.video.__main__ import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "ingest", "/path/to/videos/", "-o", "output/",
        ])
        assert args.video == "/path/to/videos/"

    def test_ingest_help_text(self) -> None:
        """Ingest help mentions directory support."""
        from klippbok.video.__main__ import build_parser
        parser = build_parser()
        # Check the subparser help text
        for action in parser._subparsers._actions:
            if hasattr(action, '_parser_class'):
                for name, subparser in action.choices.items():
                    if name == "ingest":
                        help_text = subparser.format_help()
                        assert "directory" in help_text.lower()
