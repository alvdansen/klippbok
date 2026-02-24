"""Tests for triage CLI integration in klippbok.video.__main__.

Tests parser flags and argument handling. Does NOT test actual triage
execution (that requires torch + CLIP).
"""

import pytest

from klippbok.video.__main__ import build_parser


class TestTriageParser:
    """Tests for the triage subcommand parser."""

    def test_triage_command_exists(self) -> None:
        """triage is a recognized subcommand."""
        parser = build_parser()
        args = parser.parse_args(["triage", ".", "--concepts", "concepts/"])
        assert args.command == "triage"

    def test_required_concepts_flag(self) -> None:
        """--concepts is required."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["triage", "."])

    def test_directory_argument(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["triage", "/path/to/clips", "-s", "concepts/"])
        assert args.directory == "/path/to/clips"

    def test_concepts_short_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["triage", ".", "-s", "my_concepts/"])
        assert args.concepts == "my_concepts/"

    def test_default_threshold(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["triage", ".", "-s", "concepts/"])
        assert args.threshold == 0.70

    def test_custom_threshold(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["triage", ".", "-s", "concepts/", "--threshold", "0.85"])
        assert args.threshold == 0.85

    def test_default_frames(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["triage", ".", "-s", "concepts/"])
        assert args.frames == 5

    def test_custom_frames(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["triage", ".", "-s", "concepts/", "--frames", "3"])
        assert args.frames == 3

    def test_output_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["triage", ".", "-s", "concepts/", "-o", "result.json"])
        assert args.output == "result.json"

    def test_default_output_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["triage", ".", "-s", "concepts/"])
        assert args.output is None

    def test_clip_model_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "triage", ".", "-s", "concepts/",
            "--clip-model", "openai/clip-vit-large-patch14",
        ])
        assert args.clip_model == "openai/clip-vit-large-patch14"

    def test_default_clip_model(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["triage", ".", "-s", "concepts/"])
        assert args.clip_model == "openai/clip-vit-base-patch32"
