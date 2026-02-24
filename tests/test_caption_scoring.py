"""Tests for klippbok.caption.scoring — caption quality scoring.

All pure function tests except score_directory which uses tmp_path.
No API calls, no heavy dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from klippbok.caption.scoring import (
    CaptionScore,
    ScoringConfig,
    _score_length,
    _score_repetition,
    _score_specificity,
    _score_temporal,
    format_score_report,
    score_caption,
    score_directory,
)


# ---------------------------------------------------------------------------
# ScoringConfig tests
# ---------------------------------------------------------------------------

class TestScoringConfig:
    """Tests for ScoringConfig defaults and customization."""

    def test_defaults(self) -> None:
        config = ScoringConfig()
        assert config.min_good_length == 80
        assert config.max_good_length == 400
        assert config.weight_length == 0.25
        assert config.weight_temporal == 0.30
        assert config.weight_specificity == 0.25
        assert config.weight_repetition == 0.20

    def test_weights_sum_to_one(self) -> None:
        """Default weights should sum to 1.0 for intuitive overall score."""
        config = ScoringConfig()
        total = (config.weight_length + config.weight_temporal
                 + config.weight_specificity + config.weight_repetition)
        assert abs(total - 1.0) < 0.001

    def test_custom_thresholds(self) -> None:
        config = ScoringConfig(min_good_length=100, max_good_length=300)
        assert config.min_good_length == 100
        assert config.max_good_length == 300


# ---------------------------------------------------------------------------
# Length scoring tests
# ---------------------------------------------------------------------------

class TestScoreLength:
    """Tests for _score_length() dimension."""

    def test_empty_caption(self) -> None:
        score, issues = _score_length("", ScoringConfig())
        assert score == 0.0
        assert any("Empty" in i for i in issues)

    def test_very_short(self) -> None:
        """Below min_acceptable (50 chars) scores 0.2."""
        score, issues = _score_length("Short text", ScoringConfig())
        assert score == 0.2
        assert any("Very short" in i for i in issues)

    def test_ideal_length(self) -> None:
        """Caption in sweet spot (80-400 chars) scores 1.0."""
        caption = "A young woman walks through a dimly lit alley, " * 3
        assert len(caption) > 80
        assert len(caption) < 400
        score, issues = _score_length(caption, ScoringConfig())
        assert score == 1.0
        assert issues == []

    def test_too_long(self) -> None:
        """Caption over max_acceptable (600) gets penalized."""
        caption = "word " * 200  # ~1000 chars
        score, issues = _score_length(caption, ScoringConfig())
        assert score < 0.5
        assert any("Very long" in i for i in issues)

    def test_slightly_short(self) -> None:
        """Caption between min_acceptable and min_good gets partial score."""
        caption = "A" * 65  # 65 chars, between 50 and 80
        score, issues = _score_length(caption, ScoringConfig())
        assert 0.4 < score < 1.0

    def test_slightly_long(self) -> None:
        """Caption between max_good and max_acceptable gets partial score."""
        caption = "A" * 500  # between 400 and 600
        score, issues = _score_length(caption, ScoringConfig())
        assert 0.5 < score < 1.0


# ---------------------------------------------------------------------------
# Temporal awareness tests
# ---------------------------------------------------------------------------

class TestScoreTemporal:
    """Tests for _score_temporal() dimension."""

    def test_no_temporal_words(self) -> None:
        """Caption with zero temporal language scores 0.0."""
        score, issues = _score_temporal(
            "A cat on a red couch in a bright room",
            ScoringConfig(),
        )
        assert score == 0.0
        assert any("No temporal" in i for i in issues)

    def test_single_temporal_word(self) -> None:
        """One temporal word gives partial score."""
        score, _ = _score_temporal(
            "A cat walks across the room",
            ScoringConfig(),
        )
        assert score == 0.3

    def test_multiple_temporal_words(self) -> None:
        """Multiple temporal words increase score."""
        score, _ = _score_temporal(
            "The camera pans slowly as a figure walks then turns around",
            ScoringConfig(),
        )
        # "pans", "slowly", "walks", "turns" = 4 matches
        assert score >= 0.7

    def test_rich_temporal_language(self) -> None:
        """6+ temporal words scores 1.0."""
        score, _ = _score_temporal(
            "The camera pans slowly as a figure walks then turns, "
            "gradually spinning while zooming in and the light fades",
            ScoringConfig(),
        )
        assert score == 1.0

    def test_camera_motion_counts(self) -> None:
        """Camera motion terms are recognized as temporal."""
        score, _ = _score_temporal(
            "The scene is captured with handheld tracking as the subject runs",
            ScoringConfig(),
        )
        assert score > 0.0


# ---------------------------------------------------------------------------
# Specificity tests
# ---------------------------------------------------------------------------

class TestScoreSpecificity:
    """Tests for _score_specificity() dimension."""

    def test_neutral_caption(self) -> None:
        """Caption with no vague phrases and no specific terms scores 0.5."""
        score, issues = _score_specificity(
            "A person stands near a table",
            ScoringConfig(),
        )
        assert score == 0.5
        assert issues == []

    def test_vague_phrases_penalized(self) -> None:
        """Vague phrases reduce the score."""
        score, issues = _score_specificity(
            "The video shows a beautiful scene with a nice atmosphere",
            ScoringConfig(),
        )
        assert score < 0.5
        assert any("Vague" in i for i in issues)

    def test_specific_terms_rewarded(self) -> None:
        """Specific spatial/lighting terms increase the score."""
        score, _ = _score_specificity(
            "A close-up shot with soft light in the foreground, "
            "warm tones and shallow focus creating bokeh in the background",
            ScoringConfig(),
        )
        assert score > 0.5

    def test_score_clamped_to_range(self) -> None:
        """Score never exceeds 1.0 or goes below 0.0."""
        # Many specific terms
        caption = " ".join([
            "foreground", "background", "close-up", "wide shot",
            "backlit", "silhouette", "bokeh", "warm tones",
            "high contrast", "low angle", "amber", "muted",
        ])
        score, _ = _score_specificity(caption, ScoringConfig())
        assert 0.0 <= score <= 1.0

        # Many vague phrases
        caption = "The video shows a beautiful scene we can see, a nice overall vibe"
        score, _ = _score_specificity(caption, ScoringConfig())
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Repetition tests
# ---------------------------------------------------------------------------

class TestScoreRepetition:
    """Tests for _score_repetition() dimension."""

    def test_no_repetition(self) -> None:
        """Caption with unique trigrams scores 1.0."""
        score, issues = _score_repetition(
            "A young woman walks through a brightly lit corridor at night",
            ScoringConfig(),
        )
        assert score == 1.0
        assert issues == []

    def test_heavy_repetition(self) -> None:
        """Caption with lots of repeated trigrams scores low."""
        # Extreme case: same phrase repeated many times
        caption = "the cat sits " * 10
        score, issues = _score_repetition(caption, ScoringConfig())
        assert score < 0.5
        assert any("repetition" in i.lower() for i in issues)

    def test_short_caption_skipped(self) -> None:
        """Captions too short for trigram analysis score 1.0."""
        score, issues = _score_repetition("hello world", ScoringConfig())
        assert score == 1.0

    def test_mild_repetition(self) -> None:
        """Some repetition gets moderate score."""
        caption = "the red ball bounced and the red ball rolled and the red ball stopped"
        score, _ = _score_repetition(caption, ScoringConfig())
        assert 0.3 < score < 1.0


# ---------------------------------------------------------------------------
# score_caption (integration) tests
# ---------------------------------------------------------------------------

class TestScoreCaption:
    """Tests for score_caption() — full scoring pipeline."""

    def test_good_caption(self) -> None:
        """Well-written video caption gets high overall score."""
        caption = (
            "A young woman walks slowly through a dimly lit alley, the camera "
            "panning to follow her movement. Warm amber light spills from a "
            "window in the background as she turns to look over her shoulder."
        )
        result = score_caption(caption)
        assert result.overall > 0.5
        assert result.length_score > 0.8
        assert result.temporal_score > 0.0

    def test_bad_caption(self) -> None:
        """Short, vague caption gets low overall score."""
        result = score_caption("The video shows a nice scene.")
        assert result.overall < 0.4
        assert len(result.issues) > 0

    def test_returns_caption_score(self) -> None:
        """Returns a CaptionScore dataclass."""
        result = score_caption("Test caption text for scoring")
        assert isinstance(result, CaptionScore)
        assert result.caption == "Test caption text for scoring"

    def test_custom_config(self) -> None:
        """Accepts custom ScoringConfig."""
        config = ScoringConfig(
            min_acceptable_length=5,
            min_good_length=10,
            max_good_length=50,
        )
        result = score_caption("Short but good enough", config)
        assert result.length_score == 1.0

    def test_overall_is_weighted_average(self) -> None:
        """Overall score is a weighted average of dimension scores."""
        result = score_caption("A person walks through a forest with warm tones and soft light in the foreground, the camera panning slowly")
        config = ScoringConfig()
        expected = (
            config.weight_length * result.length_score
            + config.weight_temporal * result.temporal_score
            + config.weight_specificity * result.specificity_score
            + config.weight_repetition * result.repetition_score
        )
        assert abs(result.overall - expected) < 0.001

    def test_empty_caption(self) -> None:
        """Empty caption gets low score."""
        result = score_caption("")
        assert result.overall < 0.35
        assert result.length_score == 0.0
        assert any("Empty" in i for i in result.issues)


# ---------------------------------------------------------------------------
# score_directory tests
# ---------------------------------------------------------------------------

class TestScoreDirectory:
    """Tests for score_directory() — filesystem scoring."""

    def test_scores_all_txt_files(self, tmp_path: Path) -> None:
        """Scores every .txt file in the directory."""
        (tmp_path / "clip_01.txt").write_text("A woman walks through a forest slowly")
        (tmp_path / "clip_02.txt").write_text("The camera pans across a wide landscape")
        (tmp_path / "clip_03.txt").write_text("Short")

        scores = score_directory(tmp_path)
        assert len(scores) == 3

    def test_sorted_worst_first(self, tmp_path: Path) -> None:
        """Results are sorted by overall score, worst first."""
        (tmp_path / "good.txt").write_text(
            "A young woman walks slowly through a dimly lit alley, the camera "
            "panning to follow her movement. Warm light spills from a window."
        )
        (tmp_path / "bad.txt").write_text("Short")

        scores = score_directory(tmp_path)
        assert scores[0].overall <= scores[-1].overall

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Returns empty list for directory with no .txt files."""
        scores = score_directory(tmp_path)
        assert scores == []

    def test_nonexistent_directory(self) -> None:
        """Raises FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            score_directory("/nonexistent/path")

    def test_ignores_non_txt_files(self, tmp_path: Path) -> None:
        """Only processes .txt files."""
        (tmp_path / "clip.mp4").write_bytes(b"\x00")
        (tmp_path / "clip.txt").write_text("A person walks slowly through a park")
        (tmp_path / "notes.md").write_text("# Notes")

        scores = score_directory(tmp_path)
        assert len(scores) == 1

    def test_paths_set_on_results(self, tmp_path: Path) -> None:
        """Results have path set to the .txt file."""
        (tmp_path / "clip_01.txt").write_text("A person walks slowly")
        scores = score_directory(tmp_path)
        assert scores[0].path == tmp_path / "clip_01.txt"


# ---------------------------------------------------------------------------
# format_score_report tests
# ---------------------------------------------------------------------------

class TestFormatScoreReport:
    """Tests for format_score_report()."""

    def test_empty_report(self) -> None:
        result = format_score_report([])
        assert "No captions" in result

    def test_includes_summary_stats(self, tmp_path: Path) -> None:
        """Report includes average, worst, best scores."""
        (tmp_path / "a.txt").write_text("Short")
        (tmp_path / "b.txt").write_text(
            "A young woman walks slowly through a forest, "
            "the camera panning to follow. Warm tones fill the background."
        )
        scores = score_directory(tmp_path)
        report = format_score_report(scores)
        assert "Average:" in report
        assert "Worst:" in report
        assert "Best:" in report

    def test_includes_per_file_scores(self, tmp_path: Path) -> None:
        """Report shows dimension breakdown for each file."""
        (tmp_path / "clip.txt").write_text("A person walks slowly through a park")
        scores = score_directory(tmp_path)
        report = format_score_report(scores)
        assert "L=" in report
        assert "T=" in report
        assert "S=" in report
        assert "R=" in report

    def test_includes_issues(self, tmp_path: Path) -> None:
        """Report shows detected issues."""
        (tmp_path / "clip.txt").write_text("Short")
        scores = score_directory(tmp_path)
        report = format_score_report(scores)
        assert "!" in report  # Issue marker
