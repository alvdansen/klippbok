"""Tests for klippbok.triage.models — pure Python, no external deps.

Tests concept types, type alias resolution, and data model behavior.
"""

import pytest
from pathlib import Path

from klippbok.triage.models import (
    ClipMatch,
    ClipTriage,
    ConceptReference,
    ConceptType,
    TriageReport,
    TYPE_ALIASES,
    resolve_concept_type,
)


class TestConceptType:
    """Tests for the ConceptType enum."""

    def test_all_values(self) -> None:
        """All five concept types exist."""
        assert ConceptType.CHARACTER == "character"
        assert ConceptType.STYLE == "style"
        assert ConceptType.MOTION == "motion"
        assert ConceptType.OBJECT == "object"
        assert ConceptType.SETTING == "setting"

    def test_five_types(self) -> None:
        """Exactly five concept types."""
        assert len(ConceptType) == 5

    def test_string_enum(self) -> None:
        """ConceptType values are plain strings."""
        for ct in ConceptType:
            assert isinstance(ct.value, str)


class TestTypeAliases:
    """Tests for TYPE_ALIASES and resolve_concept_type()."""

    def test_canonical_names(self) -> None:
        """Canonical names (character, style, etc.) resolve correctly."""
        assert resolve_concept_type("character") == ConceptType.CHARACTER
        assert resolve_concept_type("style") == ConceptType.STYLE
        assert resolve_concept_type("motion") == ConceptType.MOTION
        assert resolve_concept_type("object") == ConceptType.OBJECT
        assert resolve_concept_type("setting") == ConceptType.SETTING

    def test_case_insensitive(self) -> None:
        """Lookup is case-insensitive."""
        assert resolve_concept_type("Character") == ConceptType.CHARACTER
        assert resolve_concept_type("STYLE") == ConceptType.STYLE
        assert resolve_concept_type("Setting") == ConceptType.SETTING

    def test_squishy_character_aliases(self) -> None:
        """Common folder names for characters resolve correctly."""
        for alias in ["humans", "people", "face", "actors", "person"]:
            assert resolve_concept_type(alias) == ConceptType.CHARACTER, \
                f"'{alias}' should resolve to CHARACTER"

    def test_squishy_setting_aliases(self) -> None:
        """Common folder names for settings resolve correctly."""
        for alias in ["location", "place", "environment", "scene", "background"]:
            assert resolve_concept_type(alias) == ConceptType.SETTING, \
                f"'{alias}' should resolve to SETTING"

    def test_squishy_object_aliases(self) -> None:
        """Common folder names for objects resolve correctly."""
        for alias in ["things", "items", "prop", "props"]:
            assert resolve_concept_type(alias) == ConceptType.OBJECT, \
                f"'{alias}' should resolve to OBJECT"

    def test_strips_hyphens_underscores(self) -> None:
        """Hyphens and underscores are stripped before lookup."""
        assert resolve_concept_type("back-ground") == ConceptType.SETTING
        assert resolve_concept_type("back_ground") == ConceptType.SETTING

    def test_strips_spaces(self) -> None:
        """Spaces are stripped before lookup."""
        assert resolve_concept_type(" character ") == ConceptType.CHARACTER
        assert resolve_concept_type("back ground") == ConceptType.SETTING

    def test_unknown_returns_none(self) -> None:
        """Unrecognized names return None."""
        assert resolve_concept_type("foobar") is None
        assert resolve_concept_type("my-custom-folder") is None
        assert resolve_concept_type("") is None

    def test_plurals(self) -> None:
        """Plural forms are recognized."""
        assert resolve_concept_type("characters") == ConceptType.CHARACTER
        assert resolve_concept_type("objects") == ConceptType.OBJECT
        assert resolve_concept_type("locations") == ConceptType.SETTING
        assert resolve_concept_type("styles") == ConceptType.STYLE
        assert resolve_concept_type("actions") == ConceptType.MOTION

    def test_all_aliases_map_to_valid_type(self) -> None:
        """Every alias in TYPE_ALIASES maps to a valid ConceptType."""
        for alias, concept_type in TYPE_ALIASES.items():
            assert isinstance(concept_type, ConceptType), \
                f"Alias '{alias}' maps to non-ConceptType: {concept_type}"


class TestConceptReference:
    """Tests for ConceptReference dataclass."""

    def test_creation(self) -> None:
        ref = ConceptReference(
            name="holly",
            concept_type=ConceptType.CHARACTER,
            image_path=Path("/concepts/character/holly.jpg"),
            folder_name="character",
        )
        assert ref.name == "holly"
        assert ref.concept_type == ConceptType.CHARACTER
        assert ref.image_path == Path("/concepts/character/holly.jpg")
        assert ref.folder_name == "character"

    def test_none_type(self) -> None:
        """ConceptReference works with None concept_type."""
        ref = ConceptReference(
            name="mystery",
            concept_type=None,
            image_path=Path("/concepts/unknown/mystery.png"),
            folder_name="unknown",
        )
        assert ref.concept_type is None

    def test_frozen(self) -> None:
        """ConceptReference is immutable."""
        ref = ConceptReference(
            name="holly",
            concept_type=ConceptType.CHARACTER,
            image_path=Path("/concepts/character/holly.jpg"),
            folder_name="character",
        )
        with pytest.raises(AttributeError):
            ref.name = "other"


class TestClipMatch:
    """Tests for ClipMatch dataclass."""

    def test_creation(self) -> None:
        ref = ConceptReference("holly", ConceptType.CHARACTER, Path("x.jpg"), "character")
        match = ClipMatch(concept=ref, similarity=0.85, best_frame_index=2)
        assert match.similarity == 0.85
        assert match.best_frame_index == 2

    def test_frozen(self) -> None:
        ref = ConceptReference("holly", ConceptType.CHARACTER, Path("x.jpg"), "character")
        match = ClipMatch(concept=ref, similarity=0.85, best_frame_index=2)
        with pytest.raises(AttributeError):
            match.similarity = 0.5


class TestClipTriage:
    """Tests for ClipTriage dataclass."""

    def _make_ref(self, name: str, ctype: ConceptType) -> ConceptReference:
        return ConceptReference(name, ctype, Path(f"{name}.jpg"), ctype.value)

    def test_empty_matches(self) -> None:
        ct = ClipTriage(clip_path=Path("clip.mp4"))
        assert ct.best_match is None
        assert not ct.is_matched
        assert ct.concept_types == []

    def test_single_match(self) -> None:
        ref = self._make_ref("holly", ConceptType.CHARACTER)
        match = ClipMatch(concept=ref, similarity=0.8, best_frame_index=1)
        ct = ClipTriage(clip_path=Path("clip.mp4"), matches=[match])
        assert ct.is_matched
        assert ct.best_match == match
        assert ct.concept_types == [ConceptType.CHARACTER]

    def test_multiple_matches_best_is_first(self) -> None:
        """best_match returns the first match (should be sorted highest-first)."""
        ref1 = self._make_ref("holly", ConceptType.CHARACTER)
        ref2 = self._make_ref("tiffanys", ConceptType.SETTING)
        m1 = ClipMatch(concept=ref1, similarity=0.9, best_frame_index=0)
        m2 = ClipMatch(concept=ref2, similarity=0.75, best_frame_index=2)
        ct = ClipTriage(clip_path=Path("clip.mp4"), matches=[m1, m2])
        assert ct.best_match == m1

    def test_concept_types_unique(self) -> None:
        """concept_types returns unique types only."""
        ref1 = self._make_ref("holly", ConceptType.CHARACTER)
        ref2 = self._make_ref("paul", ConceptType.CHARACTER)
        m1 = ClipMatch(concept=ref1, similarity=0.9, best_frame_index=0)
        m2 = ClipMatch(concept=ref2, similarity=0.8, best_frame_index=1)
        ct = ClipTriage(clip_path=Path("clip.mp4"), matches=[m1, m2])
        assert ct.concept_types == [ConceptType.CHARACTER]

    def test_multiple_types(self) -> None:
        ref1 = self._make_ref("holly", ConceptType.CHARACTER)
        ref2 = self._make_ref("tiffanys", ConceptType.SETTING)
        m1 = ClipMatch(concept=ref1, similarity=0.9, best_frame_index=0)
        m2 = ClipMatch(concept=ref2, similarity=0.8, best_frame_index=1)
        ct = ClipTriage(clip_path=Path("clip.mp4"), matches=[m1, m2])
        assert ConceptType.CHARACTER in ct.concept_types
        assert ConceptType.SETTING in ct.concept_types


class TestTriageReport:
    """Tests for TriageReport dataclass."""

    def _make_clip(self, matched: bool) -> ClipTriage:
        ct = ClipTriage(clip_path=Path(f"clip_{id(matched)}.mp4"))
        if matched:
            ref = ConceptReference("x", ConceptType.CHARACTER, Path("x.jpg"), "char")
            ct.matches = [ClipMatch(concept=ref, similarity=0.8, best_frame_index=0)]
        return ct

    def test_empty_report(self) -> None:
        report = TriageReport()
        assert report.total == 0
        assert report.matched_count == 0
        assert report.unmatched_count == 0

    def test_all_matched(self) -> None:
        clips = [self._make_clip(True) for _ in range(3)]
        report = TriageReport(clips=clips)
        assert report.total == 3
        assert report.matched_count == 3
        assert report.unmatched_count == 0

    def test_mixed(self) -> None:
        clips = [self._make_clip(True), self._make_clip(False), self._make_clip(True)]
        report = TriageReport(clips=clips)
        assert report.total == 3
        assert report.matched_count == 2
        assert report.unmatched_count == 1

    def test_default_threshold(self) -> None:
        report = TriageReport()
        assert report.threshold == 0.70

    def test_default_model(self) -> None:
        report = TriageReport()
        assert "clip" in report.model_name.lower()
