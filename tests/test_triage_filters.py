"""Tests for triage content filters (text overlay detection).

Tests the zero-shot CLIP text-image matching used to detect
title cards, credits, and text overlays. Uses mock embeddings
to avoid requiring torch/CLIP.
"""

import numpy as np
import pytest

from klippbok.triage.filters import (
    DEFAULT_TEXT_OVERLAY_THRESHOLD,
    TEXT_OVERLAY_PROMPTS,
    detect_text_overlays,
)


class TestTextOverlayPrompts:
    """Tests for the text overlay prompt definitions."""

    def test_prompts_not_empty(self) -> None:
        """We have a meaningful set of detection prompts."""
        assert len(TEXT_OVERLAY_PROMPTS) >= 5

    def test_prompts_are_strings(self) -> None:
        """All prompts are non-empty strings."""
        for prompt in TEXT_OVERLAY_PROMPTS:
            assert isinstance(prompt, str)
            assert len(prompt) > 5

    def test_default_threshold_reasonable(self) -> None:
        """Threshold is in a sensible range for CLIP text-image similarity."""
        # CLIP text-image scores are typically 0.15-0.35 for good matches
        assert 0.20 <= DEFAULT_TEXT_OVERLAY_THRESHOLD <= 0.40


class TestDetectTextOverlays:
    """Tests for detect_text_overlays() with mock embeddings."""

    def _make_prompt_embeddings(self, n: int = 9) -> list[np.ndarray]:
        """Create normalized random embeddings to simulate prompt embeddings."""
        rng = np.random.RandomState(42)
        embeddings = []
        for _ in range(n):
            emb = rng.randn(512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            embeddings.append(emb)
        return embeddings

    def test_no_frames_returns_false(self) -> None:
        """Empty frame list is not a text overlay."""
        # embedder not used when frame_embeddings is empty
        is_text, score = detect_text_overlays(
            None,  # type: ignore
            [],
            _prompt_embeddings=[],
        )
        assert is_text is False
        assert score == 0.0

    def test_high_similarity_detected(self) -> None:
        """Frames highly similar to text prompts are flagged."""
        # Create a frame embedding that's identical to a prompt embedding
        prompt_embs = self._make_prompt_embeddings()
        # Frame = exact copy of first prompt (similarity = 1.0)
        frame_embs = [prompt_embs[0].copy()]

        is_text, score = detect_text_overlays(
            None,  # type: ignore — not called with _prompt_embeddings
            frame_embs,
            threshold=0.27,
            _prompt_embeddings=prompt_embs,
        )
        assert is_text is True
        assert score > 0.99

    def test_low_similarity_not_detected(self) -> None:
        """Frames dissimilar to text prompts are not flagged."""
        prompt_embs = self._make_prompt_embeddings()
        # Create a frame embedding orthogonal to all prompts
        rng = np.random.RandomState(999)
        frame = rng.randn(512).astype(np.float32)
        frame /= np.linalg.norm(frame)
        frame_embs = [frame]

        is_text, score = detect_text_overlays(
            None,  # type: ignore
            frame_embs,
            threshold=0.27,
            _prompt_embeddings=prompt_embs,
        )
        # Random 512-dim vectors have near-zero dot product
        assert score < 0.15
        assert is_text is False

    def test_threshold_boundary(self) -> None:
        """Score exactly at threshold is detected."""
        # Create controlled embeddings with known similarity
        prompt = np.zeros(512, dtype=np.float32)
        prompt[0] = 1.0

        # Frame with known cosine similarity of 0.3
        frame = np.zeros(512, dtype=np.float32)
        frame[0] = 0.3
        frame[1] = np.sqrt(1 - 0.09)  # normalize to unit
        frame /= np.linalg.norm(frame)

        actual_sim = float(np.dot(frame, prompt))

        is_text, score = detect_text_overlays(
            None,  # type: ignore
            [frame],
            threshold=actual_sim,  # exact threshold
            _prompt_embeddings=[prompt],
        )
        assert is_text is True
        assert abs(score - actual_sim) < 0.001

    def test_multiple_frames_takes_max(self) -> None:
        """Detection uses the best score across all frames."""
        prompt_embs = self._make_prompt_embeddings(3)

        # First frame: low similarity. Second frame: high (copy of prompt).
        rng = np.random.RandomState(123)
        low_frame = rng.randn(512).astype(np.float32)
        low_frame /= np.linalg.norm(low_frame)
        high_frame = prompt_embs[0].copy()

        is_text, score = detect_text_overlays(
            None,  # type: ignore
            [low_frame, high_frame],
            threshold=0.27,
            _prompt_embeddings=prompt_embs,
        )
        assert is_text is True
        assert score > 0.99  # high_frame matched exactly


class TestClipTriageTextOverlay:
    """Tests for text overlay fields on ClipTriage model."""

    def test_default_no_text_overlay(self) -> None:
        """ClipTriage defaults to no text overlay."""
        from pathlib import Path
        from klippbok.triage.models import ClipTriage

        triage = ClipTriage(clip_path=Path("/fake/clip.mp4"))
        assert triage.has_text_overlay is False
        assert triage.text_overlay_score == 0.0

    def test_text_overlay_count_on_report(self) -> None:
        """TriageReport counts text overlay clips."""
        from pathlib import Path
        from klippbok.triage.models import ClipTriage, TriageReport

        clip1 = ClipTriage(clip_path=Path("/fake/1.mp4"), has_text_overlay=True)
        clip2 = ClipTriage(clip_path=Path("/fake/2.mp4"), has_text_overlay=False)
        clip3 = ClipTriage(clip_path=Path("/fake/3.mp4"), has_text_overlay=True)

        report = TriageReport(clips=[clip1, clip2, clip3])
        assert report.text_overlay_count == 2


class TestOrganizeTextOverlay:
    """Tests for text overlay handling in organize_clips."""

    def test_text_overlay_clips_go_to_text_overlay_folder(self, tmp_path) -> None:
        """Clips flagged as text overlays are organized into text_overlay/."""
        from pathlib import Path
        from klippbok.triage.models import ClipTriage, TriageReport
        from klippbok.triage.triage import organize_clips

        # Create a fake clip file
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        clip_path = clips_dir / "title_card.mp4"
        clip_path.write_bytes(b"fake")

        clip = ClipTriage(
            clip_path=clip_path,
            has_text_overlay=True,
            text_overlay_score=0.35,
        )

        output_dir = tmp_path / "organized"
        result = organize_clips(
            TriageReport(clips=[clip]),
            output_dir,
            copy=True,
        )

        assert "text_overlay" in result
        assert (output_dir / "text_overlay" / "title_card.mp4").exists()

    def test_text_overlay_overrides_subject_match(self, tmp_path) -> None:
        """Text overlay clips go to text_overlay/ even if they matched a concept.

        A title card might show a character's name or face — the text
        overlay flag should take priority.
        """
        from pathlib import Path
        from klippbok.triage.models import (
            ClipMatch, ClipTriage, ConceptReference, ConceptType, TriageReport,
        )
        from klippbok.triage.triage import organize_clips

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        clip_path = clips_dir / "credits.mp4"
        clip_path.write_bytes(b"fake")

        ref = ConceptReference(
            name="holly", concept_type=ConceptType.CHARACTER,
            image_path=Path("/fake/holly.jpg"), folder_name="character",
        )

        # Clip matches holly BUT also flagged as text overlay
        clip = ClipTriage(
            clip_path=clip_path,
            has_text_overlay=True,
            text_overlay_score=0.32,
        )
        clip.matches = [ClipMatch(concept=ref, similarity=0.80, best_frame_index=0)]

        output_dir = tmp_path / "organized"
        result = organize_clips(
            TriageReport(clips=[clip], concepts=[ref]),
            output_dir,
            copy=True,
        )

        # Should go to text_overlay, not holly
        assert "text_overlay" in result
        assert "holly" not in result
