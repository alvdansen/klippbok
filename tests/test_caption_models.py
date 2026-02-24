"""Tests for klippbok.caption.models — pure Python, no API calls.

Tests the Pydantic models for caption configuration and results.
"""

from pathlib import Path

import pytest

from klippbok.caption.models import AuditResult, CaptionConfig, CaptionResult


class TestCaptionConfig:
    """Tests for CaptionConfig model."""

    def test_defaults(self) -> None:
        """Default config uses Gemini with sensible settings."""
        config = CaptionConfig()
        assert config.provider == "gemini"
        assert config.use_case is None
        assert config.anchor_word is None
        assert config.overwrite is False
        assert config.timeout == 120
        assert config.max_retries == 5
        assert config.between_request_delay == 10.0
        assert config.audit_mode == "report_only"

    def test_gemini_provider(self) -> None:
        config = CaptionConfig(provider="gemini")
        assert config.provider == "gemini"

    def test_replicate_provider(self) -> None:
        config = CaptionConfig(provider="replicate")
        assert config.provider == "replicate"

    def test_use_cases(self) -> None:
        for uc in ["character", "style", "motion", "object"]:
            config = CaptionConfig(use_case=uc)
            assert config.use_case == uc

    def test_anchor_word(self) -> None:
        config = CaptionConfig(anchor_word="Luna")
        assert config.anchor_word == "Luna"

    def test_overwrite(self) -> None:
        config = CaptionConfig(overwrite=True)
        assert config.overwrite is True

    def test_audit_modes(self) -> None:
        for mode in ["report_only", "save_audit"]:
            config = CaptionConfig(audit_mode=mode)
            assert config.audit_mode == mode

    def test_custom_delay(self) -> None:
        config = CaptionConfig(between_request_delay=2.0)
        assert config.between_request_delay == 2.0

    def test_frozen(self) -> None:
        config = CaptionConfig()
        with pytest.raises(Exception):
            config.provider = "replicate"  # type: ignore[misc]


class TestCaptionResult:
    """Tests for CaptionResult model."""

    def test_success(self) -> None:
        result = CaptionResult(
            path=Path("/test/clip.mp4"),
            caption="A person walks through a forest",
            provider="gemini",
            duration=3.5,
            success=True,
        )
        assert result.success is True
        assert result.caption == "A person walks through a forest"
        assert result.error == ""
        assert result.skipped is False

    def test_failure(self) -> None:
        result = CaptionResult(
            path=Path("/test/clip.mp4"),
            provider="gemini",
            duration=1.2,
            success=False,
            error="API rate limit exceeded",
        )
        assert result.success is False
        assert result.error == "API rate limit exceeded"
        assert result.caption == ""

    def test_skipped(self) -> None:
        result = CaptionResult(
            path=Path("/test/clip.mp4"),
            skipped=True,
            provider="gemini",
        )
        assert result.skipped is True

    def test_defaults(self) -> None:
        result = CaptionResult(path=Path("/test/clip.mp4"))
        assert result.caption == ""
        assert result.provider == ""
        assert result.duration == 0.0
        assert result.success is True
        assert result.error == ""
        assert result.skipped is False


class TestAuditResult:
    """Tests for AuditResult model."""

    def test_keep(self) -> None:
        result = AuditResult(
            path=Path("/test/clip.mp4"),
            existing_caption="Luna stands on a rooftop",
            vlm_caption="Luna is standing on a rooftop looking out",
            recommendation="keep",
            provider="gemini",
        )
        assert result.recommendation == "keep"

    def test_review(self) -> None:
        result = AuditResult(
            path=Path("/test/clip.mp4"),
            existing_caption="A girl walks",
            vlm_caption="Luna runs through an alley at night",
            recommendation="review",
            provider="gemini",
        )
        assert result.recommendation == "review"

    def test_default_recommendation(self) -> None:
        result = AuditResult(
            path=Path("/test/clip.mp4"),
            existing_caption="test",
            vlm_caption="test2",
        )
        assert result.recommendation == "review"
