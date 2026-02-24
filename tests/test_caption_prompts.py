"""Tests for klippbok.caption.prompts — pure Python, no API calls.

Tests prompt template selection, anchor word injection, and secondary anchors.
"""

import pytest

from klippbok.caption.prompts import (
    format_prompt,
    get_image_prompt,
    get_video_prompt,
)


class TestGetVideoPrompt:
    """Tests for video prompt selection."""

    def test_general(self) -> None:
        """None use_case returns a general prompt."""
        prompt = get_video_prompt(None)
        assert "caption" in prompt.lower()
        assert "prompt style" in prompt.lower()

    def test_character(self) -> None:
        prompt = get_video_prompt("character")
        assert "do not describe" in prompt.lower()
        assert "appearance" in prompt.lower()

    def test_style(self) -> None:
        prompt = get_video_prompt("style")
        assert "do not describe" in prompt.lower()
        assert "style" in prompt.lower()

    def test_motion(self) -> None:
        prompt = get_video_prompt("motion")
        assert "move" in prompt.lower()
        assert "do not describe" in prompt.lower()

    def test_object(self) -> None:
        prompt = get_video_prompt("object")
        assert "do not describe" in prompt.lower()

    def test_invalid(self) -> None:
        """Invalid use_case raises ValueError."""
        with pytest.raises(ValueError, match="Unknown use_case"):
            get_video_prompt("invalid_use_case")


class TestAnchorWordInjection:
    """Tests that anchor words are woven into prompts naturally."""

    def test_character_anchor_uses_name(self) -> None:
        """Anchor word appears as the subject's name in the prompt."""
        prompt = get_video_prompt("character", anchor_word="luna")
        assert "luna" in prompt.lower()
        # Should tell VLM to use it naturally
        assert "naturally" in prompt.lower()

    def test_character_no_anchor_uses_generic(self) -> None:
        """Without anchor, prompt uses a generic subject reference."""
        prompt = get_video_prompt("character")
        assert "the subject" in prompt.lower()
        # No anchor instruction line
        assert "name is" not in prompt.lower()

    def test_object_anchor(self) -> None:
        """Object use case also accepts anchor words."""
        prompt = get_video_prompt("object", anchor_word="shimmer")
        assert "shimmer" in prompt.lower()

    def test_style_anchor(self) -> None:
        """Style use case accepts anchor word."""
        prompt = get_video_prompt("style", anchor_word="vintage")
        assert "vintage" in prompt.lower()

    def test_general_no_placeholders_leak(self) -> None:
        """General prompt doesn't have unfilled placeholders."""
        prompt = get_video_prompt(None)
        assert "{" not in prompt
        assert "}" not in prompt

    def test_all_prompts_no_placeholders_leak(self) -> None:
        """No unfilled {placeholders} in any prompt, with or without anchor."""
        for use_case in [None, "character", "style", "motion", "object"]:
            prompt = get_video_prompt(use_case)
            assert "{" not in prompt, f"{use_case} without anchor has unfilled placeholder"
            prompt = get_video_prompt(use_case, anchor_word="test")
            assert "{" not in prompt, f"{use_case} with anchor has unfilled placeholder"


class TestSecondaryAnchors:
    """Tests for secondary anchor tags in prompts."""

    def test_secondary_anchors_included(self) -> None:
        """Secondary anchors appear in the prompt."""
        prompt = get_video_prompt(
            "character", anchor_word="luna",
            secondary_anchors=["vintage", "retro"],
        )
        assert "vintage" in prompt.lower()
        assert "retro" in prompt.lower()

    def test_secondary_anchors_conditional(self) -> None:
        """Prompt tells VLM to only use them if visible."""
        prompt = get_video_prompt(
            "character", anchor_word="luna",
            secondary_anchors=["vintage"],
        )
        assert "only" in prompt.lower()

    def test_no_secondary_anchors(self) -> None:
        """Without secondary anchors, no extra tag instruction appears."""
        prompt = get_video_prompt("character", anchor_word="luna")
        assert "may be relevant" not in prompt.lower()

    def test_secondary_anchors_without_primary(self) -> None:
        """Secondary anchors work even without a primary anchor word."""
        prompt = get_video_prompt(
            "character",
            secondary_anchors=["vintage", "retro"],
        )
        assert "vintage" in prompt.lower()
        assert "retro" in prompt.lower()

    def test_secondary_anchors_on_all_use_cases(self) -> None:
        """All use cases support secondary anchors."""
        for use_case in ["character", "style", "motion", "object"]:
            prompt = get_video_prompt(
                use_case, anchor_word="test",
                secondary_anchors=["tag1"],
            )
            assert "tag1" in prompt.lower(), f"{use_case} missing secondary anchor"


class TestPromptContent:
    """Tests that prompts contain the right instructions."""

    def test_character_omits_appearance(self) -> None:
        """Character prompt tells VLM to skip appearance."""
        prompt = get_video_prompt("character")
        assert "do not describe" in prompt.lower()
        assert "appearance" in prompt.lower()

    def test_style_omits_aesthetics(self) -> None:
        """Style prompt tells VLM to skip style descriptors."""
        prompt = get_video_prompt("style")
        assert "do not describe" in prompt.lower()
        assert "style" in prompt.lower()

    def test_motion_focuses_on_movement(self) -> None:
        """Motion prompt emphasizes movement description."""
        prompt = get_video_prompt("motion")
        assert "move" in prompt.lower()
        assert "do not describe" in prompt.lower()

    def test_all_prompts_request_brevity(self) -> None:
        """All prompts ask for short captions."""
        for use_case in [None, "character", "style", "motion", "object"]:
            prompt = get_video_prompt(use_case)
            assert "brief" in prompt.lower() or "short" in prompt.lower()


class TestGetImagePrompt:
    """Tests for image prompt selection."""

    def test_general(self) -> None:
        """Returns general image prompt."""
        prompt = get_image_prompt(None)
        assert "image" in prompt.lower()

    def test_unknown_use_case_falls_back(self) -> None:
        """Unknown use_case falls back to general."""
        prompt = get_image_prompt("unknown")
        assert "image" in prompt.lower()

    def test_character_image_with_anchor(self) -> None:
        """Character image prompt uses anchor word as name."""
        prompt = get_image_prompt("character", anchor_word="luna")
        assert "luna" in prompt.lower()
        assert "do not describe" in prompt.lower()

    def test_style_image_prompt(self) -> None:
        prompt = get_image_prompt("style")
        assert "do not describe" in prompt.lower()
        assert "style" in prompt.lower()

    def test_no_placeholders_in_image_prompts(self) -> None:
        """No unfilled placeholders in any image prompt."""
        for use_case in [None, "character", "style", "motion", "object"]:
            prompt = get_image_prompt(use_case)
            assert "{" not in prompt
            prompt = get_image_prompt(use_case, anchor_word="test")
            assert "{" not in prompt


class TestFormatPrompt:
    """Tests for format_prompt() — safe template variable substitution."""

    def test_basic_substitution(self) -> None:
        """Substitutes a known variable."""
        result = format_prompt("Describe {anchor_word} in detail", anchor_word="Luna")
        assert result == "Describe Luna in detail"

    def test_multiple_variables(self) -> None:
        """Substitutes multiple variables."""
        result = format_prompt(
            "{name} is in {place}",
            name="Luna",
            place="the plaza",
        )
        assert result == "Luna is in the plaza"

    def test_no_variables(self) -> None:
        """Prompt with no placeholders passes through unchanged."""
        result = format_prompt("No variables here")
        assert result == "No variables here"

    def test_missing_variable_preserved(self) -> None:
        """Missing variables are left as-is (no KeyError)."""
        result = format_prompt("{missing} stays", other="ignored")
        assert result == "{missing} stays"

    def test_extra_variables_ignored(self) -> None:
        """Extra variables that don't appear in the prompt are ignored."""
        result = format_prompt("Hello world", unused="value")
        assert result == "Hello world"


class TestCustomPrompt:
    """Tests for custom_prompt field on CaptionConfig."""

    def test_custom_prompt_field(self) -> None:
        """CaptionConfig accepts custom_prompt."""
        from klippbok.caption.models import CaptionConfig
        config = CaptionConfig(custom_prompt="My custom instructions here")
        assert config.custom_prompt == "My custom instructions here"

    def test_custom_prompt_default_none(self) -> None:
        """custom_prompt defaults to None."""
        from klippbok.caption.models import CaptionConfig
        config = CaptionConfig()
        assert config.custom_prompt is None

    def test_custom_prompt_overrides_in_captioner(self) -> None:
        """custom_prompt takes priority over use_case in caption_clips."""
        from pathlib import Path

        from klippbok.caption.base import VLMBackend
        from klippbok.caption.captioner import caption_clips
        from klippbok.caption.models import CaptionConfig

        class PromptCapture(VLMBackend):
            """Records the prompt it receives."""
            def __init__(self):
                self.received_prompts: list[str] = []
            def caption_video(self, path: Path, prompt: str) -> str:
                self.received_prompts.append(prompt)
                return "test caption"
            def caption_image(self, path: Path, prompt: str) -> str:
                return "test"

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "clip.mp4").write_bytes(b"\x00")

            capture = PromptCapture()

            import klippbok.caption.captioner as cap_mod
            original = cap_mod._create_backend
            cap_mod._create_backend = lambda config: capture

            try:
                config = CaptionConfig(
                    provider="gemini",
                    use_case="character",
                    custom_prompt="My totally custom prompt",
                    between_request_delay=0,
                )
                caption_clips(tmp, config)
            finally:
                cap_mod._create_backend = original

            assert capture.received_prompts == ["My totally custom prompt"]


class TestSecondaryAnchorsConfig:
    """Tests for secondary_anchors on CaptionConfig."""

    def test_secondary_anchors_field(self) -> None:
        from klippbok.caption.models import CaptionConfig
        config = CaptionConfig(secondary_anchors=["vintage", "retro"])
        assert config.secondary_anchors == ["vintage", "retro"]

    def test_secondary_anchors_default_none(self) -> None:
        from klippbok.caption.models import CaptionConfig
        config = CaptionConfig()
        assert config.secondary_anchors is None
