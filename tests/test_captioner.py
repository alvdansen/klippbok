"""Tests for klippbok.caption.captioner — orchestrator tests with mock backend.

Tests the batch captioning and auditing orchestration logic using a mock
VLM backend. Zero real API calls — all backend behavior is simulated.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from klippbok.caption.base import VLMBackend
from klippbok.caption.captioner import (
    VIDEO_EXTENSIONS,
    _create_backend,
    _find_video_files,
    _prepend_anchor,
    caption_clips,
    audit_captions,
)
from klippbok.caption.models import CaptionConfig


# ---------------------------------------------------------------------------
# Mock backend — implements VLMBackend ABC for testing
# ---------------------------------------------------------------------------

class MockBackend(VLMBackend):
    """Fake VLM backend that returns predictable captions.

    Records all calls for assertion. Can be configured to fail on
    specific files.
    """

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.calls: list[dict] = []
        self.fail_on = fail_on or set()

    def caption_video(self, path: Path, prompt: str) -> str:
        self.calls.append({"type": "video", "path": path, "prompt": prompt})
        if path.name in self.fail_on:
            raise RuntimeError(f"Mock failure on {path.name}")
        return f"Caption for {path.stem}"

    def caption_image(self, path: Path, prompt: str) -> str:
        self.calls.append({"type": "image", "path": path, "prompt": prompt})
        if path.name in self.fail_on:
            raise RuntimeError(f"Mock failure on {path.name}")
        return f"Image caption for {path.stem}"


# ---------------------------------------------------------------------------
# _prepend_anchor tests
# ---------------------------------------------------------------------------

class TestPrependAnchor:
    """Tests for _prepend_anchor() — pure function, no I/O."""

    def test_prepend_when_absent(self) -> None:
        """Adds anchor word with comma when not present."""
        result = _prepend_anchor("A girl walks through a forest", "Luna")
        assert result == "Luna, a girl walks through a forest"

    def test_no_prepend_when_present(self) -> None:
        """Does not duplicate anchor word if already at start."""
        result = _prepend_anchor("Luna is walking through a forest", "Luna")
        assert result == "Luna is walking through a forest"

    def test_case_insensitive(self) -> None:
        """Anchor detection is case-insensitive."""
        result = _prepend_anchor("luna stands on a rooftop", "Luna")
        assert result == "luna stands on a rooftop"

    def test_lowercases_first_char(self) -> None:
        """First character of existing caption is lowered for natural flow."""
        result = _prepend_anchor("The camera pans left", "Vi")
        assert result == "Vi, the camera pans left"

    def test_empty_caption(self) -> None:
        """Handles empty caption gracefully."""
        result = _prepend_anchor("", "Luna")
        assert result == "Luna, "

    def test_anchor_is_full_caption(self) -> None:
        """Caption that IS the anchor word returns as-is."""
        result = _prepend_anchor("Luna", "Luna")
        assert result == "Luna"

    def test_lowercase_start(self) -> None:
        """Caption starting with lowercase stays lowercase."""
        result = _prepend_anchor("walking through rain", "Vi")
        assert result == "Vi, walking through rain"


# ---------------------------------------------------------------------------
# _find_video_files tests
# ---------------------------------------------------------------------------

class TestFindVideoFiles:
    """Tests for _find_video_files() — filesystem scanning."""

    def test_finds_video_extensions(self, tmp_path: Path) -> None:
        """Finds all recognized video file extensions."""
        for ext in VIDEO_EXTENSIONS:
            (tmp_path / f"clip{ext}").write_bytes(b"\x00")

        found = _find_video_files(tmp_path)
        assert len(found) == len(VIDEO_EXTENSIONS)

    def test_ignores_non_video(self, tmp_path: Path) -> None:
        """Ignores non-video files."""
        (tmp_path / "clip.mp4").write_bytes(b"\x00")
        (tmp_path / "caption.txt").write_text("test")
        (tmp_path / "readme.md").write_text("test")
        (tmp_path / "data.json").write_text("{}")

        found = _find_video_files(tmp_path)
        assert len(found) == 1
        assert found[0].name == "clip.mp4"

    def test_sorted_by_name(self, tmp_path: Path) -> None:
        """Results are sorted alphabetically."""
        (tmp_path / "c_clip.mp4").write_bytes(b"\x00")
        (tmp_path / "a_clip.mp4").write_bytes(b"\x00")
        (tmp_path / "b_clip.mp4").write_bytes(b"\x00")

        found = _find_video_files(tmp_path)
        names = [f.name for f in found]
        assert names == ["a_clip.mp4", "b_clip.mp4", "c_clip.mp4"]

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Returns empty list for empty directory."""
        found = _find_video_files(tmp_path)
        assert found == []

    def test_ignores_directories(self, tmp_path: Path) -> None:
        """Ignores subdirectories even if they have video-like names."""
        (tmp_path / "clip.mp4").mkdir()
        (tmp_path / "real.mp4").write_bytes(b"\x00")

        found = _find_video_files(tmp_path)
        assert len(found) == 1
        assert found[0].name == "real.mp4"

    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        """Finds video files regardless of extension case."""
        (tmp_path / "clip.MP4").write_bytes(b"\x00")
        (tmp_path / "clip2.Mp4").write_bytes(b"\x00")

        found = _find_video_files(tmp_path)
        assert len(found) == 2


# ---------------------------------------------------------------------------
# _create_backend tests
# ---------------------------------------------------------------------------

class TestCreateBackend:
    """Tests for _create_backend() — factory function."""

    def test_gemini_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Creates GeminiBackend for 'gemini' provider."""
        # Mock the google.genai import inside GeminiBackend.__init__
        import sys
        mock_genai = MagicMock()
        mock_genai.Client.return_value = MagicMock()
        google_mod = MagicMock()
        google_mod.genai = mock_genai
        monkeypatch.setitem(sys.modules, "google", google_mod)
        monkeypatch.setitem(sys.modules, "google.genai", mock_genai)

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = CaptionConfig(provider="gemini")
        backend = _create_backend(config)

        from klippbok.caption.gemini import GeminiBackend
        assert isinstance(backend, GeminiBackend)

    def test_replicate_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Creates ReplicateBackend for 'replicate' provider."""
        monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token")
        config = CaptionConfig(provider="replicate")
        backend = _create_backend(config)

        from klippbok.caption.replicate import ReplicateBackend
        assert isinstance(backend, ReplicateBackend)

    def test_unknown_provider(self) -> None:
        """Raises ValueError for unknown provider."""
        # We need to bypass Pydantic validation to pass an invalid provider
        # for testing the _create_backend guard
        config = CaptionConfig.__new__(CaptionConfig)
        object.__setattr__(config, "provider", "unknown")
        object.__setattr__(config, "api_key", None)
        with pytest.raises(ValueError, match="Unknown caption provider"):
            _create_backend(config)


# ---------------------------------------------------------------------------
# caption_clips tests (with MockBackend)
# ---------------------------------------------------------------------------

class TestCaptionClips:
    """Tests for caption_clips() — full orchestration with MockBackend."""

    def _make_clips(self, tmp_path: Path, count: int = 3) -> list[Path]:
        """Helper: create fake video files for testing."""
        clips = []
        for i in range(count):
            clip = tmp_path / f"clip_{i:02d}.mp4"
            clip.write_bytes(b"\x00" * 100)
            clips.append(clip)
        return clips

    def test_basic_captioning(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Captions all clips and writes .txt files."""
        self._make_clips(tmp_path, 3)
        mock = MockBackend()

        # Patch _create_backend to return our mock
        monkeypatch.setattr(
            "klippbok.caption.captioner._create_backend",
            lambda config: mock,
        )

        config = CaptionConfig(provider="gemini", between_request_delay=0)
        results = caption_clips(tmp_path, config)

        assert len(results) == 3
        assert all(r.success for r in results)
        # Check .txt files were written
        for i in range(3):
            txt = tmp_path / f"clip_{i:02d}.txt"
            assert txt.exists()
            assert txt.read_text(encoding="utf-8") == f"Caption for clip_{i:02d}"

    def test_skip_existing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Skips clips that already have .txt captions when overwrite=False."""
        self._make_clips(tmp_path, 3)
        # Pre-create caption for clip_01
        (tmp_path / "clip_01.txt").write_text("existing caption")

        mock = MockBackend()
        monkeypatch.setattr(
            "klippbok.caption.captioner._create_backend",
            lambda config: mock,
        )

        config = CaptionConfig(provider="gemini", overwrite=False, between_request_delay=0)
        results = caption_clips(tmp_path, config)

        assert len(results) == 3
        skipped = [r for r in results if r.skipped]
        assert len(skipped) == 1
        # Backend was only called twice (skipped clip_01)
        assert len(mock.calls) == 2
        # Original caption preserved
        assert (tmp_path / "clip_01.txt").read_text() == "existing caption"

    def test_overwrite_existing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Overwrites existing captions when overwrite=True."""
        self._make_clips(tmp_path, 2)
        (tmp_path / "clip_00.txt").write_text("old caption")

        mock = MockBackend()
        monkeypatch.setattr(
            "klippbok.caption.captioner._create_backend",
            lambda config: mock,
        )

        config = CaptionConfig(provider="gemini", overwrite=True, between_request_delay=0)
        results = caption_clips(tmp_path, config)

        assert len(results) == 2
        assert all(r.success for r in results)
        # Caption was overwritten
        assert (tmp_path / "clip_00.txt").read_text(encoding="utf-8") == "Caption for clip_00"

    def test_anchor_word_in_prompt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Anchor word is baked into the prompt sent to the VLM."""
        self._make_clips(tmp_path, 1)

        received_prompts: list[str] = []

        class PromptCapture(MockBackend):
            def caption_video(self, path: Path, prompt: str) -> str:
                received_prompts.append(prompt)
                return super().caption_video(path, prompt)

        monkeypatch.setattr(
            "klippbok.caption.captioner._create_backend",
            lambda config: PromptCapture(),
        )

        config = CaptionConfig(
            provider="gemini",
            use_case="character",
            anchor_word="Luna",
            between_request_delay=0,
        )
        caption_clips(tmp_path, config)

        assert len(received_prompts) == 1
        # The prompt itself should contain the anchor word as a name
        assert "Luna" in received_prompts[0]
        assert "naturally" in received_prompts[0].lower()

    def test_backend_failure_continues_batch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failing clip doesn't stop the rest of the batch."""
        self._make_clips(tmp_path, 3)

        # Fail on the middle clip
        mock = MockBackend(fail_on={"clip_01.mp4"})
        monkeypatch.setattr(
            "klippbok.caption.captioner._create_backend",
            lambda config: mock,
        )

        config = CaptionConfig(provider="gemini", between_request_delay=0)
        results = caption_clips(tmp_path, config)

        assert len(results) == 3
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success and not r.skipped]
        assert len(successes) == 2
        assert len(failures) == 1
        assert "Mock failure" in failures[0].error

    def test_empty_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns empty list for directory with no video files."""
        mock = MockBackend()
        monkeypatch.setattr(
            "klippbok.caption.captioner._create_backend",
            lambda config: mock,
        )

        config = CaptionConfig(provider="gemini", between_request_delay=0)
        results = caption_clips(tmp_path, config)

        assert results == []
        assert len(mock.calls) == 0

    def test_nonexistent_directory(self) -> None:
        """Raises FileNotFoundError for missing directory."""
        config = CaptionConfig(provider="gemini")
        with pytest.raises(FileNotFoundError):
            caption_clips("/nonexistent/path", config)

    def test_result_records_provider(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Results record which provider was used."""
        self._make_clips(tmp_path, 1)
        mock = MockBackend()
        monkeypatch.setattr(
            "klippbok.caption.captioner._create_backend",
            lambda config: mock,
        )

        config = CaptionConfig(provider="gemini", between_request_delay=0)
        results = caption_clips(tmp_path, config)

        assert results[0].provider == "gemini"

    def test_result_records_duration(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Results record how long captioning took."""
        self._make_clips(tmp_path, 1)
        mock = MockBackend()
        monkeypatch.setattr(
            "klippbok.caption.captioner._create_backend",
            lambda config: mock,
        )

        config = CaptionConfig(provider="gemini", between_request_delay=0)
        results = caption_clips(tmp_path, config)

        assert results[0].duration >= 0.0


# ---------------------------------------------------------------------------
# audit_captions tests
# ---------------------------------------------------------------------------

class TestAuditCaptions:
    """Tests for audit_captions() — audit flow with MockBackend."""

    def _make_captioned_clips(self, tmp_path: Path, count: int = 3) -> list[Path]:
        """Helper: create fake video files with existing captions."""
        clips = []
        for i in range(count):
            clip = tmp_path / f"clip_{i:02d}.mp4"
            clip.write_bytes(b"\x00" * 100)
            caption = tmp_path / f"clip_{i:02d}.txt"
            caption.write_text(f"Existing caption for clip {i}", encoding="utf-8")
            clips.append(clip)
        return clips

    def test_audit_all_captioned(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Audits all clips that have existing captions."""
        self._make_captioned_clips(tmp_path, 3)
        mock = MockBackend()
        monkeypatch.setattr(
            "klippbok.caption.captioner._create_backend",
            lambda config: mock,
        )

        config = CaptionConfig(provider="gemini", between_request_delay=0)
        results = audit_captions(tmp_path, config)

        assert len(results) == 3
        assert len(mock.calls) == 3

    def test_only_captioned_clips_audited(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only clips with existing .txt files are audited."""
        # 3 clips, only 1 has caption
        for i in range(3):
            (tmp_path / f"clip_{i:02d}.mp4").write_bytes(b"\x00")
        (tmp_path / "clip_01.txt").write_text("existing caption")

        mock = MockBackend()
        monkeypatch.setattr(
            "klippbok.caption.captioner._create_backend",
            lambda config: mock,
        )

        config = CaptionConfig(provider="gemini", between_request_delay=0)
        results = audit_captions(tmp_path, config)

        assert len(results) == 1
        assert len(mock.calls) == 1

    def test_recommendation_keep(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """High word overlap → 'keep' recommendation."""
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00")
        # MockBackend returns "Caption for clip" — write similar existing caption
        (tmp_path / "clip.txt").write_text("Caption for clip is very similar")

        mock = MockBackend()
        monkeypatch.setattr(
            "klippbok.caption.captioner._create_backend",
            lambda config: mock,
        )

        config = CaptionConfig(provider="gemini", between_request_delay=0)
        results = audit_captions(tmp_path, config)

        assert len(results) == 1
        # "Caption for clip" vs "Caption for clip is very similar" — high overlap
        assert results[0].recommendation == "keep"

    def test_recommendation_review(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Low word overlap → 'review' recommendation."""
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00")
        # Write completely different existing caption
        (tmp_path / "clip.txt").write_text("Totally different unrelated text here")

        mock = MockBackend()
        monkeypatch.setattr(
            "klippbok.caption.captioner._create_backend",
            lambda config: mock,
        )

        config = CaptionConfig(provider="gemini", between_request_delay=0)
        results = audit_captions(tmp_path, config)

        assert len(results) == 1
        assert results[0].recommendation == "review"

    def test_save_audit_mode(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """save_audit mode writes .audit.txt files for 'review' recommendations."""
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00")
        (tmp_path / "clip.txt").write_text("Completely different existing text")

        mock = MockBackend()
        monkeypatch.setattr(
            "klippbok.caption.captioner._create_backend",
            lambda config: mock,
        )

        config = CaptionConfig(
            provider="gemini",
            audit_mode="save_audit",
            between_request_delay=0,
        )
        results = audit_captions(tmp_path, config)

        assert len(results) == 1
        if results[0].recommendation == "review":
            audit_file = tmp_path / "clip.audit.txt"
            assert audit_file.exists()
            assert audit_file.read_text(encoding="utf-8") == "Caption for clip"

    def test_report_only_no_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """report_only mode does NOT write .audit.txt files."""
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00")
        (tmp_path / "clip.txt").write_text("Completely different existing text")

        mock = MockBackend()
        monkeypatch.setattr(
            "klippbok.caption.captioner._create_backend",
            lambda config: mock,
        )

        config = CaptionConfig(
            provider="gemini",
            audit_mode="report_only",
            between_request_delay=0,
        )
        audit_captions(tmp_path, config)

        audit_file = tmp_path / "clip.audit.txt"
        assert not audit_file.exists()

    def test_nonexistent_directory(self) -> None:
        """Raises FileNotFoundError for missing directory."""
        config = CaptionConfig(provider="gemini")
        with pytest.raises(FileNotFoundError):
            audit_captions("/nonexistent/path", config)

    def test_no_captioned_clips(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns empty list when no clips have captions."""
        (tmp_path / "clip.mp4").write_bytes(b"\x00")
        # No .txt file

        mock = MockBackend()
        monkeypatch.setattr(
            "klippbok.caption.captioner._create_backend",
            lambda config: mock,
        )

        config = CaptionConfig(provider="gemini", between_request_delay=0)
        results = audit_captions(tmp_path, config)

        assert results == []

    def test_audit_records_both_captions(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """AuditResult records both existing and VLM captions."""
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00")
        (tmp_path / "clip.txt").write_text("My existing caption")

        mock = MockBackend()
        monkeypatch.setattr(
            "klippbok.caption.captioner._create_backend",
            lambda config: mock,
        )

        config = CaptionConfig(provider="gemini", between_request_delay=0)
        results = audit_captions(tmp_path, config)

        assert results[0].existing_caption == "My existing caption"
        assert results[0].vlm_caption == "Caption for clip"
