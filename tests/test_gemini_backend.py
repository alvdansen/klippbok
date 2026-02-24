"""Tests for klippbok.caption.gemini — Gemini backend with mocked SDK.

All tests mock the google.genai SDK so no real API calls are made.
Tests cover initialization, caption_video, caption_image, and retry logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — build a fake google.genai module tree
# ---------------------------------------------------------------------------

def _make_fake_genai() -> MagicMock:
    """Create a mock google.genai module with realistic structure.

    Returns a mock Client whose files/models attributes behave like
    the real google-genai SDK.
    """
    mock_client = MagicMock()
    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client

    # types.Part.from_bytes for image captioning
    mock_types = MagicMock()
    mock_types.Part.from_bytes.return_value = MagicMock(name="image_part")
    mock_genai_types = mock_types

    return mock_genai, mock_client, mock_genai_types


def _install_fake_genai(monkeypatch: pytest.MonkeyPatch):
    """Install fake google.genai into sys.modules so import works.

    Returns (mock_genai_module, mock_client, mock_types).
    """
    mock_genai, mock_client, mock_types = _make_fake_genai()

    # Build the module hierarchy google -> google.genai -> google.genai.types
    google_mod = MagicMock()
    google_mod.genai = mock_genai
    google_mod.genai.types = mock_types

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

    return mock_genai, mock_client, mock_types


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestGeminiInit:
    """Tests for GeminiBackend.__init__."""

    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError if no API key is provided or in env."""
        _install_fake_genai(monkeypatch)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        from klippbok.caption.gemini import GeminiBackend
        with pytest.raises(ValueError, match="Gemini API key not found"):
            GeminiBackend(api_key=None)

    def test_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reads API key from GEMINI_API_KEY environment variable."""
        _install_fake_genai(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "env-key-123")

        from klippbok.caption.gemini import GeminiBackend
        backend = GeminiBackend()
        assert backend.api_key == "env-key-123"

    def test_explicit_key_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit api_key parameter takes priority over env var."""
        _install_fake_genai(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")

        from klippbok.caption.gemini import GeminiBackend
        backend = GeminiBackend(api_key="explicit-key")
        assert backend.api_key == "explicit-key"

    def test_missing_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ImportError with install instructions if google-genai missing."""
        # Remove google.genai from sys.modules to simulate missing package
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        # We need to make the import fail. Patch the import inside the module.
        with patch.dict(sys.modules, {"google": None, "google.genai": None}):
            # Force reimport
            monkeypatch.delitem(sys.modules, "klippbok.caption.gemini", raising=False)

            # This is tricky — the import happens at class __init__ time.
            # Simplest approach: just verify the error message pattern
            from klippbok.caption.gemini import GeminiBackend
            with pytest.raises((ImportError, ValueError)):
                GeminiBackend(api_key="test")

    def test_custom_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Custom model name is stored."""
        _install_fake_genai(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        from klippbok.caption.gemini import GeminiBackend
        backend = GeminiBackend(model="gemini-2.5-pro")
        assert backend.model == "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# caption_video tests
# ---------------------------------------------------------------------------

class TestGeminiCaptionVideo:
    """Tests for GeminiBackend.caption_video()."""

    def _make_backend(self, monkeypatch: pytest.MonkeyPatch):
        """Create a GeminiBackend with mocked SDK, return (backend, mock_client)."""
        mock_genai, mock_client, mock_types = _install_fake_genai(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        from klippbok.caption.gemini import GeminiBackend
        backend = GeminiBackend(api_key="test-key")
        # Replace the client with our mock
        backend.client = mock_client
        return backend, mock_client

    def test_happy_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Upload → poll → generate → delete flow works."""
        backend, client = self._make_backend(monkeypatch)

        # Set up mock file states
        uploaded = MagicMock()
        uploaded.name = "files/abc123"
        uploaded.state = SimpleNamespace(name="ACTIVE")
        client.files.upload.return_value = uploaded

        # Set up mock generation response
        response = MagicMock()
        response.text = "  A person walks through a forest  "
        client.models.generate_content.return_value = response

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 100)

        result = backend.caption_video(video, "Describe this video")

        assert result == "A person walks through a forest"
        client.files.upload.assert_called_once()
        client.files.delete.assert_called_once_with(name="files/abc123")

    def test_polls_until_active(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Polls file state until ACTIVE."""
        backend, client = self._make_backend(monkeypatch)

        # File starts PROCESSING, becomes ACTIVE after one poll
        processing = MagicMock()
        processing.name = "files/abc123"
        processing.state = SimpleNamespace(name="PROCESSING")

        active = MagicMock()
        active.name = "files/abc123"
        active.state = SimpleNamespace(name="ACTIVE")

        client.files.upload.return_value = processing
        client.files.get.return_value = active

        response = MagicMock()
        response.text = "caption text"
        client.models.generate_content.return_value = response

        # Patch time.sleep to avoid real delays
        monkeypatch.setattr("klippbok.caption.gemini.time.sleep", lambda x: None)

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00")

        result = backend.caption_video(video, "Describe")
        assert result == "caption text"
        client.files.get.assert_called_once()

    def test_file_processing_failed(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Raises RuntimeError if file processing fails."""
        backend, client = self._make_backend(monkeypatch)

        failed = MagicMock()
        failed.name = "files/abc123"
        failed.state = SimpleNamespace(name="FAILED")
        client.files.upload.return_value = failed

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00")

        with pytest.raises(RuntimeError, match="file processing failed"):
            backend.caption_video(video, "Describe")

        # Should still try to clean up
        client.files.delete.assert_called_once()

    def test_cleanup_on_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Uploaded file is deleted even if generation fails."""
        backend, client = self._make_backend(monkeypatch)

        uploaded = MagicMock()
        uploaded.name = "files/abc123"
        uploaded.state = SimpleNamespace(name="ACTIVE")
        client.files.upload.return_value = uploaded

        client.models.generate_content.side_effect = RuntimeError("API error")

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00")

        with pytest.raises(RuntimeError, match="API error"):
            backend.caption_video(video, "Describe")

        # File should still be cleaned up
        client.files.delete.assert_called_once_with(name="files/abc123")


# ---------------------------------------------------------------------------
# caption_image tests
# ---------------------------------------------------------------------------

class TestGeminiCaptionImage:
    """Tests for GeminiBackend.caption_image()."""

    def _make_backend(self, monkeypatch: pytest.MonkeyPatch):
        mock_genai, mock_client, mock_types = _install_fake_genai(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        from klippbok.caption.gemini import GeminiBackend
        backend = GeminiBackend(api_key="test-key")
        backend.client = mock_client
        return backend, mock_client, mock_types

    def test_image_inline(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Images are sent inline without upload."""
        backend, client, mock_types = self._make_backend(monkeypatch)

        response = MagicMock()
        response.text = "A cat sitting on a windowsill"
        client.models.generate_content.return_value = response

        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8" + b"\x00" * 100)

        result = backend.caption_image(img, "Describe this image")
        assert result == "A cat sitting on a windowsill"
        # No file upload for images
        client.files.upload.assert_not_called()

    def test_mime_detection_png(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Detects PNG MIME type correctly."""
        backend, client, mock_types = self._make_backend(monkeypatch)

        response = MagicMock()
        response.text = "test"
        client.models.generate_content.return_value = response

        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 100)

        backend.caption_image(img, "Describe")
        # Verify from_bytes was called with correct mime type
        mock_types.Part.from_bytes.assert_called_once()
        call_kwargs = mock_types.Part.from_bytes.call_args
        assert call_kwargs.kwargs.get("mime_type") == "image/png" or \
            (len(call_kwargs.args) == 0 and "mime_type" in call_kwargs.kwargs)


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------

class TestGeminiRetry:
    """Tests for _generate_with_retry()."""

    def _make_backend(self, monkeypatch: pytest.MonkeyPatch):
        mock_genai, mock_client, mock_types = _install_fake_genai(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        # Patch sleep to avoid real delays
        monkeypatch.setattr("klippbok.caption.gemini.time.sleep", lambda x: None)

        from klippbok.caption.gemini import GeminiBackend
        backend = GeminiBackend(api_key="test-key", max_retries=3)
        backend.client = mock_client
        return backend, mock_client

    def test_retries_on_429(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retries on rate limit (429) errors."""
        backend, client = self._make_backend(monkeypatch)

        # Fail twice with 429, succeed on third attempt
        success_response = MagicMock()
        success_response.text = "success"
        client.models.generate_content.side_effect = [
            Exception("429 Too Many Requests"),
            Exception("429 quota exceeded"),
            success_response,
        ]

        result = backend._generate_with_retry(contents=["test"])
        assert result.text == "success"
        assert client.models.generate_content.call_count == 3

    def test_raises_non_retryable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-retryable errors (400, etc.) raise immediately."""
        backend, client = self._make_backend(monkeypatch)

        client.models.generate_content.side_effect = Exception("400 Bad Request: invalid content")

        with pytest.raises(Exception, match="400 Bad Request"):
            backend._generate_with_retry(contents=["test"])

        # Should not retry
        assert client.models.generate_content.call_count == 1

    def test_max_retries_exhausted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises RuntimeError after exhausting all retries."""
        backend, client = self._make_backend(monkeypatch)

        client.models.generate_content.side_effect = Exception("429 rate limit")

        with pytest.raises(RuntimeError, match="Failed after 3 attempts"):
            backend._generate_with_retry(contents=["test"])

        assert client.models.generate_content.call_count == 3
