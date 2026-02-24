"""Tests for klippbok.caption.openai_compat — OpenAI-compatible backend.

All tests mock HTTP requests and frame extraction so no real API calls
or ffmpeg invocations are needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests as _real_requests

from klippbok.caption.openai_compat import OpenAICompatBackend


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestOpenAIInit:
    """Tests for OpenAICompatBackend.__init__."""

    def test_default_config(self) -> None:
        """Default config points to Ollama with llama3.2-vision."""
        backend = OpenAICompatBackend()
        assert backend.base_url == "http://localhost:11434/v1"
        assert backend.model == "llama3.2-vision"
        assert backend.caption_fps == 1

    def test_custom_config(self) -> None:
        """Custom base URL, model, and FPS."""
        backend = OpenAICompatBackend(
            base_url="http://myserver:8000/v1",
            model="qwen2-vl",
            caption_fps=4,
        )
        assert backend.base_url == "http://myserver:8000/v1"
        assert backend.model == "qwen2-vl"
        assert backend.caption_fps == 4

    def test_strips_trailing_slash(self) -> None:
        """Strips trailing slash from base_url."""
        backend = OpenAICompatBackend(base_url="http://localhost:8000/v1/")
        assert backend.base_url == "http://localhost:8000/v1"

    def test_api_key_optional(self) -> None:
        """API key defaults to 'not-needed' for local servers."""
        backend = OpenAICompatBackend()
        assert backend.api_key == "not-needed"

    def test_custom_api_key(self) -> None:
        """Can provide a real API key for hosted services."""
        backend = OpenAICompatBackend(api_key="sk-real-key")
        assert backend.api_key == "sk-real-key"


# ---------------------------------------------------------------------------
# Image content building tests
# ---------------------------------------------------------------------------

class TestBuildImageContent:
    """Tests for _build_image_content()."""

    def test_single_image(self, tmp_path: Path) -> None:
        """Single image creates one image block + one text block."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8" + b"\x00" * 100)

        backend = OpenAICompatBackend()
        content = backend._build_image_content([img], "Describe this")

        assert len(content) == 2
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "Describe this"

    def test_multiple_images(self, tmp_path: Path) -> None:
        """Multiple images create multiple image blocks before the text."""
        for i in range(3):
            (tmp_path / f"frame_{i:03d}.jpg").write_bytes(b"\xff\xd8\x00")

        frames = sorted(tmp_path.glob("*.jpg"))
        backend = OpenAICompatBackend()
        content = backend._build_image_content(frames, "Describe the sequence")

        # 3 image blocks + 1 text block
        assert len(content) == 4
        assert all(c["type"] == "image_url" for c in content[:3])
        assert content[3]["type"] == "text"

    def test_png_mime_type(self, tmp_path: Path) -> None:
        """PNG files get the correct MIME type."""
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 100)

        backend = OpenAICompatBackend()
        content = backend._build_image_content([img], "Describe")

        assert "image/png" in content[0]["image_url"]["url"]


# ---------------------------------------------------------------------------
# API call tests
# ---------------------------------------------------------------------------

class TestCallAPI:
    """Tests for _call_api()."""

    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Successful API call returns text response."""
        backend = OpenAICompatBackend()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "  A cat sitting on a windowsill  ",
                    }
                }
            ]
        }

        monkeypatch.setattr(_real_requests, "post", MagicMock(return_value=mock_resp))

        result = backend._call_api([{"type": "text", "text": "test"}])
        assert result == "A cat sitting on a windowsill"

    def test_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-200 response raises RuntimeError."""
        backend = OpenAICompatBackend()

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        monkeypatch.setattr(_real_requests, "post", MagicMock(return_value=mock_resp))

        with pytest.raises(RuntimeError, match="API error 500"):
            backend._call_api([{"type": "text", "text": "test"}])

    def test_empty_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty choices array raises RuntimeError."""
        backend = OpenAICompatBackend()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": []}

        monkeypatch.setattr(_real_requests, "post", MagicMock(return_value=mock_resp))

        with pytest.raises(RuntimeError, match="No choices"):
            backend._call_api([{"type": "text", "text": "test"}])

    def test_sends_correct_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sends correct model and message format."""
        backend = OpenAICompatBackend(model="qwen2-vl")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "test"}}]
        }

        mock_post = MagicMock(return_value=mock_resp)
        monkeypatch.setattr(_real_requests, "post", mock_post)

        content = [{"type": "text", "text": "hello"}]
        backend._call_api(content)

        # Verify the request was made to the right URL
        call_args = mock_post.call_args
        assert "/chat/completions" in call_args.args[0]
        assert call_args.kwargs["json"]["model"] == "qwen2-vl"


# ---------------------------------------------------------------------------
# caption_image tests
# ---------------------------------------------------------------------------

class TestCaptionImage:
    """Tests for caption_image() — single image, no frame extraction."""

    def test_caption_image(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Captions an image without any frame extraction."""
        backend = OpenAICompatBackend()

        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8" + b"\x00" * 50)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "A sunset over mountains"}}]
        }

        monkeypatch.setattr(_real_requests, "post", MagicMock(return_value=mock_resp))

        result = backend.caption_image(img, "Describe this image")
        assert result == "A sunset over mountains"


# ---------------------------------------------------------------------------
# caption_video tests (with mocked frame extraction)
# ---------------------------------------------------------------------------

class TestCaptionVideo:
    """Tests for caption_video() — mocks frame extraction + API call."""

    def test_extracts_and_captions(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Extracts frames, builds multi-image prompt, calls API."""
        backend = OpenAICompatBackend(caption_fps=2)

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 100)

        # Mock extract_frames to return fake frame files
        frame_dir = tmp_path / "mock_frames"
        frame_dir.mkdir()
        frame_files = []
        for i in range(3):
            f = frame_dir / f"frame_{i:04d}.jpg"
            f.write_bytes(b"\xff\xd8" + b"\x00" * 10)
            frame_files.append(f)

        monkeypatch.setattr(
            "klippbok.caption.openai_compat.extract_frames",
            lambda path, output_dir, fps: frame_files,
        )

        # Mock API response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "A person walking through a park"}}]
        }
        monkeypatch.setattr(_real_requests, "post", MagicMock(return_value=mock_resp))

        # Mock shutil.rmtree to not actually clean up
        monkeypatch.setattr("klippbok.caption.openai_compat.shutil.rmtree", lambda *a, **kw: None)

        result = backend.caption_video(video, "Describe this video")
        assert result == "A person walking through a park"

    def test_includes_frame_prefix(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Prompt includes frame count and FPS info."""
        backend = OpenAICompatBackend(caption_fps=4)

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00")

        # Mock frames
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        frames = []
        for i in range(5):
            f = frame_dir / f"frame_{i:04d}.jpg"
            f.write_bytes(b"\xff\xd8\x00")
            frames.append(f)

        monkeypatch.setattr(
            "klippbok.caption.openai_compat.extract_frames",
            lambda path, output_dir, fps: frames,
        )
        monkeypatch.setattr("klippbok.caption.openai_compat.shutil.rmtree", lambda *a, **kw: None)

        # Capture the content sent to _call_api
        captured_content = []
        original_call_api = backend._call_api

        def mock_call_api(content):
            captured_content.extend(content)
            return "test caption"

        monkeypatch.setattr(backend, "_call_api", mock_call_api)

        backend.caption_video(video, "Describe")

        # The text block should include the frame prefix
        text_blocks = [c for c in captured_content if c["type"] == "text"]
        assert len(text_blocks) == 1
        assert "5 images" in text_blocks[0]["text"]  # {count} frames
        assert "4 fps" in text_blocks[0]["text"]  # {fps}

    def test_no_frames_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Raises RuntimeError if no frames are extracted."""
        backend = OpenAICompatBackend()

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00")

        monkeypatch.setattr(
            "klippbok.caption.openai_compat.extract_frames",
            lambda path, output_dir, fps: [],
        )
        monkeypatch.setattr("klippbok.caption.openai_compat.shutil.rmtree", lambda *a, **kw: None)

        with pytest.raises(RuntimeError, match="No frames extracted"):
            backend.caption_video(video, "Describe")

    def test_cleanup_on_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Temp frames are cleaned up even if API call fails."""
        backend = OpenAICompatBackend()

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00")

        # Mock frames
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        f = frame_dir / "frame_0001.jpg"
        f.write_bytes(b"\xff\xd8\x00")

        monkeypatch.setattr(
            "klippbok.caption.openai_compat.extract_frames",
            lambda path, output_dir, fps: [f],
        )

        # Track cleanup calls
        cleanup_calls = []
        monkeypatch.setattr(
            "klippbok.caption.openai_compat.shutil.rmtree",
            lambda *a, **kw: cleanup_calls.append(a),
        )

        # Make API call fail
        monkeypatch.setattr(
            _real_requests, "post",
            MagicMock(side_effect=RuntimeError("connection refused")),
        )

        with pytest.raises(RuntimeError):
            backend.caption_video(video, "Describe")

        # Cleanup should still have been called
        assert len(cleanup_calls) == 1


# ---------------------------------------------------------------------------
# Factory integration test
# ---------------------------------------------------------------------------

class TestOpenAIFactory:
    """Tests that _create_backend properly creates OpenAI backend."""

    def test_create_openai_backend(self) -> None:
        """_create_backend creates OpenAICompatBackend for 'openai' provider."""
        from klippbok.caption.captioner import _create_backend
        from klippbok.caption.models import CaptionConfig

        config = CaptionConfig(
            provider="openai",
            openai_base_url="http://localhost:8000/v1",
            openai_model="qwen2-vl",
            caption_fps=2,
        )
        backend = _create_backend(config)

        assert isinstance(backend, OpenAICompatBackend)
        assert backend.base_url == "http://localhost:8000/v1"
        assert backend.model == "qwen2-vl"
        assert backend.caption_fps == 2
