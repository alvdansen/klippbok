"""Tests for klippbok.caption.replicate — Replicate backend with mocked HTTP.

All tests mock the requests library so no real API calls are made.
Tests cover initialization, schema fetching, payload building,
post prediction, and retry logic.

The Replicate backend imports `requests` inside each method call,
so we mock functions on the actual requests module object.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests as _real_requests  # the real module — we mock its methods


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestReplicateInit:
    """Tests for ReplicateBackend.__init__."""

    def test_requires_api_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError if no API token is provided or in env."""
        monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)

        from klippbok.caption.replicate import ReplicateBackend
        with pytest.raises(ValueError, match="Replicate API token not found"):
            ReplicateBackend(api_token=None)

    def test_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reads API token from REPLICATE_API_TOKEN environment variable."""
        monkeypatch.setenv("REPLICATE_API_TOKEN", "env-token-123")

        from klippbok.caption.replicate import ReplicateBackend
        backend = ReplicateBackend()
        assert backend.api_token == "env-token-123"

    def test_explicit_token_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit api_token parameter takes priority over env var."""
        monkeypatch.setenv("REPLICATE_API_TOKEN", "env-token")

        from klippbok.caption.replicate import ReplicateBackend
        backend = ReplicateBackend(api_token="explicit-token")
        assert backend.api_token == "explicit-token"

    def test_api_url_built_from_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """API URL is constructed from the model identifier."""
        monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token")

        from klippbok.caption.replicate import ReplicateBackend
        backend = ReplicateBackend(model="meta/llama-vision")
        assert "meta/llama-vision" in backend.api_url

    def test_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default model is google/gemini-2.5-flash."""
        monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token")

        from klippbok.caption.replicate import ReplicateBackend
        backend = ReplicateBackend()
        assert backend.model == "google/gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Schema fetching tests
# ---------------------------------------------------------------------------

class TestReplicateSchema:
    """Tests for _fetch_input_schema()."""

    def _make_backend(self, monkeypatch: pytest.MonkeyPatch):
        """Create a ReplicateBackend with test token."""
        monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token")
        from klippbok.caption.replicate import ReplicateBackend
        return ReplicateBackend(api_token="test-token")

    def test_fetches_schema_from_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fetches and parses model input schema from Replicate API."""
        backend = self._make_backend(monkeypatch)

        # Mock the two GET requests (model info -> version info)
        mock_model_resp = MagicMock()
        mock_model_resp.status_code = 200
        mock_model_resp.json.return_value = {
            "latest_version": {"id": "abc123"}
        }

        mock_version_resp = MagicMock()
        mock_version_resp.status_code = 200
        mock_version_resp.json.return_value = {
            "openapi_schema": {
                "components": {
                    "schemas": {
                        "Input": {
                            "properties": {
                                "prompt": {"type": "string"},
                                "videos": {"type": "array"},
                                "images": {"type": "array"},
                            }
                        }
                    }
                }
            }
        }

        monkeypatch.setattr(
            _real_requests, "get",
            MagicMock(side_effect=[mock_model_resp, mock_version_resp]),
        )

        schema = backend._fetch_input_schema()

        assert "prompt" in schema
        assert "videos" in schema
        assert "images" in schema

    def test_caches_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Schema is fetched once and cached."""
        backend = self._make_backend(monkeypatch)

        # Pre-set the cache
        backend._input_schema = {"prompt": {}, "videos": {}}
        schema = backend._fetch_input_schema()

        assert schema == {"prompt": {}, "videos": {}}

    def test_fallback_on_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to empty schema on API error."""
        backend = self._make_backend(monkeypatch)

        mock_resp = MagicMock()
        mock_resp.status_code = 500

        monkeypatch.setattr(_real_requests, "get", MagicMock(return_value=mock_resp))

        schema = backend._fetch_input_schema()
        assert schema == {}


# ---------------------------------------------------------------------------
# Payload building tests
# ---------------------------------------------------------------------------

class TestReplicatePayload:
    """Tests for _build_payload()."""

    def _make_backend(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token")
        from klippbok.caption.replicate import ReplicateBackend
        return ReplicateBackend(api_token="test-token")

    def test_videos_field_for_video(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Uses 'videos' array field when schema has it."""
        backend = self._make_backend(monkeypatch)
        backend._input_schema = {"prompt": {}, "videos": {}, "images": {}}

        payload = backend._build_payload("Describe", "data:video/mp4;base64,abc", is_video=True)

        assert payload["input"]["videos"] == ["data:video/mp4;base64,abc"]
        assert "video" not in payload["input"]

    def test_video_singular_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Uses 'video' singular field when schema has it (not 'videos')."""
        backend = self._make_backend(monkeypatch)
        backend._input_schema = {"prompt": {}, "video": {}}

        payload = backend._build_payload("Describe", "data:video/mp4;base64,abc", is_video=True)

        assert payload["input"]["video"] == "data:video/mp4;base64,abc"

    def test_media_field_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Uses 'media' field when neither 'videos' nor 'video' in schema."""
        backend = self._make_backend(monkeypatch)
        backend._input_schema = {"prompt": {}, "media": {}}

        payload = backend._build_payload("Describe", "data:video/mp4;base64,abc", is_video=True)

        assert payload["input"]["media"] == "data:video/mp4;base64,abc"

    def test_images_field_for_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Uses 'images' array field for images when schema has it."""
        backend = self._make_backend(monkeypatch)
        backend._input_schema = {"prompt": {}, "images": {}}

        payload = backend._build_payload("Describe", "data:image/jpeg;base64,abc", is_video=False)

        assert payload["input"]["images"] == ["data:image/jpeg;base64,abc"]

    def test_fallback_videos_when_no_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to 'videos' when schema is empty."""
        backend = self._make_backend(monkeypatch)
        backend._input_schema = {}

        payload = backend._build_payload("Describe", "data:video/mp4;base64,abc", is_video=True)

        assert payload["input"]["videos"] == ["data:video/mp4;base64,abc"]


# ---------------------------------------------------------------------------
# Post prediction tests
# ---------------------------------------------------------------------------

class TestReplicatePostPrediction:
    """Tests for _post_prediction()."""

    def _make_backend(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token")
        from klippbok.caption.replicate import ReplicateBackend
        backend = ReplicateBackend(api_token="test-token")
        backend._input_schema = {"prompt": {}, "videos": {}, "images": {}}
        return backend

    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Successful prediction returns output text."""
        backend = self._make_backend(monkeypatch)

        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "status": "succeeded",
            "output": "A cat sitting on a roof",
        }

        monkeypatch.setattr(_real_requests, "post", MagicMock(return_value=mock_resp))

        result = backend._post_prediction("Describe", "data:video;base64,abc", is_video=True)
        assert result == "A cat sitting on a roof"

    def test_list_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Handles list output (some models return chunks)."""
        backend = self._make_backend(monkeypatch)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "succeeded",
            "output": ["A cat ", "sitting on ", "a roof"],
        }

        monkeypatch.setattr(_real_requests, "post", MagicMock(return_value=mock_resp))

        result = backend._post_prediction("Describe", "data:video;base64,abc", is_video=True)
        assert result == "A cat sitting on a roof"

    def test_422_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """422 response raises ValueError (non-retryable)."""
        backend = self._make_backend(monkeypatch)

        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.text = "Unprocessable Entity: invalid input"

        monkeypatch.setattr(_real_requests, "post", MagicMock(return_value=mock_resp))

        with pytest.raises(ValueError, match="422"):
            backend._post_prediction("Describe", "data:video;base64,abc", is_video=True)

    def test_prediction_failed_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises RuntimeError when prediction status is 'failed'."""
        backend = self._make_backend(monkeypatch)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "failed",
            "error": "Model ran out of memory",
        }

        monkeypatch.setattr(_real_requests, "post", MagicMock(return_value=mock_resp))

        with pytest.raises(RuntimeError, match="Prediction failed"):
            backend._post_prediction("Describe", "data:video;base64,abc", is_video=True)


# ---------------------------------------------------------------------------
# Data URI encoding tests
# ---------------------------------------------------------------------------

class TestReplicateDataUri:
    """Tests for _file_to_data_uri()."""

    def _make_backend(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token")
        from klippbok.caption.replicate import ReplicateBackend
        return ReplicateBackend(api_token="test-token")

    def test_mp4_data_uri(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Creates correct data URI for .mp4 files."""
        backend = self._make_backend(monkeypatch)

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00\x01\x02\x03")

        uri = backend._file_to_data_uri(video)
        assert uri.startswith("data:video/mp4;base64,")

    def test_jpg_data_uri(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Creates correct data URI for .jpg files."""
        backend = self._make_backend(monkeypatch)

        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0")

        uri = backend._file_to_data_uri(img)
        assert uri.startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------

class TestReplicateRetry:
    """Tests for _run_with_retry()."""

    def _make_backend(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token")
        monkeypatch.setattr("klippbok.caption.replicate.time.sleep", lambda x: None)
        from klippbok.caption.replicate import ReplicateBackend
        backend = ReplicateBackend(api_token="test-token", max_retries=3)
        backend._input_schema = {"prompt": {}, "videos": {}}
        return backend

    def test_retries_on_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retries on 503 service unavailable."""
        backend = self._make_backend(monkeypatch)

        # First call: 503 error raises RuntimeError inside _post_prediction.
        # Second call: success.
        mock_resp_fail = MagicMock()
        mock_resp_fail.status_code = 503
        mock_resp_fail.text = "503 Service Unavailable"

        mock_resp_ok = MagicMock()
        mock_resp_ok.status_code = 200
        mock_resp_ok.json.return_value = {
            "status": "succeeded",
            "output": "success caption",
        }

        monkeypatch.setattr(
            _real_requests, "post",
            MagicMock(side_effect=[mock_resp_fail, mock_resp_ok]),
        )

        result = backend._run_with_retry("data:video;base64,abc", "Describe", is_video=True)
        assert result == "success caption"

    def test_422_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """422 errors are not retried (raised immediately)."""
        backend = self._make_backend(monkeypatch)

        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.text = "Unprocessable"

        mock_post = MagicMock(return_value=mock_resp)
        monkeypatch.setattr(_real_requests, "post", mock_post)

        with pytest.raises(ValueError, match="422"):
            backend._run_with_retry("data:video;base64,abc", "Describe", is_video=True)

        # Should have called only once (no retries for 422)
        assert mock_post.call_count == 1
