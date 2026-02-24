"""Replicate VLM backend for video/image captioning.

Uses raw HTTP requests to Replicate's predictions API.

The google/gemini-2.5-flash model on Replicate expects:
  - "videos": list of URIs (data URIs or URLs)
  - "images": list of URIs (data URIs or URLs)
  - "prompt": text string

We send files as base64 data URIs so no external hosting is needed.

Requires: pip install requests
Environment variable: REPLICATE_API_TOKEN
"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from pathlib import Path

from klippbok.caption.base import VLMBackend

# Video extensions we recognize
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".wmv", ".flv"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}


class ReplicateBackend(VLMBackend):
    """Replicate API backend for captioning.

    Sends files as base64 data URIs via raw HTTP POST to Replicate's
    predictions API. Uses the `Prefer: wait` header for synchronous
    predictions (blocks until result is ready).

    The model's input schema determines which fields to use:
    - google/gemini-2.5-flash: "videos" (list) and "images" (list)
    - Other models may use "media", "image", "video" etc.

    We auto-detect the model's schema on first use and cache it.
    """

    def __init__(
        self,
        api_token: str | None = None,
        model: str = "google/gemini-2.5-flash",
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        """Initialize the Replicate backend.

        Args:
            api_token: Replicate API token. If None, reads from REPLICATE_API_TOKEN.
            model: Replicate model identifier (default: google/gemini-2.5-flash).
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts for failed requests.
        """
        self.api_token = api_token or os.environ.get("REPLICATE_API_TOKEN", "")
        if not self.api_token:
            raise ValueError(
                "Replicate API token not found. Set the REPLICATE_API_TOKEN "
                "environment variable or pass api_token to the constructor."
            )

        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # Build the API endpoint URL
        self.api_url = f"https://api.replicate.com/v1/models/{model}/predictions"
        self.model_url = f"https://api.replicate.com/v1/models/{model}"

        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Prefer": "wait",  # synchronous - blocks until result ready
        }

        # Cached input schema (fetched once on first use)
        self._input_schema: dict | None = None

        # Verify requests is available
        try:
            import requests as _requests  # noqa: F401
        except ImportError:
            raise ImportError(
                "requests package not installed. "
                "Install with: pip install requests"
            )

    def _fetch_input_schema(self) -> dict:
        """Fetch the model's input schema from Replicate API.

        Returns a dict of property names to their schema definitions.
        This tells us exactly which field names the model accepts.
        """
        if self._input_schema is not None:
            return self._input_schema

        import requests

        resp = requests.get(self.model_url, headers=self.headers, timeout=30)
        if resp.status_code != 200:
            # Fallback: assume gemini-style schema
            self._input_schema = {}
            return self._input_schema

        model_data = resp.json()
        version_id = model_data.get("latest_version", {}).get("id")
        if not version_id:
            self._input_schema = {}
            return self._input_schema

        vresp = requests.get(
            f"{self.model_url}/versions/{version_id}",
            headers=self.headers,
            timeout=30,
        )
        if vresp.status_code != 200:
            self._input_schema = {}
            return self._input_schema

        schemas = (
            vresp.json()
            .get("openapi_schema", {})
            .get("components", {})
            .get("schemas", {})
        )
        self._input_schema = schemas.get("Input", {}).get("properties", {})
        return self._input_schema

    def _file_to_data_uri(self, path: Path) -> str:
        """Convert a file to a base64 data URI.

        Format: data:<mime_type>;base64,<encoded_data>

        Args:
            path: Path to the file.

        Returns:
            Base64 data URI string.
        """
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            suffix = path.suffix.lower()
            mime_map = {
                ".mp4": "video/mp4",
                ".mov": "video/quicktime",
                ".mkv": "video/x-matroska",
                ".avi": "video/x-msvideo",
                ".webm": "video/webm",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }
            mime_type = mime_map.get(suffix, "application/octet-stream")

        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{data}"

    def _build_payload(self, prompt: str, data_uri: str, is_video: bool) -> dict:
        """Build the API payload using the model's actual input schema.

        Different Replicate models use different field names:
        - google/gemini-2.5-flash: "videos" (list), "images" (list)
        - Some models: "media" (single URI)
        - Some models: "video" (single URI), "image" (single URI)

        We check the schema and use whatever the model expects.

        Args:
            prompt: The captioning prompt.
            data_uri: Base64 data URI of the file.
            is_video: Whether the file is a video (vs image).

        Returns:
            The request payload dict.
        """
        schema = self._fetch_input_schema()
        props = set(schema.keys())

        input_data: dict = {"prompt": prompt}

        if is_video:
            # Try field names in preference order for video
            if "videos" in props:
                # Array field (gemini-2.5-flash style)
                input_data["videos"] = [data_uri]
            elif "video" in props:
                # Single-value field
                input_data["video"] = data_uri
            elif "media" in props:
                # Generic media field
                input_data["media"] = data_uri
            else:
                # Fallback: try "videos" anyway (model might accept it)
                input_data["videos"] = [data_uri]
        else:
            # Try field names in preference order for images
            if "images" in props:
                input_data["images"] = [data_uri]
            elif "image" in props:
                input_data["image"] = data_uri
            elif "media" in props:
                input_data["media"] = data_uri
            else:
                input_data["images"] = [data_uri]

        return {"input": input_data}

    def _post_prediction(self, prompt: str, data_uri: str, is_video: bool) -> str:
        """Make a raw HTTP POST to Replicate's predictions API.

        Args:
            prompt: The captioning prompt.
            data_uri: Base64 data URI of the file.
            is_video: Whether the file is a video (vs image).

        Returns:
            The generated caption text.

        Raises:
            RuntimeError: on API errors.
        """
        import requests

        payload = self._build_payload(prompt, data_uri, is_video)

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code == 422:
            raise ValueError(
                f"422 Unprocessable: the model rejected our input. "
                f"Response: {response.text[:300]}"
            )

        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"Replicate API error {response.status_code}: "
                f"{response.text[:200]}"
            )

        result = response.json()

        # Check if the prediction actually succeeded
        status = result.get("status", "")
        if status == "failed":
            error = result.get("error", "Unknown error")
            raise RuntimeError(f"Prediction failed: {error}")

        # Extract the output text from the response
        output = result.get("output", "")
        if isinstance(output, list):
            return "".join(str(item) for item in output).strip()
        return str(output).strip()

    def caption_video(self, path: Path, prompt: str) -> str:
        """Caption a video via Replicate API.

        Args:
            path: Path to the video file.
            prompt: The captioning prompt.

        Returns:
            Generated caption text.
        """
        data_uri = self._file_to_data_uri(path)
        return self._run_with_retry(data_uri, prompt, is_video=True)

    def caption_image(self, path: Path, prompt: str) -> str:
        """Caption an image via Replicate API.

        Args:
            path: Path to the image file.
            prompt: The captioning prompt.

        Returns:
            Generated caption text.
        """
        data_uri = self._file_to_data_uri(path)
        return self._run_with_retry(data_uri, prompt, is_video=False)

    def _run_with_retry(self, data_uri: str, prompt: str, is_video: bool) -> str:
        """Run prediction with retry logic.

        Retries on timeouts and server errors. Uses the model's actual
        input schema to send data in the correct field.

        Args:
            data_uri: Base64 data URI of the file.
            prompt: The captioning prompt.
            is_video: Whether the file is a video (vs image).

        Returns:
            Generated caption text.
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return self._post_prediction(prompt, data_uri, is_video)

            except ValueError:
                # 422 = model rejected input, no point retrying
                raise

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Retryable errors
                if any(kw in error_str for kw in ("timeout", "503", "rate", "429")):
                    wait_time = 15 * (attempt + 1)
                    print(f"    Retrying in {wait_time}s "
                          f"(attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue

                raise

        raise RuntimeError(
            f"Failed after {self.max_retries} attempts: {last_error}"
        )
