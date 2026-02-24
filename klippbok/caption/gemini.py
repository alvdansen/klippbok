"""Gemini VLM backend for video/image captioning.

Uses the Google Generative AI SDK (google-genai) for direct API access.
Handles video upload, polling, captioning, and cleanup.

Based on patterns from lora-gym captioning scripts.

Requires: pip install google-genai  (or pip install 'klippbok[caption]')
Environment variable: GEMINI_API_KEY
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from klippbok.caption.base import VLMBackend


class GeminiBackend(VLMBackend):
    """Google Gemini API backend for captioning.

    Handles two modes:
    - Image: direct inline data (fast, no upload needed)
    - Video: upload to Gemini Files API → poll until ACTIVE → generate → delete

    Rate limiting: configurable delay between requests (default 10s
    for Gemini free tier at ~20 req/min).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        timeout: int = 120,
        max_retries: int = 5,
        caption_fps: int = 1,
    ) -> None:
        """Initialize the Gemini backend.

        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
            model: Gemini model name (default: gemini-2.0-flash).
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts for failed/rate-limited requests.
            caption_fps: Frame sampling rate hint for server-side processing.
                Gemini extracts frames server-side — this controls the rate.
                Default 1 FPS. Increase for motion-focused datasets.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set the GEMINI_API_KEY environment "
                "variable or pass api_key to the constructor."
            )

        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.caption_fps = caption_fps

        # Import and configure the SDK
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install 'klippbok[caption]' "
                "or: pip install google-genai"
            )

        self.client = genai.Client(api_key=self.api_key)

    def caption_video(self, path: Path, prompt: str) -> str:
        """Caption a video via Gemini API.

        Workflow:
        1. Upload video to Gemini Files API
        2. Poll until file state is ACTIVE
        3. Generate caption
        4. Delete uploaded file (cleanup)

        Args:
            path: Path to the video file.
            prompt: The captioning prompt.

        Returns:
            Generated caption text.
        """
        uploaded_file = None
        try:
            # Upload video — use IO wrapper to avoid ASCII encoding errors
            # in the Gemini SDK when filenames contain Unicode (e.g. curly
            # quotes in "Tiffany's").
            import io

            video_bytes = path.read_bytes()
            suffix = path.suffix.lower()
            video_mime = {
                ".mp4": "video/mp4",
                ".mov": "video/quicktime",
                ".mkv": "video/x-matroska",
                ".avi": "video/x-msvideo",
                ".webm": "video/webm",
            }.get(suffix, "video/mp4")
            # Use ASCII-safe stem for the upload display name
            safe_name = path.stem.encode("ascii", "replace").decode("ascii") + suffix
            uploaded_file = self.client.files.upload(
                file=io.BytesIO(video_bytes),
                config={"mime_type": video_mime, "display_name": safe_name},
            )

            # Poll until processing is complete
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = self.client.files.get(name=uploaded_file.name)

            if uploaded_file.state.name != "ACTIVE":
                raise RuntimeError(
                    f"Gemini file processing failed: state={uploaded_file.state.name}"
                )

            # Generate caption
            response = self._generate_with_retry(
                contents=[uploaded_file, prompt],
            )
            return response.text.strip()

        finally:
            # Always clean up uploaded files
            if uploaded_file:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass  # cleanup failure is not critical

    def caption_image(self, path: Path, prompt: str) -> str:
        """Caption an image via Gemini API.

        Images are sent inline (no upload needed) — much faster than video.

        Args:
            path: Path to the image file.
            prompt: The captioning prompt.

        Returns:
            Generated caption text.
        """
        # Read image as bytes for inline sending
        image_bytes = path.read_bytes()

        # Determine MIME type
        suffix = path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime_type = mime_types.get(suffix, "image/jpeg")

        from google.genai import types

        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        response = self._generate_with_retry(
            contents=[image_part, prompt],
        )
        return response.text.strip()

    def _generate_with_retry(self, contents: list) -> object:
        """Call Gemini generate_content with retry logic.

        Retries on 429 (rate limit) and 503 (service unavailable) errors
        with exponential backoff.

        Args:
            contents: Content parts to send to the model.

        Returns:
            The generation response object.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                )
                return response

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Retry on rate limits and server errors
                if any(kw in error_str for kw in ("429", "quota", "rate", "503", "overloaded")):
                    wait_time = 45 * (attempt + 1)  # 45s, 90s, 135s, ...
                    print(f"    Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue

                # Non-retryable error
                raise

        raise RuntimeError(
            f"Failed after {self.max_retries} attempts: {last_error}"
        )
