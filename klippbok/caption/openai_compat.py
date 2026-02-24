"""OpenAI-compatible VLM backend for local model captioning.

Works with any server that implements the OpenAI chat completions API:
- Ollama (default: http://localhost:11434/v1)
- vLLM
- LM Studio
- Any OpenAI-format endpoint

Local models accept images, not video. So we extract keyframes from
each clip using ffmpeg and send them as a multi-image prompt. A prefix
tells the VLM "these N frames are sampled at X fps from a video clip,
describe the continuous sequence."

Uses raw HTTP with requests — no openai package dependency.

Requires: pip install requests
"""

from __future__ import annotations

import base64
import shutil
import tempfile
from pathlib import Path

from klippbok.caption.base import VLMBackend
from klippbok.video.frames import extract_frames


# Frame prefix template — tells the VLM how to interpret the images
_FRAME_PREFIX = (
    "The following {count} images are frames extracted at {fps} fps "
    "from a video clip. Describe the continuous video sequence they "
    "represent, not each frame individually.\n\n"
)


class OpenAICompatBackend(VLMBackend):
    """OpenAI-compatible API backend for captioning with local VLMs.

    Extracts frames from video clips and sends them as multi-image
    prompts to any OpenAI-format chat completions endpoint.

    For image captioning, sends the image directly (no extraction).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.2-vision",
        api_key: str | None = None,
        timeout: int = 120,
        caption_fps: int = 1,
    ) -> None:
        """Initialize the OpenAI-compatible backend.

        Args:
            base_url: Base URL for the API endpoint (without /chat/completions).
                Default points to Ollama's OpenAI-compatible endpoint.
            model: Model name to use (must support vision/images).
            api_key: API key (optional — most local servers don't need one).
            timeout: Request timeout in seconds.
            caption_fps: Frames per second to extract from video clips.
                Higher = more frames = better motion capture, but slower.
                Default 1 FPS is standard for most captioning.
        """
        import requests as _requests  # noqa: F401 — verify available

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "not-needed"  # Many local servers ignore this
        self.timeout = timeout
        self.caption_fps = caption_fps

    def caption_video(self, path: Path, prompt: str) -> str:
        """Caption a video by extracting frames and sending as images.

        Workflow:
        1. Extract frames at caption_fps rate using ffmpeg
        2. Build multi-image prompt with frame prefix
        3. Send to OpenAI-compatible endpoint
        4. Clean up temp frames

        Args:
            path: Path to the video file.
            prompt: The captioning prompt.

        Returns:
            Generated caption text.
        """
        # Extract frames to a temp directory
        temp_dir = tempfile.mkdtemp(prefix="klippbok_frames_")
        try:
            frames = extract_frames(path, temp_dir, fps=self.caption_fps)

            if not frames:
                raise RuntimeError(f"No frames extracted from {path}")

            # Build the frame prefix
            prefix = _FRAME_PREFIX.format(
                count=len(frames),
                fps=self.caption_fps,
            )
            full_prompt = prefix + prompt

            # Build multi-image message content
            content = self._build_image_content(frames, full_prompt)

            return self._call_api(content)

        finally:
            # Always clean up temp frames
            shutil.rmtree(temp_dir, ignore_errors=True)

    def caption_image(self, path: Path, prompt: str) -> str:
        """Caption a single image.

        Args:
            path: Path to the image file.
            prompt: The captioning prompt.

        Returns:
            Generated caption text.
        """
        content = self._build_image_content([path], prompt)
        return self._call_api(content)

    def _build_image_content(
        self,
        image_paths: list[Path],
        prompt: str,
    ) -> list[dict]:
        """Build OpenAI-format message content with images.

        Encodes each image as a base64 data URL and combines with
        the text prompt in the OpenAI multi-modal format.

        Args:
            image_paths: Paths to image files to include.
            prompt: Text prompt to accompany the images.

        Returns:
            List of content blocks for the messages API.
        """
        content: list[dict] = []

        # Add images first, then the text prompt
        for img_path in image_paths:
            data = base64.b64encode(img_path.read_bytes()).decode("utf-8")

            # Determine MIME type
            suffix = img_path.suffix.lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }
            mime_type = mime_map.get(suffix, "image/jpeg")

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{data}",
                },
            })

        # Add the text prompt
        content.append({
            "type": "text",
            "text": prompt,
        })

        return content

    def _call_api(self, content: list[dict]) -> str:
        """Make the HTTP POST to the chat completions endpoint.

        Args:
            content: OpenAI-format message content blocks.

        Returns:
            The assistant's response text.

        Raises:
            RuntimeError: on API errors.
        """
        import requests

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"API error {response.status_code}: "
                f"{response.text[:300]}"
            )

        result = response.json()

        # Extract text from the standard OpenAI response format
        choices = result.get("choices", [])
        if not choices:
            raise RuntimeError(
                f"No choices in API response: {result}"
            )

        message = choices[0].get("message", {})
        text = message.get("content", "")
        return text.strip()
