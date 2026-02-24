"""Abstract base class for VLM caption backends.

Every backend (Gemini, Replicate, future providers) implements this
interface. The captioner orchestrator works with the base class,
so adding a new provider doesn't touch the orchestration code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class VLMBackend(ABC):
    """Abstract VLM backend for generating captions.

    Subclasses must implement caption_video() and caption_image().
    Each backend handles its own authentication, API format, and retry logic.
    """

    @abstractmethod
    def caption_video(self, path: Path, prompt: str) -> str:
        """Generate a caption for a video clip.

        Args:
            path: Path to the video file.
            prompt: The captioning prompt (from prompts.py).

        Returns:
            The generated caption text.

        Raises:
            Exception: on API errors, timeouts, etc.
        """
        ...

    @abstractmethod
    def caption_image(self, path: Path, prompt: str) -> str:
        """Generate a caption for an image.

        Args:
            path: Path to the image file.
            prompt: The captioning prompt.

        Returns:
            The generated caption text.

        Raises:
            Exception: on API errors, timeouts, etc.
        """
        ...
