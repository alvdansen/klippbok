"""Caption pipeline data models.

Pydantic v2 models for caption configuration, results, and audit output.
These models are used by the captioner orchestrator and VLM backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CaptionConfig(BaseModel):
    """Configuration for the captioning pipeline.

    Controls which VLM backend to use, retry behavior, and output settings.
    API keys are read from environment variables by default.
    """
    model_config = ConfigDict(frozen=True)

    provider: Literal["gemini", "replicate", "openai"] = "gemini"
    """Which VLM backend to use for generating captions.
    'openai' uses any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio)."""

    use_case: Literal["character", "style", "motion", "object"] | None = None
    """Dataset use case — determines which prompt template to use.
    If None, uses the general-purpose prompt."""

    anchor_word: str | None = None
    """Primary trigger word — used as the character/object's name in the caption.
    The VLM is told to use this as the subject's name, not just a tag.
    Example: anchor_word='Luna' → 'Luna looks up at the sky with worried eyes'"""

    secondary_anchors: list[str] | None = None
    """Additional tags the VLM should try to mention when relevant.
    Example: ['vintage', 'retro'] → the caption may include these
    if the VLM sees something that matches."""

    overwrite: bool = False
    """Whether to overwrite existing .txt caption files.
    If False, clips with existing captions are skipped."""

    api_key: str | None = None
    """API key for the VLM provider. If None, reads from environment:
    - Gemini: GEMINI_API_KEY
    - Replicate: REPLICATE_API_TOKEN"""

    timeout: int = 120
    """Request timeout in seconds per caption."""

    max_retries: int = 5
    """Maximum retry attempts for failed requests."""

    between_request_delay: float = 10.0
    """Seconds to wait between requests (rate limiting).
    Gemini free tier: ~20 req/min → 10s is safe.
    Replicate: 2s is usually fine."""

    audit_mode: Literal["report_only", "save_audit"] = "report_only"
    """For audit command: report_only prints to console,
    save_audit writes .audit.txt files alongside existing captions."""

    custom_prompt: str | None = None
    """Custom prompt that overrides the use-case prompt entirely.
    When set, this prompt is sent to the VLM instead of the built-in
    use-case templates. Useful for one-off experiments or specialized
    datasets that don't fit the standard use cases."""

    gemini_model: str = "gemini-2.5-flash"
    """Gemini model to use for captioning."""

    replicate_model: str = "google/gemini-2.5-flash"
    """Replicate model to use for captioning (same model lora-gym uses)."""

    openai_base_url: str = "http://localhost:11434/v1"
    """Base URL for OpenAI-compatible API endpoint.
    Default points to Ollama's OpenAI-compatible endpoint."""

    openai_model: str = "llama3.2-vision"
    """Model name for OpenAI-compatible backend.
    Must be a vision model that accepts image inputs."""

    caption_fps: int = 1
    """Frame sampling rate for captioning (frames per second).
    Controls how many frames are extracted from each video clip.
    - For Gemini: passed to videoMetadata.fps for server-side sampling.
    - For local backends: controls ffmpeg frame extraction rate.
    Default 1 FPS is standard. Increase to 2-4 for motion-focused datasets."""


class CaptionResult(BaseModel):
    """Result of captioning a single video clip.

    Records what happened — success or failure, timing, output.
    """
    model_config = ConfigDict(frozen=True)

    path: Path
    """Path to the video clip that was captioned."""

    caption: str = ""
    """The generated caption text (empty if failed)."""

    provider: str = ""
    """Which VLM backend produced this caption."""

    duration: float = 0.0
    """How long the API call took in seconds."""

    success: bool = True
    """Whether captioning succeeded."""

    error: str = ""
    """Error message if captioning failed."""

    skipped: bool = False
    """True if this clip was skipped (existing caption, not overwriting)."""


class AuditResult(BaseModel):
    """Result of auditing a single caption against VLM output.

    Compares the existing human/previous caption with a fresh VLM caption.
    """
    model_config = ConfigDict(frozen=True)

    path: Path
    """Path to the video clip."""

    existing_caption: str
    """The current caption from the .txt file."""

    vlm_caption: str
    """The fresh caption from the VLM."""

    recommendation: Literal["keep", "review"] = "review"
    """Whether the existing caption should be kept or reviewed.
    'review' means the VLM suggested something significantly different."""

    provider: str = ""
    """Which VLM backend produced the comparison caption."""
