"""Default values and constants for Klippbok data config.

This module defines the "ground truth" defaults that the schema uses.
Every default lives here so there's exactly one place to update them.
"""

# ─── Wan Model Constants ───
# These come from the Wan 2.2 architecture and training data documentation.

WAN_TRAINING_FPS: int = 16
"""Wan models were trained at 16 FPS. This is the default for all Klippbok configs."""

VALID_RESOLUTIONS: set[int] = {480, 720}
"""Resolution tiers supported by Wan models (height in pixels)."""

UMT5_MAX_TOKENS: int = 512
"""Maximum token count for the UMT5 text encoder used by Wan models."""


def valid_frame_counts(max_frames: int = 161) -> set[int]:
    """Generate valid frame counts for Wan's 3D causal VAE.

    The VAE requires frame counts of the form 4n+1 (1, 5, 9, 13, ..., 81, ...).
    This is because the temporal compression factor is 4, and the causal
    convolution needs one extra frame as the "seed" frame.

    Args:
        max_frames: Upper bound for generated frame counts. Default 161
            covers well beyond the typical training range (81 frames).

    Returns:
        Set of valid frame counts up to max_frames.
    """
    return {4 * n + 1 for n in range(max_frames // 4 + 1)}


# Pre-compute for use in validators
VALID_FRAME_COUNTS: set[int] = valid_frame_counts()

VALID_USE_CASES: set[str] = {"character", "style", "motion", "object"}
"""Dataset use cases that inform captioning strategy.

- character: omit appearance details (LoRA teaches appearance)
- style: omit art medium/aesthetic descriptors (LoRA teaches the look)
- motion: omit identity, focus on movement description
- object: omit the object, describe context and interaction
"""

VALID_UPSCALE_POLICIES: set[str] = {"never", "warn"}
"""How to handle clips below target resolution.

- never: reject clips that would need upscaling (Wan was trained on downscale-only data)
- warn: allow but warn — useful during dataset exploration
"""

VALID_SAR_POLICIES: set[str] = {"auto_correct", "reject"}
"""How to handle non-square pixel aspect ratios (SAR).

- auto_correct: fix SAR by resampling to square pixels (what most tools should do)
- reject: error on non-square SAR (strict mode)
"""

VALID_DOWNSCALE_METHODS: set[str] = {"lanczos", "bicubic", "bilinear", "area"}
"""Scaling algorithm for downscaling clips above target resolution.

- lanczos: sharp, high quality, best for significant downscaling. Standard for training data. Default.
- bicubic: smooth, slightly softer than lanczos. Fewer ringing artifacts on fine detail.
- bilinear: fast, softer results. Not recommended for training data.
- area: averages source pixels. Good for very large downscale factors (e.g. 4K to 480p).
"""

VALID_TEXT_FORMATS: set[str] = {"txt", "jsonl"}
"""Caption file formats.

- txt: one .txt sidecar file per video clip (default, most common)
- jsonl: all captions in a single .jsonl file (consolidated format)
"""

VALID_REFERENCE_SOURCES: set[str] = {"first_frame", "folder", "none"}
"""Where reference images come from.

- first_frame: auto-extract first frame from each video clip
- folder: look in a specified folder for stem-matched images
- none: no reference images (T2V training, or user provides them later)
"""

VALID_BUCKETING_DIMENSIONS: set[str] = {"aspect_ratio", "frame_count", "resolution"}
"""Properties that clips can be grouped by for batch construction."""
