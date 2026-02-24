"""Zero-shot content filters for triage.

Uses CLIP's text-image matching to detect frames that are useless
for training: title cards, credits, text overlays, black frames, etc.

CLIP is already loaded during triage, so this adds zero extra cost
beyond encoding a few text prompts (instant) and comparing them
against frame embeddings we already have.

How it works:
    1. Encode a set of text prompts describing unwanted content
    2. For each clip's sampled frames, compare against these prompts
    3. If the BEST frame-to-prompt similarity exceeds a threshold,
       flag the clip as that content type

This is intentionally conservative — false negatives (missing a
title card) are harmless, but false positives (rejecting a good
training clip) waste data.
"""

from __future__ import annotations

import numpy as np

from klippbok.triage.embeddings import CLIPEmbedder


# Text prompts for detecting non-training content.
# Each category has multiple prompts to cover visual variations.
# The detector takes the MAX similarity across all prompts in a category.
TEXT_OVERLAY_PROMPTS = [
    "a movie title card with large text",
    "opening credits with text on a dark background",
    "closing credits scrolling text",
    "chapter title text overlay on screen",
    "movie poster with title text",
    "white text on black background",
    "end credits rolling text",
    "a title sequence with stylized text",
    "intertitle card with text",
]

# Threshold for text overlay detection.
# CLIP text-image similarity is typically lower than image-image,
# so this is lower than the concept matching threshold.
# Tuned conservatively — rather miss a title card than reject a scene.
DEFAULT_TEXT_OVERLAY_THRESHOLD = 0.27


def detect_text_overlays(
    embedder: CLIPEmbedder,
    frame_embeddings: list[np.ndarray],
    threshold: float = DEFAULT_TEXT_OVERLAY_THRESHOLD,
    _prompt_embeddings: list[np.ndarray] | None = None,
) -> tuple[bool, float]:
    """Check if sampled frames contain text overlays or title cards.

    Compares frame embeddings against text prompts describing common
    non-training content. Returns True if any frame strongly matches.

    Args:
        embedder: CLIPEmbedder instance (already loaded).
        frame_embeddings: CLIP embeddings of sampled clip frames.
        threshold: Similarity threshold for detection.
        _prompt_embeddings: Pre-computed prompt embeddings (optimization
            for batch processing — encode prompts once, reuse for all clips).

    Returns:
        Tuple of (is_text_overlay, best_score).
        is_text_overlay is True if best_score >= threshold.
    """
    if not frame_embeddings:
        return False, 0.0

    # Encode text prompts (or use pre-computed)
    if _prompt_embeddings is None:
        _prompt_embeddings = embedder.encode_texts(TEXT_OVERLAY_PROMPTS)

    # Find the highest similarity between any frame and any prompt
    best_score = 0.0
    for frame_emb in frame_embeddings:
        for prompt_emb in _prompt_embeddings:
            score = float(np.dot(frame_emb, prompt_emb))
            if score > best_score:
                best_score = score

    return best_score >= threshold, best_score


def build_prompt_cache(embedder: CLIPEmbedder) -> list[np.ndarray]:
    """Pre-compute text overlay prompt embeddings for batch use.

    Call this once before processing multiple clips, then pass
    the result to detect_text_overlays via _prompt_embeddings.

    Args:
        embedder: CLIPEmbedder instance.

    Returns:
        List of normalized text embeddings for TEXT_OVERLAY_PROMPTS.
    """
    return embedder.encode_texts(TEXT_OVERLAY_PROMPTS)
