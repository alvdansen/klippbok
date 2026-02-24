"""Caption quality scoring — heuristic analysis of .txt caption files.

Standalone scoring module that evaluates caption quality across four
dimensions. No API calls needed — works entirely from the text content.

Helps find weak captions in a batch so you can re-caption or manually
edit them before training. This is a curator's tool: it flags potential
issues, not definitive problems.

Scoring dimensions:
- **Length**: penalizes too-short (<50 chars) and too-long (>600 chars)
- **Temporal awareness**: looks for motion/change language (video captions
  with zero temporal words are suspicious)
- **Specificity**: penalizes vague phrases, rewards concrete spatial/
  lighting/color terms
- **Repetition**: detects repeated ngrams and VLM stutter patterns
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Scoring config — all thresholds and weights are configurable
# ---------------------------------------------------------------------------

@dataclass
class ScoringConfig:
    """Configuration for caption quality scoring.

    All thresholds and weights can be adjusted to match your dataset's
    characteristics. Defaults are tuned for Wan-style video training
    captions (80–400 chars, temporal awareness expected).
    """

    # Length scoring
    min_good_length: int = 80
    """Captions shorter than this get penalized."""

    max_good_length: int = 400
    """Captions longer than this get mildly penalized."""

    min_acceptable_length: int = 50
    """Below this length, score drops sharply."""

    max_acceptable_length: int = 600
    """Above this length, score drops sharply."""

    # Dimension weights (must sum to ~1.0 for intuitive overall score)
    weight_length: float = 0.25
    weight_temporal: float = 0.30
    weight_specificity: float = 0.25
    weight_repetition: float = 0.20

    # Specificity tuning
    vague_penalty: float = 0.15
    """How much each vague phrase costs (subtracted from 1.0)."""

    specific_bonus: float = 0.10
    """How much each specific term adds (capped at 1.0)."""


# ---------------------------------------------------------------------------
# Word lists for heuristic scoring
# ---------------------------------------------------------------------------

# Words that indicate temporal awareness (motion, change, sequence)
TEMPORAL_KEYWORDS: set[str] = {
    # Motion verbs
    "walks", "walking", "runs", "running", "moves", "moving",
    "turns", "turning", "spins", "spinning", "falls", "falling",
    "rises", "rising", "jumps", "jumping", "flies", "flying",
    "slides", "sliding", "drifts", "drifting", "sways", "swaying",
    "dances", "dancing", "flows", "flowing", "shifts", "shifting",
    # Camera motion
    "panning", "pans", "tilting", "tilts", "zooming", "zooms",
    "tracking", "dollying", "handheld", "steadicam",
    # Temporal words
    "gradually", "slowly", "quickly", "rapidly", "suddenly",
    "then", "before", "after", "while", "during", "eventually",
    "begins", "ends", "starts", "continues", "transitions",
    # Change words
    "changes", "transforms", "emerges", "appears", "disappears",
    "fades", "brightens", "darkens", "intensifies",
}

# Vague phrases that VLMs tend to produce (penalized)
VAGUE_PHRASES: list[str] = [
    "a beautiful scene",
    "a stunning view",
    "the video shows",
    "the clip shows",
    "in this video",
    "in this clip",
    "we can see",
    "can be seen",
    "a nice",
    "very beautiful",
    "really nice",
    "quite interesting",
    "overall vibe",
    "general atmosphere",
]

# Specific terms that indicate concrete description (rewarded)
SPECIFIC_KEYWORDS: set[str] = {
    # Spatial
    "foreground", "background", "left", "right", "center",
    "above", "below", "behind", "beside", "overhead",
    # Lighting
    "backlit", "silhouette", "rim light", "ambient",
    "overexposed", "underexposed", "harsh", "soft light",
    "golden hour", "daylight", "fluorescent", "neon",
    # Color
    "crimson", "azure", "emerald", "amber", "ivory",
    "muted", "saturated", "desaturated", "monochrome",
    "warm tones", "cool tones", "high contrast", "low contrast",
    # Composition
    "close-up", "wide shot", "medium shot", "aerial",
    "low angle", "high angle", "eye level", "dutch angle",
    "depth of field", "bokeh", "shallow focus",
}


# ---------------------------------------------------------------------------
# Score result
# ---------------------------------------------------------------------------

@dataclass
class CaptionScore:
    """Quality scores for a single caption.

    Each dimension is 0.0–1.0 (higher = better). The overall score
    is a weighted average of all dimensions.
    """

    path: Path | None = None
    """Path to the .txt file (None for inline scoring)."""

    caption: str = ""
    """The caption text that was scored."""

    length_score: float = 0.0
    """How appropriate the caption length is (0=too short/long, 1=ideal)."""

    temporal_score: float = 0.0
    """How much motion/temporal language appears (0=none, 1=rich)."""

    specificity_score: float = 0.0
    """How concrete vs vague the description is (0=vague, 1=specific)."""

    repetition_score: float = 0.0
    """How free of repetition (1=no repetition, 0=heavy repetition)."""

    overall: float = 0.0
    """Weighted average of all dimension scores."""

    issues: list[str] = field(default_factory=list)
    """Human-readable descriptions of detected problems."""


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _score_length(caption: str, config: ScoringConfig) -> tuple[float, list[str]]:
    """Score caption length — sweet spot is min_good to max_good chars.

    Returns (score, issues).
    """
    length = len(caption.strip())
    issues: list[str] = []

    if length == 0:
        return 0.0, ["Empty caption"]

    if length < config.min_acceptable_length:
        issues.append(f"Very short ({length} chars, minimum {config.min_acceptable_length})")
        return 0.2, issues

    if length < config.min_good_length:
        # Linear ramp from 0.4 at min_acceptable to 1.0 at min_good
        ratio = (length - config.min_acceptable_length) / (config.min_good_length - config.min_acceptable_length)
        score = 0.4 + 0.6 * ratio
        issues.append(f"Short ({length} chars, target {config.min_good_length}+)")
        return score, issues

    if length <= config.max_good_length:
        return 1.0, issues

    if length <= config.max_acceptable_length:
        # Linear ramp down from 1.0 at max_good to 0.6 at max_acceptable
        ratio = (length - config.max_good_length) / (config.max_acceptable_length - config.max_good_length)
        score = 1.0 - 0.4 * ratio
        issues.append(f"Long ({length} chars, target under {config.max_good_length})")
        return score, issues

    issues.append(f"Very long ({length} chars, maximum {config.max_acceptable_length})")
    return 0.4, issues


def _score_temporal(caption: str, config: ScoringConfig) -> tuple[float, list[str]]:
    """Score temporal awareness — video captions should describe motion/change.

    Returns (score, issues).
    """
    words = set(caption.lower().split())
    matches = words & TEMPORAL_KEYWORDS
    issues: list[str] = []

    if not matches:
        issues.append("No temporal/motion language detected")
        return 0.0, issues

    # Score ramps up: 1 match = 0.3, 2 = 0.5, 3 = 0.7, 4+ = 0.85, 6+ = 1.0
    count = len(matches)
    if count == 1:
        return 0.3, issues
    elif count == 2:
        return 0.5, issues
    elif count == 3:
        return 0.7, issues
    elif count <= 5:
        return 0.85, issues
    else:
        return 1.0, issues


def _score_specificity(caption: str, config: ScoringConfig) -> tuple[float, list[str]]:
    """Score specificity — penalize vague phrases, reward concrete terms.

    Returns (score, issues).
    """
    lower = caption.lower()
    issues: list[str] = []

    # Start at 0.5 (neutral) and adjust from there
    score = 0.5

    # Penalize vague phrases
    vague_found = []
    for phrase in VAGUE_PHRASES:
        if phrase in lower:
            score -= config.vague_penalty
            vague_found.append(phrase)

    if vague_found:
        issues.append(f"Vague phrases: {', '.join(repr(p) for p in vague_found[:3])}")

    # Reward specific terms
    specific_found = 0
    for term in SPECIFIC_KEYWORDS:
        if term in lower:
            score += config.specific_bonus
            specific_found += 1

    # Clamp to [0.0, 1.0]
    score = max(0.0, min(1.0, score))

    return score, issues


def _score_repetition(caption: str, config: ScoringConfig) -> tuple[float, list[str]]:
    """Score repetition — detect repeated ngrams and VLM stutter patterns.

    Returns (score, issues). 1.0 = no repetition, 0.0 = heavy repetition.
    """
    words = caption.lower().split()
    issues: list[str] = []

    if len(words) < 4:
        # Too short to meaningfully check for repetition
        return 1.0, issues

    # Check for repeated trigrams (3-word sequences)
    trigrams: dict[str, int] = {}
    for i in range(len(words) - 2):
        trigram = " ".join(words[i:i+3])
        trigrams[trigram] = trigrams.get(trigram, 0) + 1

    repeated = {k: v for k, v in trigrams.items() if v > 1}

    if not repeated:
        return 1.0, issues

    # Score based on how many trigrams are repeated
    total_trigrams = len(words) - 2
    repeated_count = sum(v - 1 for v in repeated.values())
    repeat_ratio = repeated_count / total_trigrams

    if repeat_ratio > 0.3:
        issues.append(f"Heavy repetition ({repeated_count} repeated trigrams)")
        return 0.2, issues
    elif repeat_ratio > 0.15:
        issues.append(f"Some repetition ({repeated_count} repeated trigrams)")
        return 0.5, issues
    elif repeat_ratio > 0.05:
        return 0.8, issues
    else:
        return 0.95, issues


def score_caption(
    caption: str,
    config: ScoringConfig | None = None,
) -> CaptionScore:
    """Score a single caption across all quality dimensions.

    Args:
        caption: The caption text to evaluate.
        config: Scoring configuration. If None, uses defaults.

    Returns:
        CaptionScore with per-dimension scores, overall score, and issues.
    """
    if config is None:
        config = ScoringConfig()

    length_score, length_issues = _score_length(caption, config)
    temporal_score, temporal_issues = _score_temporal(caption, config)
    specificity_score, specificity_issues = _score_specificity(caption, config)
    repetition_score, repetition_issues = _score_repetition(caption, config)

    all_issues = length_issues + temporal_issues + specificity_issues + repetition_issues

    overall = (
        config.weight_length * length_score
        + config.weight_temporal * temporal_score
        + config.weight_specificity * specificity_score
        + config.weight_repetition * repetition_score
    )

    return CaptionScore(
        caption=caption,
        length_score=length_score,
        temporal_score=temporal_score,
        specificity_score=specificity_score,
        repetition_score=repetition_score,
        overall=overall,
        issues=all_issues,
    )


def score_directory(
    directory: str | Path,
    config: ScoringConfig | None = None,
) -> list[CaptionScore]:
    """Score all .txt caption files in a directory.

    Args:
        directory: Directory containing .txt caption files.
        config: Scoring configuration. If None, uses defaults.

    Returns:
        List of CaptionScore for each .txt file found, sorted by
        overall score (worst first) for easy review.

    Raises:
        FileNotFoundError: if the directory doesn't exist.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    scores: list[CaptionScore] = []
    for txt_file in sorted(directory.glob("*.txt")):
        caption = txt_file.read_text(encoding="utf-8").strip()
        result = score_caption(caption, config)
        result.path = txt_file
        scores.append(result)

    # Sort worst-first for easy review
    scores.sort(key=lambda s: s.overall)
    return scores


def format_score_report(scores: list[CaptionScore]) -> str:
    """Format a human-readable score report.

    Sorted worst-first so you see the problems at the top.

    Args:
        scores: List of CaptionScore results.

    Returns:
        Formatted report string.
    """
    if not scores:
        return "No captions to score."

    lines: list[str] = []
    lines.append(f"Caption Quality Report ({len(scores)} files)")
    lines.append("=" * 60)

    # Summary stats
    overall_scores = [s.overall for s in scores]
    avg = sum(overall_scores) / len(overall_scores)
    worst = min(overall_scores)
    best = max(overall_scores)
    lines.append(f"Average: {avg:.2f}  Worst: {worst:.2f}  Best: {best:.2f}")
    lines.append("")

    # Individual scores (worst first — scores list is already sorted)
    for score in scores:
        name = score.path.name if score.path else "(inline)"
        preview = score.caption[:60].replace("\n", " ")
        if len(score.caption) > 60:
            preview += "..."

        lines.append(f"  {score.overall:.2f}  {name}")
        lines.append(f"        L={score.length_score:.1f} T={score.temporal_score:.1f} "
                      f"S={score.specificity_score:.1f} R={score.repetition_score:.1f}")
        if score.issues:
            for issue in score.issues:
                lines.append(f"        ! {issue}")
        lines.append(f"        \"{preview}\"")
        lines.append("")

    return "\n".join(lines)
