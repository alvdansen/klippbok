"""Prompt templates for VLM captioning.

Each use case gets a tailored prompt that tells the VLM what to describe
and — critically — what to OMIT. The LoRA learns what you leave out.

- character LoRA: DON'T describe appearance (the LoRA teaches that)
- style LoRA: DON'T describe art style/medium (the LoRA teaches that)
- motion LoRA: DON'T describe identity, focus on movement
- object LoRA: DON'T describe the object, describe context

Captions should read like GENERATION PROMPTS — short, direct,
comma-separated phrases. Not prose, not narration, not film analysis.

Good: "close-up, Luna looks up at the open sky"
Bad:  "It's a close-up of Luna, shot from below, as she looks intently up at the open sky."

Framing (closeup, wide shot, etc.) leads as a tag when notable.
The anchor word is used as a natural name, not mechanically prepended.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Video captioning prompts
# ---------------------------------------------------------------------------

# Prompts use {anchor_word} and {secondary_anchors} placeholders.
# These are filled in by format_prompt() before being sent to the VLM.
# If no anchor word is set, the placeholders are stripped out.

VIDEO_PROMPT_GENERAL = """\
Write a short caption for this video clip in prompt style.
Use direct, comma-separated phrases — not prose or narration.
Start with the framing if notable (e.g. "close-up," "wide shot,").
Then describe who or what, the action, and the setting.
Good: "wide shot, a woman walks through a rain-soaked alley at night"
Bad: "The video shows a wide shot of a woman walking through an alley in the rain at night."
Do NOT start with "The video shows", "In this clip", or "It's a".
Do NOT describe color palettes, lighting mood, or cinematography."""

VIDEO_PROMPT_CHARACTER = """\
Write a short caption for this video clip in prompt style.{anchor_line}{secondary_line}
Use direct, comma-separated phrases — not prose or narration.
Start with the framing if notable (e.g. "close-up," "wide shot,").
Then describe what {subject} is doing and the setting.
Good: "close-up, {subject} looks up at the open sky"
Good: "medium shot, {subject} sits on a rooftop, legs dangling over the edge"
Bad: "It's a close-up of {subject}, shot from below, as they look intently up at the open sky."
Do NOT describe {subject}'s physical appearance, clothing, or features.
Do NOT describe color palettes, lighting mood, or cinematography.
The visual details are learned from the video itself — just describe the action and setting."""

VIDEO_PROMPT_STYLE = """\
Write a short caption for this video clip in prompt style.{style_anchor_line}{secondary_line}
Use direct, comma-separated phrases — not prose or narration.
Start with the framing if notable (e.g. "close-up," "wide shot,").
Then describe who is in the scene and what is happening.
Good: "wide shot, a girl walks through a neon-lit market"
Bad: "The scene captures a wide shot of a girl as she walks through a brightly lit market."
Do NOT describe the visual style, art direction, color grading, or lighting mood.
The style is learned from the video itself — just describe the content."""

VIDEO_PROMPT_MOTION = """\
Write a short caption for this video clip in prompt style.{anchor_line}{secondary_line}
Use direct, comma-separated phrases — not prose or narration.
Start with the framing if notable (e.g. "tracking shot," "close-up,").
Focus on how things MOVE — speed, direction, body mechanics, camera motion.
Good: "tracking shot, figure sprints down a corridor, camera following from behind"
Bad: "In this clip, a figure is shown sprinting rapidly down a long corridor."
Do NOT describe identity, appearance, or clothing.
Keep it short and focused on the dynamics."""

VIDEO_PROMPT_OBJECT = """\
Write a short caption for this video clip in prompt style.{anchor_line}{secondary_line}
Use direct, comma-separated phrases — not prose or narration.
Start with the framing if notable (e.g. "close-up," "wide shot,").
Describe the scene around {subject} — the setting, context, and what's happening.
Good: "close-up, {subject} resting on a wooden table, warm kitchen in the background"
Bad: "The clip shows a close-up of {subject}, which is resting on a wooden table in a warm kitchen."
Do NOT describe {subject}'s appearance or details.
The object's look is learned from the video — just describe the world around it."""

# ---------------------------------------------------------------------------
# Image captioning prompts
# ---------------------------------------------------------------------------

IMAGE_PROMPT_GENERAL = """\
Write a short caption for this image in prompt style.
Use direct, comma-separated phrases — not prose or narration.
Start with the framing if notable (e.g. "close-up," "wide shot,").
Then describe the subject, setting, and mood.
Do NOT start with "The image shows" or "This is".
Do NOT describe color palettes, lighting mood, or cinematography."""

IMAGE_PROMPT_CHARACTER = """\
Write a short caption for this image in prompt style.{anchor_line}{secondary_line}
Use direct, comma-separated phrases — not prose or narration.
Start with the framing if notable (e.g. "close-up," "portrait,").
Then describe what {subject} is doing and the setting.
Do NOT describe {subject}'s physical appearance, clothing, or features."""

IMAGE_PROMPT_STYLE = """\
Write a short caption for this image in prompt style.{style_anchor_line}{secondary_line}
Use direct, comma-separated phrases — not prose or narration.
Start with the framing if notable (e.g. "close-up," "wide shot,").
Describe the subject and what's happening.
Do NOT describe the visual style, art medium, color grading, or lighting mood."""

IMAGE_PROMPT_MOTION = """\
Write a short caption for this image in prompt style.{anchor_line}{secondary_line}
Use direct, comma-separated phrases — not prose or narration.
Start with the framing if notable (e.g. "close-up," "wide shot,").
Describe the implied motion or action.
Do NOT describe identity, appearance, or clothing."""

IMAGE_PROMPT_OBJECT = """\
Write a short caption for this image in prompt style.{anchor_line}{secondary_line}
Use direct, comma-separated phrases — not prose or narration.
Start with the framing if notable (e.g. "close-up," "wide shot,").
Describe the scene around {subject}.
Do NOT describe {subject}'s appearance."""

# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

VIDEO_PROMPTS: dict[str | None, str] = {
    None: VIDEO_PROMPT_GENERAL,
    "character": VIDEO_PROMPT_CHARACTER,
    "style": VIDEO_PROMPT_STYLE,
    "motion": VIDEO_PROMPT_MOTION,
    "object": VIDEO_PROMPT_OBJECT,
}

IMAGE_PROMPTS: dict[str | None, str] = {
    None: IMAGE_PROMPT_GENERAL,
    "character": IMAGE_PROMPT_CHARACTER,
    "style": IMAGE_PROMPT_STYLE,
    "motion": IMAGE_PROMPT_MOTION,
    "object": IMAGE_PROMPT_OBJECT,
}


def get_video_prompt(
    use_case: str | None = None,
    anchor_word: str | None = None,
    secondary_anchors: list[str] | None = None,
) -> str:
    """Get the appropriate video captioning prompt for a use case.

    When an anchor word is provided, it's woven into the prompt so the
    VLM uses it as the character/object's name naturally. Secondary
    anchors are additional tags the VLM should try to mention.

    Args:
        use_case: One of 'character', 'style', 'motion', 'object', or None.
        anchor_word: Primary trigger word — used as the subject's name.
        secondary_anchors: Additional tags to mention (e.g. ["vintage", "retro"]).

    Returns:
        The prompt string, ready to send to the VLM.

    Raises:
        ValueError: if use_case is not recognized.
    """
    if use_case not in VIDEO_PROMPTS:
        valid = ", ".join(repr(k) for k in VIDEO_PROMPTS if k is not None)
        raise ValueError(
            f"Unknown use_case: {use_case!r}. Valid options: {valid}, or None for general."
        )
    template = VIDEO_PROMPTS[use_case]
    return _fill_prompt(template, anchor_word, secondary_anchors)


def get_image_prompt(
    use_case: str | None = None,
    anchor_word: str | None = None,
    secondary_anchors: list[str] | None = None,
) -> str:
    """Get the appropriate image captioning prompt for a use case.

    Args:
        use_case: One of 'character', 'style', 'motion', 'object', or None.
            Unknown use cases fall back to the general prompt.
        anchor_word: Primary trigger word — used as the subject's name.
        secondary_anchors: Additional tags to mention.

    Returns:
        The prompt string, ready to send to the VLM.
    """
    template = IMAGE_PROMPTS.get(use_case, IMAGE_PROMPT_GENERAL)
    return _fill_prompt(template, anchor_word, secondary_anchors)


def _fill_prompt(
    template: str,
    anchor_word: str | None = None,
    secondary_anchors: list[str] | None = None,
) -> str:
    """Fill a prompt template with anchor word and secondary anchors.

    Handles the {subject}, {anchor_line}, and {secondary_line} placeholders.
    When no anchor is set, these are cleaned out so the prompt reads naturally.

    The anchor word is presented as the subject's name — the VLM should
    weave it into the caption naturally, not prepend/append it mechanically.

    Secondary anchors are things that may or may not be visible in the clip.
    The VLM should only mention them if it actually sees them.
    """
    subject = anchor_word or "the subject"

    if anchor_word:
        anchor_line = (
            f"\nThe subject's name is \"{anchor_word}\". "
            f"Use \"{anchor_word}\" naturally in the caption as their name."
        )
        # Style LoRAs: anchor is a style descriptor, not a character name
        style_anchor_line = (
            f"\nThe style is called \"{anchor_word}\". "
            f"Use \"{anchor_word}\" naturally in the caption where it fits."
        )
    else:
        anchor_line = ""
        style_anchor_line = ""

    if secondary_anchors:
        tags = ", ".join(f'"{t}"' for t in secondary_anchors)
        secondary_line = (
            f"\nThese words may be relevant: {tags}. "
            f"Only use them if you can actually see what they describe — "
            f"do not force them in."
        )
    else:
        secondary_line = ""

    result = template
    result = result.replace("{subject}", subject)
    result = result.replace("{anchor_line}", anchor_line)
    result = result.replace("{style_anchor_line}", style_anchor_line)
    result = result.replace("{secondary_line}", secondary_line)
    return result


def format_prompt(prompt: str, **variables: str) -> str:
    """Safely substitute template variables in a prompt string.

    Variables use {curly_brace} syntax. Only variables that appear in the
    prompt are substituted — extra variables are silently ignored, and
    missing variables are left as-is (no KeyError).

    This is intentionally NOT str.format() because we want missing
    variables to pass through unchanged rather than raising errors.

    Args:
        prompt: The prompt template with {variable} placeholders.
        **variables: Variable name → value mappings.

    Returns:
        The prompt with known variables substituted.

    Examples:
        >>> format_prompt("Describe {anchor_word} in detail", anchor_word="Luna")
        "Describe Luna in detail"
        >>> format_prompt("No variables here")
        "No variables here"
        >>> format_prompt("{missing} stays", other="ignored")
        "{missing} stays"
    """
    result = prompt
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", value)
    return result
