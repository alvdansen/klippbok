# Captioning Methodology

How Klippbok generates training captions — and why it matters.

---

## The core problem

Captions are one of the most important control signals in video LoRA training, and most people get them wrong. The most common mistakes:

1. **Too verbose** — cinematic analysis ("The scene opens with a dramatic wide shot...") instead of generation prompts
2. **Describing what the LoRA should learn** — if you're training a character LoRA and your captions describe the character's appearance, the model learns to associate that text with the appearance rather than learning the visual pattern
3. **Prose instead of prompt-style** — training captions should read like the prompts people will use at inference time

Klippbok's captioning system encodes a methodology developed over three years of production LoRA training. Every prompt template, every use-case rule, and every default comes from shipping real models.

**Every prompt is fully customizable.** Klippbok ships with battle-tested defaults, but you're not locked in. You can override the system prompt entirely with `--custom-prompt` on the CLI or `custom_prompt=` in the config — pass any instruction you want directly to the VLM. The built-in use-case templates (`character`, `style`, `motion`, `object`) are a starting point. If your dataset needs something different, write your own prompt and Klippbok will use it as-is.

---

## The golden rule: don't describe what the LoRA teaches

This is the single most important captioning principle. The LoRA learns visual patterns from the video pixels. The caption tells the model what *isn't* the LoRA's job — what context and conditions to associate with those visuals.

| Training type | What the LoRA teaches | What captions describe |
|---|---|---|
| **Character** | What the character looks like | What they're doing, where they are |
| **Style** | The visual aesthetic | What's happening in the scene |
| **Motion** | How things move | The dynamics, speed, direction |
| **Object** | What the object looks like | The scene around the object |

If you're training a character LoRA and your caption says "a woman with blue hair and red eyes," the model learns to produce blue hair when it sees that text — not when it recognizes the character. The LoRA becomes fragile and prompt-dependent instead of learning a robust visual identity.

---

## Prompt-style captions

Training captions should read like generation prompts. Short, factual, comma-separated phrases.

**Good:**
```
close-up, Luna looks up at the open sky
```

**Bad:**
```
It's a close-up of Luna, shot from below, as she looks intently up at the open sky.
```

**Why:** At inference time, users write prompts like the "good" example. If your training captions match that style, the model learns to respond to the same kind of language it will see in production.

Rules:
- Start with framing when notable: "close-up,", "wide shot,", "tracking shot,"
- Comma-separated phrases, not prose
- Never start with "The video shows", "In this clip", or "It's a"
- Never describe color palettes, lighting mood, or cinematography (unless training a style LoRA that specifically targets those)

---

## Use-case prompts

Klippbok has four specialized prompt templates, each tailored to omit what the LoRA should learn:

### character

For character LoRAs. Captions describe action and setting. Explicitly tells the VLM:

> *Do NOT describe the subject's physical appearance, clothing, or features. The visual details are learned from the video itself — just describe the action and setting.*

```
python -m klippbok.video caption clips -p gemini -u character -a "Luna"
```

Example output: `"close-up, Luna looks up at the open sky"` — no description of what Luna looks like.

### style

For style LoRAs. Captions describe content and composition. Explicitly tells the VLM:

> *Do NOT describe the visual style, art direction, color grading, or lighting mood. The style is learned from the video itself — just describe the content.*

```
python -m klippbok.video caption clips -p gemini -u style
```

Example output: `"wide shot, a girl walks through a neon-lit market"` — no description of the artistic style.

**Note:** Style LoRAs typically don't use anchor words. The model learns the style from pixels, not from a trigger word.

### motion

For motion LoRAs. Captions focus on movement and dynamics. Explicitly tells the VLM:

> *Focus on how things MOVE — speed, direction, body mechanics, camera motion. Do NOT describe identity, appearance, or clothing.*

```
python -m klippbok.video caption clips -p gemini -u motion
```

Example output: `"tracking shot, figure sprints down a corridor, camera following from behind"` — pure dynamics.

### object

For object LoRAs. Captions describe the scene around the object. Explicitly tells the VLM:

> *Describe the scene around the subject — the setting, context, and what's happening. Do NOT describe the subject's appearance or details.*

```
python -m klippbok.video caption clips -p gemini -u object -a "the vase"
```

Example output: `"close-up, the vase resting on a wooden table, warm kitchen in the background"` — no description of the vase itself.

---

## Anchor words

The anchor word (`-a` flag) is the subject's name. It's woven into the prompt template so the VLM uses it naturally.

```
python -m klippbok.video caption clips -u character -a "Luna"
```

The VLM is told: *"The subject's name is 'Luna'. Use 'Luna' naturally in the caption as their name."*

This produces captions like `"Luna walks down the street"` instead of `"a woman walks down the street"`.

**Important:**
- Use natural names, not token-style triggers. `"Luna"` not `"sks_luna"` or `"<luna>"`
- Multi-word names need quotes: `-a "Holly Golightly"`
- The anchor word appears because the VLM naturally wrote it — it's not mechanically prepended
- If the VLM's caption doesn't start with the anchor word, Klippbok prepends it as a safety net: `"Luna, a figure walks..."`

---

## Secondary anchors

Secondary anchors (`-t` flag) are additional tags the VLM should try to mention — but only if it actually sees what they describe.

```
python -m klippbok.video caption clips -u character -a "Luna" -t vintage retro
```

The VLM is told: *"These words may be relevant: 'vintage', 'retro'. Only use them if you can actually see what they describe — do not force them in."*

This is useful for training data that has recurring themes or settings you want the model to learn to associate with the subject.

---

## Caption dropout

This is a training concept, not a captioning one, but it's important context for why Klippbok's captions are deliberately minimal.

Caption dropout means randomly dropping the caption for a percentage of training steps (typically 10-20%). When the caption is dropped, the model must rely entirely on visual signals (for I2V: the reference image) to understand what to generate.

This makes the LoRA more robust:
- Without dropout, the model can "cheat" by relying on text cues instead of learning visual patterns
- With dropout, the model must learn the visual pattern itself — the caption is a helpful signal, not a crutch

Klippbok's minimal, action-focused captions complement this: even when the caption is present, it doesn't duplicate information the model should learn from pixels.

---

## Provider comparison

| Provider | Caption quality | Speed | Cost | Setup |
|----------|----------------|-------|------|-------|
| **Gemini** | Best — short, factual | Fast | Free tier | `GEMINI_API_KEY` |
| **Replicate** | Best — short, factual | Medium | ~$0.01/clip | `REPLICATE_API_TOKEN` |
| **Ollama** (local) | Good — tends verbose | Varies | Free | Ollama + vision model |

**Gemini and Replicate** produce the short, factual, prompt-style captions that train best. They follow instructions well and naturally produce the kind of output Klippbok's prompts ask for.

**Local models** (Ollama with llama3.2-vision, gemma3, etc.) are functional but tend toward verbose, cinematic-analysis-style captions. They require more prompt tuning to produce training-quality output. Marked experimental.

### Local model recommendations

| Model | Size | Quality | Speed | Notes |
|-------|------|---------|-------|-------|
| `llama3.2-vision` | 4.9GB | Good | Fast | Recommended starting point |
| `llama3.2-vision:90b` | ~55GB | Best | Slow | Needs 48GB+ VRAM |
| `gemma3:27b` | 17GB | Very good | Medium | Strong visual understanding |

### API key setup

**Gemini:** Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and create a key. AI Studio keys come with the Generative Language API pre-enabled. If you use a key from Google Cloud Console instead, you'll need to manually enable the Generative Language API in your project.

```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

**Replicate:**

```powershell
$env:REPLICATE_API_TOKEN = "your-token-here"
```

---

## Auditing and scoring

### score — offline quality check

```
python -m klippbok.video score clips/
```

Checks caption quality using local heuristics (no API calls):
- Length (too short, too long)
- Repetition
- Common problems (starts with "The video shows", etc.)

### audit — VLM comparison

```
python -m klippbok.video audit clips/ -p gemini
```

Generates fresh captions and compares them against your existing ones. Useful for:
- Reviewing community-shared datasets
- Checking captions after manual edits
- Comparing provider output quality

---

## Custom prompts

For specialized datasets that don't fit the standard use cases, you can override the prompt entirely:

```python
from klippbok.caption.models import CaptionConfig
from klippbok.caption import caption_clips

config = CaptionConfig(
    provider="gemini",
    custom_prompt="Describe only the hand gestures and finger positions visible in this clip.",
)
caption_clips(directory, config)
```

Custom prompts bypass the use-case template system entirely. The exact text you provide is sent to the VLM.
