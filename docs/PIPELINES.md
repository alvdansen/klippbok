# Klippbok Pipelines

Which pipeline fits your training situation? This guide helps you pick the right path and shows you the commands to get there.

For detailed setup (provider config, threshold tuning, troubleshooting), see [WALKTHROUGH.md](WALKTHROUGH.md).

---

## Quick: which pipeline do I need?

| You have | You want | Pipeline |
|----------|----------|----------|
| Raw footage, full-length video | Character LoRA | [1. Character from raw footage](#pipeline-1-character-from-raw-footage-triage-first) |
| Pre-cut clips | Character LoRA | [2. Character from pre-cut clips](#pipeline-2-character-from-pre-cut-clips) |
| Clips showing a visual style | Style LoRA | [3. Style LoRA](#pipeline-3-style-lora) |
| Clips showing a motion pattern | Motion LoRA | [4. Motion LoRA](#pipeline-4-motion-lora) |
| Existing dataset, wrong specs | Fixed dataset | [5. Dataset cleanup](#pipeline-5-dataset-cleanup--re-normalization) |
| Raw footage, non-character concept | Object/setting LoRA | [6. Other concept (experimental)](#pipeline-6-other-concept-from-raw-footage-experimental) |

**Still unsure?** Two questions:
1. **Is your source already cut into short clips?** Yes → Pipeline 2-4. No → Pipeline 1 or 6.
2. **What are you training?** A character → 1 or 2. A style → 3. A motion → 4.

---

## Pipeline 1: Character from raw footage (triage-first)

**Train a character LoRA from long uncut video — movies, episodes, raw footage.**

### When to use this

- You have full-length video files, downloaded videos, or raw footage (minutes to hours long)
- You want clips of a specific character and need to skip everything else
- You have 1+ reference images of the character

Real-world example: 25 Breakfast at Tiffany's clips → triage found 162 Holly Golightly scenes out of ~1700 total. That's 90% less processing.

### What you need

- Source video file(s)
- A `concepts/` folder with reference image(s) in a `character/` subfolder
- An API key for captioning (Gemini, Replicate, or a local model via Ollama)

### The pipeline

```
raw videos
    │
    ▼
  triage ─────────► scene_triage_manifest.json
    │                   (review & edit)
    ▼
  ingest --triage ──► normalized clips (only matching scenes)
    │
    ├──► caption ───► .txt sidecar captions
    │
    ├──► extract ───► .png reference frames (for I2V)
    │
    ▼
  validate ─────────► ready for training
```

### Commands

**Step 1: Set up your concepts folder.** This is all it takes to tell Klippbok what to look for — drop a reference image (any screenshot, photo, or character sheet) into a named subfolder, and CLIP visual matching handles the rest.

```
concepts/
  character/
    ref.jpg             ← any image of your character — a screenshot is enough
```

You can use multiple reference images per concept to improve matching accuracy across different angles, lighting, and outfits. The subfolder name (`character/`) becomes the concept type.

```powershell
# 1. Triage — find scenes containing your character
python -m klippbok.video triage "C:\path\to\raw\videos" -s concepts/

# 2. (Optional) Review scene_triage_manifest.json, flip include: true/false

# 3. Filtered ingest — split only matching scenes
python -m klippbok.video ingest "C:\path\to\raw\videos" -o clips --triage scene_triage_manifest.json

# 4. Caption
python -m klippbok.video caption clips -p gemini -u character -a "Holly Golightly"

# 5. Extract reference frames (for I2V training)
python -m klippbok.video extract clips -o clips/references

# 6. Validate
python -m klippbok.dataset validate clips
```

Skip step 5 if you're training T2V (text-to-video) — no reference frames needed.

For provider setup and API keys, see [WALKTHROUGH.md — Step 3: Caption](WALKTHROUGH.md#step-3-caption--generate-captions-via-vlm).

### What you end up with

```
clips/
  scene001.mp4          ← normalized training clip (16fps, 720p, 4n+1 frames)
  scene001.txt          ← "Holly Golightly walks down the street, busy morning traffic."
  scene002.mp4
  scene002.txt
  ...
  references/
    scene001.png        ← first-frame reference (for I2V)
    scene002.png
```

### Organize (coming soon)

A dedicated organize step will structure this output for specific trainers (musubi-tuner, ai-toolkit). For now, the flat layout above works directly with most trainers.

---

## Pipeline 2: Character from pre-cut clips

**Train a character LoRA from clips that are already trimmed — client deliveries, previous projects, manually selected footage.**

### When to use this

- Your clips are already cut to individual scenes (under 30 seconds each)
- A human already curated which clips to use — no triage needed
- Clips might be wrong resolution, frame rate, or format for training

Real-world examples: Brand ambassador footage from a shoot, influencer content clips, product demos featuring a host.

### What you need

- A folder of pre-cut video clips
- An API key for captioning

### The pipeline

```
pre-cut clips
    │
    ▼
  scan ─────────────► report (what needs fixing)
    │
    ▼
  normalize ────────► training-spec clips (16fps, 720p)
    │
    ├──► caption ───► .txt sidecar captions
    │
    ├──► extract ───► .png reference frames (for I2V)
    │
    ▼
  validate ─────────► ready for training
```

### Commands

```powershell
# 1. Scan — see what needs normalizing
python -m klippbok.video scan "C:\path\to\clips"

# 2. Normalize — fix fps, resolution, frame count
python -m klippbok.video normalize "C:\path\to\clips" -o normalized

# 3. Caption
python -m klippbok.video caption normalized -p replicate -u character -a "[character name]"

# 4. Extract reference frames (for I2V)
python -m klippbok.video extract normalized -o normalized/references

# 5. Validate
python -m klippbok.dataset validate normalized
```

Skip step 4 for T2V training.

**Already have captions?** Skip step 3 entirely — normalize copies existing `.txt` sidecar files to the output folder automatically.

### What you end up with

Same structure as Pipeline 1 — normalized clips with captions and optional reference frames.

---

## Pipeline 3: Style LoRA

**Train a LoRA that captures a visual aesthetic — film noir, analog grain, anime studio style, retro VHS.**

### When to use this

- You've hand-picked clips that represent a specific visual style
- You care about the LOOK (color palette, grain, lighting, contrast), not specific subjects
- Training T2V only — no reference frames needed

Real-world examples: Film noir lighting from classic movies, Studio Ghibli's color palette, 80s VHS aesthetic, cinematic anamorphic lens look.

### What you need

- A folder of clips selected for their visual style
- An API key for captioning

### The pipeline

```
style clips
    │
    ▼
  normalize ────────► training-spec clips
    │
    ▼
  caption (-u style) ► .txt captions describing CONTENT (not style)
    │
    ▼
  validate ─────────► ready for training
```

### Commands

```powershell
# 1. Normalize
python -m klippbok.video normalize "C:\path\to\clips" -o normalized

# 2. Caption with style use-case
python -m klippbok.video caption normalized -p gemini -u style

# 3. Validate
python -m klippbok.dataset validate normalized
```

### Why no anchor word?

Style LoRAs don't use `-a` (anchor word). The model learns the style from the visual patterns across all clips — you don't need a trigger word.

### Why `-u style` matters

The `style` use-case tells the captioner to describe **what's happening** in the clip, not how it looks. "A woman walks down a rainy street" — not "A noir-lit scene with high contrast shadows." The model learns the visual style from the pixels; the caption teaches it what content to associate the style with.

### What you end up with

```
normalized/
  clip_001.mp4          ← training clip
  clip_001.txt          ← "A woman walks down a rainy street at night."
  clip_002.mp4
  clip_002.txt
```

No reference frames — this is T2V only.

---

## Pipeline 4: Motion LoRA

**Train a LoRA that captures specific movement patterns — camera orbits, slow motion, walk cycles, explosion dynamics.**

### When to use this

- You've selected clips for their movement, not their visual content
- You care about HOW things move, not what they look like
- Training T2V only

Real-world examples: Smooth camera orbit around a subject, slow-mo water splash, anime run cycle, drone flyover.

### What you need

- A folder of clips selected for their motion pattern
- An API key for captioning

### The pipeline

```
motion clips
    │
    ▼
  normalize ────────► training-spec clips
    │
    ▼
  caption (-u motion) ► .txt captions focused on movement
    │
    ▼
  validate ─────────► ready for training
```

### Commands

```powershell
# 1. Normalize
python -m klippbok.video normalize "C:\path\to\clips" -o normalized

# 2. Caption with motion use-case
python -m klippbok.video caption normalized -p gemini -u motion

# 3. Validate
python -m klippbok.dataset validate normalized
```

### Why `-u motion` matters

The `motion` use-case makes captions focus on **movement and camera behavior**: "The camera slowly orbits around the subject while they stand still." Not "A person stands in a park" — the caption needs to teach the model what the movement IS so it can reproduce it from text prompts.

### What you end up with

Same structure as Pipeline 3 — clips with motion-focused captions, no reference frames.

---

## Pipeline 5: Dataset cleanup / re-normalization

**Fix an existing dataset — wrong specs, inherited from another project, migrating between trainers.**

### When to use this

- You have a dataset that's already captioned but the videos are wrong resolution/fps
- You downloaded a community-shared dataset and need to adapt it to your specs
- You're migrating from one trainer to another and need to re-validate

Real-world examples: Inherited a dataset at 24fps and need 16fps for Wan, downloaded clips at 1080p and need 720p, switching from ai-toolkit to musubi-tuner format.

### What you need

- An existing dataset folder with clips (and hopefully captions)

### The pipeline

```
existing dataset
    │
    ▼
  scan ─────────────► what needs fixing
    │
    ▼
  normalize ────────► re-encoded clips (copies .txt sidecars)
    │
    ▼
  score ────────────► caption quality report (no API needed)
    │
    ▼
  validate ─────────► completeness check
```

### Commands

```powershell
# 1. Scan — see what's off
python -m klippbok.video scan "C:\path\to\dataset"

# 2. Normalize — fix specs, copies existing captions
python -m klippbok.video normalize "C:\path\to\dataset" -o fixed

# 3. Score captions (optional, no API needed)
python -m klippbok.video score fixed

# 4. Validate
python -m klippbok.dataset validate fixed
```

### Key detail: normalize preserves captions

When you normalize, existing `.txt` sidecar files are copied alongside the re-encoded clips. You don't lose your captions. If the captions are fine, you never need to re-caption.

The `score` command checks caption quality (length, structure, repetition) without calling any API — it's purely local analysis. Use it to spot low-quality captions in inherited datasets.

### What you end up with

A clean dataset with correct video specs and your existing captions preserved.

---

## Pipeline 6: Other concept from raw footage (experimental)

**Target objects, settings, or custom concepts from long uncut video. Same pipeline as Pipeline 1, but results need manual review.**

### When to use this

- You want to extract clips of a specific product, location, or object from raw footage
- You have reference images of the concept

Real-world examples: A specific product across marketing footage, a recurring location in a series, a particular prop or vehicle.

### What you need

Same as Pipeline 1, but with reference images in the appropriate concept subfolder:

```
concepts/
  object/
    product.jpg
  setting/
    office.jpg
```

### The pipeline

Same flow as Pipeline 1:

```
raw videos → triage → manifest review → ingest --triage → caption → extract → validate
```

### Commands

```powershell
# 1. Triage
python -m klippbok.video triage "C:\path\to\raw\videos" -s concepts/

# 2. REVIEW the manifest — this is critical for non-character concepts
#    Open scene_triage_manifest.json and check the matches

# 3. Filtered ingest
python -m klippbok.video ingest "C:\path\to\raw\videos" -o clips --triage scene_triage_manifest.json

# 4. Caption (use -u object for objects/products)
python -m klippbok.video caption clips -p gemini -u object -a "the product name"

# 5. Extract + validate
python -m klippbok.video extract clips -o clips/references
python -m klippbok.dataset validate clips
```

### Why this is experimental

CLIP-based triage works differently for non-character concepts:

- **Characters**: CLIP matches identity well — it recognizes the same person across angles, lighting, and distance. Production-ready.
- **Objects/settings**: CLIP matches visual similarity, not semantic identity. A "cat" reference might match scenes with similar framing or color palette, not just scenes containing a cat.

This means you'll get false positives. The triage still saves you a huge amount of manual work, but **you must review the manifest before ingesting**. Flip `include: false` on scenes that don't actually contain your concept.

### Cascade problem

If triage mismatches a scene, the anchor word gets baked into the caption for a clip that doesn't contain the concept. A wrong caption is worse than no caption. This is why manifest review matters — catch the mistakes before they enter your training data.

### Improvement roadmap

Better non-character triage accuracy is planned — this includes concept-type-specific matching strategies and confidence calibration. For now, the manual review step is your quality gate.

---

## What each tool does (quick reference)

| Tool | Command | What it does |
|------|---------|--------------|
| **scan** | `python -m klippbok.video scan <dir>` | Probe clips, report what needs normalizing. Read-only. |
| **ingest** | `python -m klippbok.video ingest <path> -o <out>` | Scene detect + split + normalize. Takes raw video, produces training clips. |
| **normalize** | `python -m klippbok.video normalize <dir> -o <out>` | Fix fps, resolution, frame count. Copies sidecars. |
| **triage** | `python -m klippbok.video triage <dir> -s <concepts>` | Match clips/scenes against reference images via CLIP. |
| **caption** | `python -m klippbok.video caption <dir> -p <provider>` | Generate `.txt` captions via VLM (Gemini, Replicate, Ollama). |
| **score** | `python -m klippbok.video score <dir>` | Check caption quality locally. No API needed. |
| **extract** | `python -m klippbok.video extract <dir> -o <out>` | Extract reference frames as PNG (first frame or best frame). |
| **audit** | `python -m klippbok.video audit <dir>` | Compare existing captions against fresh VLM output. |
| **validate** | `python -m klippbok.dataset validate <dir>` | Check dataset completeness — missing captions, orphaned files, quality. |
| **organize** | `python -m klippbok.dataset organize <dir> -o <out>` | Structure dataset for specific trainers (musubi, ai-toolkit). |

---

## Concept type reliability

How well does CLIP-based triage work for different concept types?

| Concept type | Triage reliability | Notes |
|--------------|-------------------|-------|
| **Character** | Ready | CLIP recognizes human identity well across angles, lighting, distance. Tested end-to-end on real data. |
| **Object** | Experimental | Matches visual similarity, not semantic identity. Manual manifest review required. |
| **Setting** | Experimental | Same limitation — matches framing/palette, not "this specific place." Review your results. |
| **Style** | Experimental | In development. Use Pipeline 3. |
| **Motion** | Not applicable | Motion LoRAs use hand-picked clips, not triage. Use Pipeline 4. |

For characters, trust the triage. For everything else, use it as a first pass and review the manifest before ingesting.
