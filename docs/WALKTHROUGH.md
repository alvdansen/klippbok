# Klippbok Walkthrough

Working document — commands tested on real data, only steps that work are recorded here.

**Tip for Windows users:** If PowerShell breaks a long command across lines, `cd` into your dataset folder first, then use `.` as the directory. Short flags (`-p`, `-u`, `-a`) also help keep commands compact.

---

## Two paths: ingest-first vs triage-first

Before you start, choose the path that matches your situation:

**Path A — Ingest first (process everything):** You want all clips from your video(s), or you don't have character reference images to filter your characters by. Ingest splits every scene into clips, then you can optionally triage and organize them afterward.

```
raw videos → ingest (split all) → triage (match clips) → caption → train
```

**Path B — Triage first (target specific content):** You have reference images of the character/object/setting you care about, and your source videos are long (full-length films, raw footage). You want to skip irrelevant scenes entirely — no point splitting 1700 clips when you only need 162. Triage runs on the raw videos first, produces a manifest of matching scenes, then filtered ingest splits only those scenes.

```
raw videos → triage (scene-level matching) → scene_triage_manifest.json
scene_triage_manifest.json → ingest --triage (split only matching scenes) → caption → train
```

**Why two paths?** Ingesting a 2-hour movie produces hundreds of clips. If you're training a character LoRA, most of those clips don't contain your character. Triage-first avoids wasting time splitting, normalizing, and captioning scenes you'll throw away. But if you want everything (or your clips are already pre-cut), ingest-first is simpler.

**How it decides:** When you run triage on a directory, Klippbok probes each video's duration. Short videos (<30s) are treated as pre-cut clips — it samples a few frames from each and matches. Long videos (>=30s) are treated as raw footage — it detects scenes first, samples 1-2 frames per scene, and matches at the scene level. This happens automatically, no flags needed.

---

## Scenario 1: Pre-cut clips WITHOUT captions (I2V character LoRA)

**Starting point:** A folder of pre-cut video clips (.mov, .mp4, etc.) with NO captions. Clips are oversized/wrong fps and need normalizing. You want them captioned and reference frames extracted for I2V training.

### Step 1: Scan — see what you have

```
cd "C:\path\to\your\clips"
python -m klippbok.video scan .
```

This probes every video file and reports what needs re-encoding. No files are modified.

The report shows your current target settings and how to change them:
```
  Normalize will convert to: 16fps, 720p
  To use a different fps:        add --fps 24 to this command
  To use a different resolution:  create a klippbok_data.yaml with a
                                  video: section (see examples/ folder)
```

Use `-v` for the full per-clip breakdown: `python -m klippbok.video scan . -v`

### Step 2: Normalize — downscale, fix fps, trim frames

```
python -m klippbok.video normalize . -o "C:\path\to\output"
```

This:
- Downscales to 720p (height), maintains aspect ratio
- Re-encodes to 16fps (Wan's training frame rate)
- Trims to nearest valid 4n+1 frame count
- Keeps the same file format as your source (.mov stays .mov, .mp4 stays .mp4)
- Copies sidecar files (.txt captions, .json) to the output folder if present

To force a specific output format: `--format .mov` or `--format .mp4`

Output clips are ready for training. Source files are never modified.

### Step 3: Caption — generate captions via VLM

`cd` into your normalized output folder, then run one of these:

**Replicate** (cloud API, pay-per-use):
```
$env:REPLICATE_API_TOKEN = "your-token-here"
python -m klippbok.video caption . -p replicate -u character -a [character name]
```

**Gemini** (cloud API, free tier available):
```
$env:GEMINI_API_KEY = "your-key-here"
python -m klippbok.video caption . -p gemini -u character -a [character name]
```

**Getting your Gemini key:** Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and create a key there — AI Studio keys come with the Generative Language API pre-enabled. If you use a key from Google Cloud Console instead, you'll need to manually enable the [Generative Language API](https://console.developers.google.com/apis/api/generativelanguage.googleapis.com) for your project or you'll get a `PERMISSION_DENIED` error.

**Local model - experimental** (Ollama — free, runs on your GPU, no API key needed):

First-time setup:
```
winget install Ollama.Ollama
ollama pull llama3.2-vision
```

The `winget` install adds Ollama to your system. `ollama pull` downloads the vision model (~4.9GB). This only needs to be done once — the model is cached locally after the first download.

**Troubleshooting PATH:** If PowerShell doesn't recognize `ollama` after install, close and reopen PowerShell. If it still doesn't work, Ollama installs to `C:\Users\<you>\AppData\Local\Programs\Ollama\`. You can either:
- Add it to your PATH for this session: `$env:PATH += ";C:\Users\<you>\AppData\Local\Programs\Ollama"`
- Or call it directly: `& "C:\Users\<you>\AppData\Local\Programs\Ollama\ollama.exe" pull llama3.2-vision`

Make sure Ollama is running (it starts automatically after install, or run `ollama serve` in a separate terminal), then caption:
```
python -m klippbok.video caption . -p openai -u character -a [character name]
```

This extracts frames from each clip and sends them as images to Ollama's local API. By default it connects to `localhost:11434` and uses `llama3.2-vision`.

**Swapping models:** Use `--model` to pick any model you've pulled:
```
python -m klippbok.video caption . -p openai -u character -a [character name] --model gemma3:27b
```

**Other local endpoints** (vLLM, LM Studio, etc.):

Any server that speaks the OpenAI chat completions API works. Just point `--base-url` at it:
```
python -m klippbok.video caption . -p openai -u character -a [character name] --base-url http://localhost:8000/v1 --model my-vision-model
```

Common base URLs:
| Server | Default base URL |
|--------|-----------------|
| Ollama | `http://localhost:11434/v1` (default, no need to specify) |
| vLLM | `http://localhost:8000/v1` |
| LM Studio | `http://localhost:1234/v1` |
| text-generation-webui | `http://localhost:5000/v1` |

**Which vision model to use with Ollama:**

| Model | Size | Quality | Speed | Notes |
|-------|------|---------|-------|-------|
| `llama3.2-vision` | 4.9GB | Good | Fast | Recommended starting point |
| `llama3.2-vision:90b` | ~55GB | Best | Slow | Needs 48GB+ VRAM |
| `gemma3:27b` | 17GB | Very good | Medium | Strong visual understanding |
| `llava:13b` | 8GB | Good | Medium | Alternative if llama3.2 isn't available |
| `moondream` | 1.7GB | Basic | Very fast | Lightweight, good for quick test runs |

For training captions, `llama3.2-vision` hits the sweet spot of quality and speed. The captions don't need to be perfect — they need to be accurate about what's happening in the scene.

**Note: Local model captioning is experimental.** Local vision models tend to produce overly verbose, cinematic-analysis-style captions ("the imposing staircase serving as the perfect backdrop to her refined elegance") rather than the short, factual descriptions that work best for training ("Holly Golightly walks down the stairs in her apartment"). The Gemini and Replicate backends produce better training captions out of the box. Local model support is functional but may need prompt tuning for best results.

**What the flags mean:**
- `-p` — provider: `gemini`, `replicate`, or `openai` (any OpenAI-compatible endpoint)
- `-u` — use case: `character`, `style`, `motion`, or `object` (picks the right prompt)
- `-a` — anchor word: the concept's name as you'd say it naturally. Use quotes for multi-word names (e.g. `-a "Holly Golightly"`). This gets woven into captions: "Holly Golightly walks down the street." Not a token or slug — a real name.
- `-t` — secondary tags: additional words the model mentions only if visible (e.g. `-t sunglasses, tiara`)

Captions are short, natural language — 1-2 sentences describing what happens, not cinematic analysis.

**Provider quality notes:**
- **Replicate** and **Gemini** produce the best training captions — short, factual, grounded in what's visible. Recommended for production use.
- **Local models** (Ollama) are functional but experimental — they tend toward verbose, flowery descriptions that don't train as well. Prompt tuning may help.

**What's validated vs experimental:**
- **Character triage + captioning** is validated and production-ready. CLIP reliably matches human characters across angles, lighting, and distance. The `-u character` caption prompts produce good training captions.
- **Other concept types** (objects, settings, styles) are experimental. Triage may match clips by visual similarity rather than the specific concept — e.g. a cat reference might also match clips with similar framing or color palette. Captioning then applies the anchor word to mismatched clips. Users can refine results by manually reviewing the organized folders before captioning. Improving non-character triage accuracy is an upcoming feature.

**If some captions fail:** API errors happen (rate limits, server hiccups). Just re-run the same command — clips that already have .txt captions are automatically skipped, so only the failed ones get retried. Add `--overwrite` if you want to regenerate all captions from scratch.

### Step 4: Extract reference frames (for I2V)

```
python -m klippbok.video extract . -o references
```

This extracts the first frame of each clip as a PNG reference image. Options:
- Default: `--strategy first_frame` (frame 0, standard for I2V)
- Better quality: `--strategy best_frame` (samples frames, picks sharpest)

Each clip gets a stem-matched PNG: `clip_001.mov` -> `references/clip_001.png`

### Step 5: Validate — check completeness

```
python -m klippbok.dataset validate .
```

Checks that every video has a matching caption file. Reports any issues:
- Missing captions
- Empty captions
- Blank reference images
- Orphaned files (captions without matching videos)

### Optional: Generate manifest

```
python -m klippbok.dataset validate . --manifest
```

Writes `klippbok_manifest.json` to the dataset folder — a portable summary of what's in the dataset and any issues found.

---

## Scenario 1b: Ingest-first — raw video(s), triage after splitting (Path A)

**Starting point:** One or more long videos (movie, YouTube downloads, raw footage). You have reference images of your concept(s) and want Klippbok to detect scenes, split into training clips, identify which concept appears in each clip, and organize everything into folders.

**When to use this:** You want ALL clips from your source video, or you want to browse everything before deciding what to keep. You don't mind processing the full video.

### Step 1: Set up concepts folder

Create a `concepts/` folder with subfolders named by type. Put one reference image per concept:

```
concepts/
  character/
    hollygolightly.jpg
  object/
    cat.jpg
  setting/
    tiffanys.webp
```

The subfolder name tells Klippbok what kind of concept it is. Common names like `character`, `people`, `humans`, `actors` all work (see full alias list below). The filename (without extension) becomes the concept name used in captions.

### Step 2: Ingest — scene detect + split + normalize

```
python -m klippbok.video ingest "C:\path\to\raw\videos" -o "C:\path\to\clips"
```

This accepts either a single video file or a directory of videos. For each video it:
- Runs scene detection (finds cuts, fades, transitions)
- Splits at scene boundaries into individual clips
- Re-encodes each clip to training spec (16fps, 720p, 4n+1 frame count)
- Subdivides long scenes into ~5-second clips (81 frames max)

**Tuning scene detection:**
- `--threshold 27.0` (default) — scene detection sensitivity. Lower = more cuts detected (catches subtle transitions). Higher = fewer cuts (only hard cuts).
- `--max-frames 81` (default) — maximum frames per clip. Scenes longer than this are split into sub-clips (scene000a, scene000b, etc.). Set to `0` for no limit.

### Step 3: Triage — match clips against references

```
python -m klippbok.video triage "C:\path\to\clips" -s "C:\path\to\concepts" --organize "C:\path\to\sorted"
```

Because the clips are short (<30s), triage operates in clip mode — sampling a few frames from each clip and matching via CLIP embeddings.

- Clips matching a concept go to `sorted/<concept_name>/`
- Clips with title cards or text overlays go to `sorted/text_overlay/`
- Unmatched clips go to `sorted/unmatched/`

Sidecar files (.txt captions, .json) are copied alongside their clips.

Use `--move` instead of the default copy behavior if you don't want to keep the flat clips directory.

### Step 4: Caption each concept folder

After triage, caption each concept folder with the appropriate use case and anchor word. Pick your provider — see Step 3 in Scenario 1 for setup details on each.

**With Gemini** (cloud, needs `$env:GEMINI_API_KEY`):
```
python -m klippbok.video caption sorted/hollygolightly -p gemini -u character -a "Holly Golightly"
python -m klippbok.video caption sorted/tiffanys -p gemini -u object -a "Tiffany's"
python -m klippbok.video caption sorted/cat -p gemini -u object -a cat
```

**With Ollama** (local, free, needs `winget install Ollama.Ollama` + `ollama pull llama3.2-vision`):
```
python -m klippbok.video caption sorted/hollygolightly -p openai -u character -a "Holly Golightly"
python -m klippbok.video caption sorted/tiffanys -p openai -u object -a "Tiffany's"
python -m klippbok.video caption sorted/cat -p openai -u object -a cat
```

Re-run any command if it gets interrupted — clips with existing `.txt` captions are skipped automatically.

### Step 5: Extract reference frames and validate

```
python -m klippbok.video extract sorted/hollygolightly -o sorted/hollygolightly/references
```

Then validate each concept's dataset:
```
python -m klippbok.dataset validate sorted/hollygolightly
```

---

## Scenario 1c: Triage-first — target a character from raw video (Path B)

**Starting point:** Same as 1b — long raw videos and reference images. But this time you KNOW you only want specific content (e.g. only Holly Golightly scenes from a Breakfast at Tiffany's movie). You don't want to ingest every scene, just the ones containing your character.

**When to use this:** Your source videos are long and you have reference images. You want to skip the majority of scenes that don't contain your concept. This is dramatically faster for targeted training data extraction.

**Real-world example:** 25 Breakfast at Tiffany's videos produced ~1700 clips via full ingest. Triage-first identified only 162 Holly scenes — that's 90% less processing.

### Step 1: Set up concepts folder

Same as Scenario 1b Step 1 — create your `concepts/` directory with reference images.

### Step 2: Triage raw videos — scene-level matching

```
python -m klippbok.video triage "C:\path\to\raw\videos" -s "C:\path\to\concepts"
```

Because the source videos are long (>=30s each), triage automatically switches to scene-aware mode:
1. Detects scene boundaries in each video (same scene detection as ingest)
2. Samples 1-2 frames per scene via fast ffmpeg seeking (no full decode)
3. Matches sampled frames against your concept references via CLIP
4. Writes `scene_triage_manifest.json` with per-scene results

The output tells you how many scenes matched, how many were unmatched, and how many were flagged as text overlays.

**Tuning scene-level triage:**
- `--frames-per-scene 2` (default) — frames sampled per scene. More frames = better accuracy but slower.
- `--scene-threshold 27.0` (default) — scene detection sensitivity (same as ingest).
- `--threshold 0.70` (default) — CLIP similarity threshold for concept matching.

**Note:** `--organize` is not supported in scene mode — scenes haven't been split into clips yet. That's what Step 3 is for.

### Step 3: Review and edit the manifest (optional)

The `scene_triage_manifest.json` is human-readable and editable:

```json
{
  "triage_mode": "scene",
  "videos": [
    {
      "file": "001.mp4",
      "scenes": [
        {"scene_index": 0, "start_time": 0.0, "end_time": 5.2, "include": true,
         "matches": [{"concept": "hollygolightly", "similarity": 0.82}]},
        {"scene_index": 1, "start_time": 5.2, "end_time": 12.8, "include": false,
         "matches": []},
        ...
      ]
    }
  ]
}
```

The `include` field controls which scenes get split in the next step. Flip `true`/`false` to add or remove scenes before ingesting. This is your chance to review the triage results and make corrections.

### Step 4: Filtered ingest — split only matching scenes

```
python -m klippbok.video ingest "C:\path\to\raw\videos" -o "C:\path\to\clips" --triage scene_triage_manifest.json
```

This reads the manifest and only splits scenes marked `include: true`. Everything else is skipped — no scene detection, no splitting, no normalizing for scenes you don't want.

Output: a folder of training-ready clips, only from matching scenes.

### Step 5: Caption and validate

Same as Scenario 1b Steps 4-5. Since you've already targeted a specific concept, you likely only need one caption command:

```
python -m klippbok.video caption "C:\path\to\clips" -p gemini -u character -a "Holly Golightly"
python -m klippbok.video extract "C:\path\to\clips" -o "C:\path\to\clips\references"
python -m klippbok.dataset validate "C:\path\to\clips"
```

---

## Understanding the thresholds

Triage uses two separate similarity thresholds. Both use CLIP cosine similarity scores, but the score ranges are different because **image-to-image** comparison produces higher scores than **text-to-image** comparison.

**Concept matching threshold** (`--threshold`, default: 0.70)

This controls how similar a clip frame must be to a reference image to count as a match. Both sides are images, so CLIP produces relatively high similarity scores.

| Score Range | What it means |
|-------------|---------------|
| 0.85-1.00 | Very strong match — same person/place, similar angle |
| 0.70-0.85 | Good match — same concept, different angle/lighting/distance |
| 0.55-0.70 | Weak match — similar content but not confident |
| Below 0.55 | Unrelated content |

**Tuning:** Lower the threshold if you're missing clips you know contain your concept (false negatives). Raise it if unrelated clips are being matched (false positives). The triage output shows the closest near-miss score for unmatched clips — if you see scores like 0.68 for clips that should match, drop the threshold to 0.65.

**Text overlay threshold** (default: 0.27, not exposed as CLI flag yet)

This compares clip frames against text descriptions like "movie title card with large text" and "closing credits scrolling text". Because one side is text and the other is an image, CLIP scores are much lower — a strong text-image match might only score 0.30-0.35.

| Score Range | What it means |
|-------------|---------------|
| 0.30+ | Strong text/title card detection |
| 0.25-0.30 | Borderline — might be text, might be a dark scene |
| Below 0.25 | Normal scene content |

The default 0.27 is intentionally conservative — it catches obvious title cards and credits but won't flag dialogue scenes that happen to have text in the background. If you find it missing title cards, you can lower it; if it's catching real scenes, raise it.

**Why the numbers are so different:** CLIP was trained to map images and text into the same embedding space, but image-image pairs naturally cluster more tightly than text-image pairs. An image of Holly Golightly is very similar to another image of Holly Golightly (0.70+). But the text "a movie title card" is only moderately similar to an image of a title card (0.27+) because language and pixels encode information very differently.

---

## Scenario 2: Pre-cut clips with existing captions (I2V character LoRA)

**Starting point:** Same clips as Scenario 1, but the source folder ALSO has .txt caption sidecar files (pre-written). You just need to normalize, extract reference frames, and validate.

### Steps

1. **Scan** (same as Scenario 1 Step 1)
2. **Normalize** — same command, sidecar .txt files get copied automatically
3. **Skip captioning** — captions already exist
4. **Extract reference frames** (same as Scenario 1 Step 4)
5. **Validate** (same as Scenario 1 Step 5)

---

## Scenario 3: Validate existing flat dataset (T2V, captions only)

**Starting point:** A folder with video clips and caption .txt files already in place. No reference images needed (T2V training). You just want to validate that everything is paired correctly.

```
cd "C:\path\to\your\dataset"
python -m klippbok.dataset validate .
```

The validator detects "flat" structure (all files in one folder), pairs videos with captions by filename stem, and reports completeness.

For T2V where you don't need reference images, this is all you need. The validator won't complain about missing references unless you configure it to require them.

---

## Concepts folder reference

The `concepts/` folder tells Klippbok what to look for during triage. Each subfolder represents a concept type, and each image inside is a reference for matching.

```
concepts/
  character/
    hollygolightly.jpg    → ConceptReference("hollygolightly", CHARACTER)
    paul.png              → ConceptReference("paul", CHARACTER)
  setting/
    tiffanys.webp         → ConceptReference("tiffanys", SETTING)
  object/
    cat.jpg               → ConceptReference("cat", OBJECT)
```

**Folder name aliases** — you don't need to use the exact type name. Klippbok recognizes common variations:

| Type | Recognized folder names |
|------|------------------------|
| CHARACTER | `character`, `characters`, `person`, `people`, `human`, `humans`, `face`, `faces`, `actor`, `actors` |
| STYLE | `style`, `styles`, `aesthetic`, `aesthetics`, `look`, `looks` |
| MOTION | `motion`, `movement`, `action`, `actions` |
| OBJECT | `object`, `objects`, `thing`, `things`, `item`, `items`, `prop`, `props` |
| SETTING | `setting`, `settings`, `location`, `locations`, `place`, `places`, `environment`, `environments`, `scene`, `scenes`, `background`, `backgrounds` |

Lookup is case-insensitive and strips hyphens/underscores/spaces. "My Characters" and "my_characters" both resolve to CHARACTER.

Unknown folder names still work — the reference images are used for matching, but won't auto-select a captioning prompt. Klippbok will print a note about unrecognized folders.

---

## Command Reference

| Command | What it does |
|---------|-------------|
| `python -m klippbok.video scan .` | Probe + validate clips (read-only) |
| `python -m klippbok.video ingest <video\|dir> -o <out>` | Scene detect + split + normalize |
| `python -m klippbok.video normalize . -o <out>` | Downscale, fix fps, copy captions |
| `python -m klippbok.video triage . -s concepts/` | Match clips/scenes against reference images |
| `python -m klippbok.video extract . -o <out>` | Extract reference frames as PNG |
| `python -m klippbok.video caption . -p gemini` | Generate captions via VLM |
| `python -m klippbok.video score .` | Score caption quality (no API) |
| `python -m klippbok.dataset validate .` | Validate dataset completeness |

### Ingest flags

| Flag | What it does |
|------|-------------|
| `--max-frames 81` | Max frames per clip (default: 81). Use `0` for no limit |
| `--threshold 27.0` | Scene detection sensitivity (default: 27.0) |
| `--triage <manifest>` | Filtered ingest — only split scenes from triage manifest |
| `--caption` | Auto-caption clips after splitting |
| `--config my_config.yaml` | Load all settings from a config file |

### Triage flags

| Flag | Short | What it does |
|------|-------|-------------|
| `--concepts` | `-s` | Path to concepts/ folder (required) |
| `--threshold` | | Similarity threshold for matching (default: 0.70) |
| `--frames` | | Frames to sample per clip (default: 5) |
| `--frames-per-scene` | | Frames per scene for long videos (default: 2) |
| `--scene-threshold` | | Scene detection threshold for long videos (default: 27.0) |
| `--organize <dir>` | | Copy clips into concept-named folders (clip mode only) |
| `--move` | | Move instead of copy (with --organize) |
| `--clip-model` | | CLIP model name (default: openai/clip-vit-base-patch32) |

### Scan / normalize flags

| Flag | What it does |
|------|-------------|
| `--fps 24` | Target frame rate (default: 16 for Wan models) |
| `-v` | Verbose — show every clip individually |
| `--format .mov` | Force output format (default: match source) |
| `--config my_config.yaml` | Load all settings from a config file |

### Caption flags

| Flag | Short | What it does |
|------|-------|-------------|
| `--provider` | `-p` | VLM provider: `gemini`, `replicate`, `openai` |
| `--use-case` | `-u` | Prompt style: `character`, `style`, `motion`, `object` |
| `--anchor-word` | `-a` | Concept name, used naturally in captions |
| `--tags` | `-t` | Secondary tags, mentioned only if visible |
| `--overwrite` | | Regenerate all captions (default: skip existing) |
| `--caption-fps` | | Frames per second sampled for captioning (default: 1) |

### Validate flags

| Flag | What it does |
|------|-------------|
| `--manifest` | Write klippbok_manifest.json |
| `--buckets` | Preview bucket distribution |
| `--quality` | Enable blur/exposure checks on references |
| `--duplicates` | Detect near-duplicate reference images |
| `--json` | Output as JSON instead of formatted report |
