# Klippbok Command Reference

Complete reference for every CLI command, flag, and option.

**Two entry points:**
- `python -m klippbok.video` — video processing, captioning, triage
- `python -m klippbok.dataset` — validation, organization

All commands accept `--config` / `-c` to load a `klippbok_data.yaml` config file. Without it, sensible Wan-model defaults are used (16fps, 720p, auto frame count).

---

## Video Commands

### scan

Probe video clips and report issues. The first step for any dataset — tells you what you have and what needs fixing.

```
python -m klippbok.video scan <directory> [options]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `directory` | | *(required)* | Directory of video clips to scan |
| `--config` | `-c` | | Path to `klippbok_data.yaml` |
| `--fps` | | 16 | Target frame rate to validate against |
| `--verbose` | `-v` | off | Show per-clip detail instead of grouped summary |

**Output:** Grouped report showing which clips are ready, which need normalization, and which are unusable. Without `-v`, identical issues are grouped (e.g. "26 clips: 24fps, needs normalization").

**When to use:** Before any other command. Scan tells you whether you can skip normalization or need it.

---

### ingest

Scene detection + splitting + normalization in one step. Takes raw video files (any length) and produces training-ready clips.

```
python -m klippbok.video ingest <video_or_dir> -o <output_dir> [options]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `video` | | *(required)* | Path to a video file or directory of videos |
| `--output` | `-o` | *(required)* | Output directory for clips |
| `--config` | `-c` | | Path to `klippbok_data.yaml` |
| `--threshold` | `-t` | 27.0 | Scene detection threshold (lower = more cuts) |
| `--max-frames` | | 81 | Max frames per clip (~5s at 16fps). Use `0` for no limit |
| `--triage` | | | Path to `scene_triage_manifest.json` — only split included scenes |
| `--caption` | | off | Auto-caption clips after splitting |
| `--provider` | | gemini | VLM provider when using `--caption` |

**Accepts:** Single video file *or* a directory of videos. Processes everything into the same output dir.

**Triage mode:** When `--triage` is provided, skips scene detection entirely and only splits scenes marked `include: true` in the manifest. This is the second step of the triage-first pipeline.

**Output:** Normalized clips (16fps, 720p, 4n+1 frames) + `manifest.json` listing all produced clips.

---

### normalize

Fix fps, resolution, and frame count on pre-cut clips. Use this when you already have clips but they need spec adjustments.

```
python -m klippbok.video normalize <directory> -o <output_dir> [options]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `directory` | | *(required)* | Directory of clips to normalize |
| `--output` | `-o` | *(required)* | Output directory for normalized clips |
| `--config` | `-c` | | Path to `klippbok_data.yaml` |
| `--fps` | | 16 | Target frame rate |
| `--format` | `-f` | *(match source)* | Force output format: `.mp4`, `.mov`, or `.mkv` |

**What it does:** Re-encodes clips to match target fps, resolution, and 4n+1 frame count. Copies existing `.txt` caption sidecars to the output directory alongside normalized clips.

**When to use:** When `scan` reports that your clips need normalization (wrong fps, resolution, or frame count).

---

### caption

Generate `.txt` caption sidecars using a VLM (vision-language model).

```
python -m klippbok.video caption <directory> -p <provider> [options]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `directory` | | *(required)* | Directory of video clips |
| `--provider` | `-p` | gemini | VLM provider: `gemini`, `replicate`, or `openai` |
| `--use-case` | `-u` | *(general)* | Prompt template: `character`, `style`, `motion`, `object` |
| `--anchor-word` | `-a` | | Subject name woven into captions (e.g. `"Luna"`) |
| `--tags` | `-t` | | Secondary tags mentioned when visible (e.g. `-t vintage retro`) |
| `--overwrite` | | off | Overwrite existing `.txt` captions |
| `--base-url` | | `localhost:11434/v1` | Base URL for OpenAI-compatible API (Ollama) |
| `--model` | | `llama3.2-vision` | Model name for OpenAI-compatible backend |
| `--caption-fps` | | 1 | Frame sampling rate for captioning |

**Use cases and what they control:**

| Use case | Prompt focus | Omits | Best for |
|----------|-------------|-------|----------|
| `character` | Action, setting, interaction | Appearance, clothing, features | Character LoRA |
| `style` | Content, composition, subject | Art style, color grading, mood | Style LoRA |
| `motion` | Movement, speed, camera motion | Identity, appearance | Motion LoRA |
| `object` | Scene context around subject | Object appearance/details | Object LoRA |
| *(none)* | General description | Color palettes, cinematography | General purpose |

**Anchor word:** The `-a` flag tells the VLM to use this name naturally in captions. For character LoRAs, this is the character's name. The VLM weaves it in — it's not mechanically prepended.

**Provider comparison:**

| Provider | Quality | Cost | Setup |
|----------|---------|------|-------|
| Gemini | Best | Free tier | `GEMINI_API_KEY` env var |
| Replicate | Best | ~$0.01/clip | `REPLICATE_API_TOKEN` env var |
| Ollama (openai) | Good | Free | Ollama installed locally |

See [CAPTIONING.md](CAPTIONING.md) for the full captioning methodology.

---

### score

Check caption quality without any API calls. Uses heuristics to flag short, repetitive, or potentially problematic captions.

```
python -m klippbok.video score <directory>
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `directory` | | *(required)* | Directory of `.txt` caption files |

**Output:** Quality scores per caption with a summary. Useful for checking community-shared datasets or reviewing after captioning.

---

### extract

Extract reference frames from video clips as PNG images. Used to produce reference images for I2V training.

```
python -m klippbok.video extract <directory> -o <output_dir> [options]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `directory` | | *(required)* | Directory of video clips / images |
| `--output` | `-o` | *(required)* | Output directory for PNG reference images |
| `--strategy` | `-s` | `first_frame` | `first_frame` or `best_frame` |
| `--samples` | | 10 | Number of frames to sample for `best_frame` |
| `--overwrite` | | off | Overwrite existing reference images |
| `--selections` | | | Path to JSON selections manifest (from `--template`) |
| `--template` | | | Generate a selection template JSON (no extraction) |

**Strategies:**
- `first_frame` — Extracts the first frame. Fast, deterministic. Best for most character LoRAs.
- `best_frame` — Samples N evenly-spaced frames, picks the sharpest (by Laplacian score). Better for clips with blurry openings.

**Selection workflow:** For manual frame selection, use `--template` to generate a JSON file listing all clips, edit the frame numbers, then run with `--selections` to extract your chosen frames.

**Mixed datasets:** If the directory contains still images (PNG, JPG), they're copied through as-is alongside extracted video frames.

---

### triage

Match clips against reference images using CLIP embeddings. The key to efficiently finding your character in hours of footage.

```
python -m klippbok.video triage <directory> -s <concepts_dir> [options]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `directory` | | *(required)* | Directory of video clips to triage |
| `--concepts` | `-s` | *(required)* | Path to concepts directory with reference images |
| `--threshold` | | 0.70 | Similarity threshold (0.0–1.0) |
| `--frames` | | 5 | Frames to sample per short clip |
| `--output` | `-o` | | Custom path for triage manifest JSON |
| `--organize` | | | Copy/move matched clips into concept-named folders |
| `--move` | | off | Move files instead of copying (with `--organize`) |
| `--frames-per-scene` | | 2 | Frames to sample per scene (long videos) |
| `--scene-threshold` | | 27.0 | Scene detection threshold (long videos) |
| `--clip-model` | | `openai/clip-vit-base-patch32` | CLIP model to use |

**Concepts directory structure:**
```
concepts/
  character/
    reference1.jpg          ← one or more reference images per concept
    reference2.png
  setting/
    location_ref.jpg
```

The subfolder name becomes the concept name. Put one or more reference images per concept. CLIP compares video frames against these images to find matches.

**Auto-detection:** Klippbok probes each video's duration:
- **Short clips (<30s):** Samples frames directly, matches per clip → `triage_manifest.json`
- **Long videos (>=30s):** Detects scenes first, samples per scene → `scene_triage_manifest.json` with editable `include` fields

**After triage:** Use `--triage manifest.json` on the `ingest` command to only split matching scenes.

---

### audit

Compare existing captions against fresh VLM output. Useful for reviewing community-shared datasets or checking your own captions after manual edits.

```
python -m klippbok.video audit <directory> [options]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `directory` | | *(required)* | Directory of captioned clips |
| `--provider` | | gemini | VLM provider |
| `--use-case` | | | Use case for prompt selection |
| `--mode` | | `report_only` | `report_only` or `save_audit` |

**Modes:**
- `report_only` — Prints comparison to console
- `save_audit` — Writes `.audit.txt` files alongside existing captions

---

## Dataset Commands

### validate

Check dataset completeness, quality, and organization. The pre-training checklist.

```
python -m klippbok.dataset validate <path> [options]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `path` | | *(required)* | Dataset folder or `klippbok_data.yaml` file |
| `--config` | | | Path to `klippbok_data.yaml` |
| `--manifest` | | off | Write `klippbok_manifest.json` |
| `--buckets` | | off | Show bucketing preview |
| `--quality` | | off | Enable blur/exposure checks on reference images |
| `--duplicates` | | off | Enable perceptual duplicate detection |
| `--json` | | off | Output as JSON instead of formatted report |

**What it checks:**
- Every video has a matching `.txt` caption sidecar
- Captions are non-empty and within token limits
- Frame counts follow 4n+1 pattern
- Reference images exist and aren't blank/corrupted
- No perceptual duplicates (with `--duplicates`)
- Image quality: blur and exposure (with `--quality`)
- Cross-dataset duplicate detection (when multiple sources configured)

**Output:** Formatted report with pass/fail summary, issue details, and a copy-pasteable `organize` command.

---

### organize

Take validated data and produce a clean, trainer-ready directory. Optionally generates trainer-specific config files.

```
python -m klippbok.dataset organize <path> -o <output_dir> [options]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `path` | | *(required)* | Source dataset folder |
| `--output` | `-o` | *(required)* | Output directory |
| `--layout` | `-l` | `flat` | `flat` (universal) or `klippbok` (hierarchical) |
| `--trainer` | `-t` | | Generate trainer config: `musubi`, `aitoolkit`. Repeatable |
| `--concepts` | | | Only organize clips from these concept folders (comma-separated) |
| `--move` | | off | Move files instead of copy |
| `--dry-run` | | off | Preview without touching files |
| `--strict` | | off | Exclude samples with warnings too (default: only errors) |
| `--config` | `-c` | | Path to `klippbok_data.yaml` |
| `--manifest` | | off | Write `klippbok_manifest.json` to output directory |

**Layouts:**
- `flat` — All files in one directory, stem-matched. Works with every trainer.
- `klippbok` — Hierarchical structure separating targets from control signals.

**Trainer configs:**
- `-t musubi` → `musubi_dataset.toml` — complete dataset config with usage instructions
- `-t aitoolkit` → `aitoolkit_config.yaml` — full training config skeleton with TODO markers

**Concepts filtering:** After triage organizes clips into concept folders (e.g. `sorted/luna/`, `sorted/cat/`), use `--concepts luna` to only include clips from specific concepts.

---

## Common Workflows

### Character LoRA from raw footage (triage-first)

```powershell
# 1. Set up reference images
#    concepts/character/ref1.jpg, concepts/character/ref2.png

# 2. Find matching scenes
python -m klippbok.video triage "C:\raw_videos" -s concepts/

# 3. Split only matching scenes
python -m klippbok.video ingest "C:\raw_videos" -o clips --triage scene_triage_manifest.json

# 4. Caption
python -m klippbok.video caption clips -p gemini -u character -a "[character name]"

# 5. Extract reference frames (for I2V)
python -m klippbok.video extract clips -o clips/references

# 6. Validate
python -m klippbok.dataset validate clips

# 7. Organize for training
python -m klippbok.dataset organize clips -o training_ready -t musubi
```

### Character LoRA from pre-cut clips

```powershell
# 1. Check what you have
python -m klippbok.video scan clips/

# 2. Normalize if needed
python -m klippbok.video normalize clips/ -o normalized

# 3. Caption
python -m klippbok.video caption normalized -p gemini -u character -a "[character name]"

# 4. Extract + validate + organize
python -m klippbok.video extract normalized -o normalized/references
python -m klippbok.dataset validate normalized
python -m klippbok.dataset organize normalized -o training_ready -t musubi
```

### Style LoRA

```powershell
python -m klippbok.video normalize clips/ -o normalized
python -m klippbok.video caption normalized -p gemini -u style
python -m klippbok.dataset validate normalized
python -m klippbok.dataset organize normalized -o training_ready -t aitoolkit
```

### Quick dataset cleanup

```powershell
python -m klippbok.video normalize old_dataset/ -o fixed
python -m klippbok.video score fixed
python -m klippbok.dataset validate fixed
```

---

## Environment Variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | caption (gemini) | Google AI Studio API key |
| `REPLICATE_API_TOKEN` | caption (replicate) | Replicate API token |

## Requirements

- **Python 3.10+**
- **ffmpeg** on PATH (for all video processing)
- **torch + transformers** (for triage only — installed with `klippbok[triage]`)
