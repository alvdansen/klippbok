```
   ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
  ╭────────────────────────────────────────────────────────────────────╮
  │                                                                    │
  │                      ✂  k l i p p b o k                           │
  │                                                                    │
  │        video dataset curation for LoRA training                    │
  │                                                                    │
  │    ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐    │
  │    │ scan  │──▶│triage │──▶│caption│──▶│  val  │──▶│ train │    │
  │    └───────┘   └───────┘   └───────┘   └───────┘   └───────┘    │
  │                                                                    │
  │                          alvdansen labs                            │
  │                                                                    │
  ╰────────────────────────────────────────────────────────────────────╯
   ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
```

**Video dataset curation, preparation, and annotation for LoRA training.**

Klippbok is a tool for processing pre-clipped and raw video footage to produce training-ready datasets. This tool features support for scene detection, CLIP-based triage, VLM captioning, reference frame extraction, and dataset validation. It works with any trainer (musubi-tuner, ai-toolkit, kohya/sd-scripts) and designed to prep data for finetuning any modern video diffusion model.

Built by [Minta](https://github.com/aramintak) and [Timothy](https://github.com/timm156), distilling three years of professional LoRA finetuning into opinionated tooling. Every default, threshold, and pipeline decision comes from shipping production models for enterprise clients — not guesswork. The same methodology behind [lora-gym](https://github.com/alvdansen/lora-gym) and 50+ published models on [Hugging Face](https://huggingface.co/alvdansen).

---

## What it does

| Stage | Tool | What happens |
|-------|------|-------------|
| **Ingest** | `scan`, `ingest`, `normalize` | Scene detection, splitting, fps/resolution normalization |
| **Triage** | `triage` | CLIP-based matching against reference images — find your character in 2 hours of footage |
| **Caption** | `caption`, `score`, `audit` | VLM-generated captions via Gemini, Replicate, or local models (Ollama) |
| **Extract** | `extract` | Reference frame extraction (first frame, best frame) for I2V training |
| **Validate** | `validate`, `organize` | Dataset completeness checks, trainer-specific output formatting |

## Why Klippbok exists

Existing training frameworks assume do not take responsibility for how your data is prepared and there is very little reliable guidance on dataset prep for video models. 

A common reality:

- Your source material is a 2-hour movie and you need 150 clips of one character
- Your clips are 1080p/24fps but training needs 720p/16fps with 4n+1 frame counts
- You need captions and you don't know where to start
- You need to know which clips are blurry, which have text overlays, which are duplicates, etc

Klippbok solves arguably the most difficult part of finetuning, preparing the data, so that you can focus on training.

Klippbok is an [Alvdansen Labs](https://huggingface.co/alvdansen) project — an open-source initiative to advance finetuning practices and make production-quality training accessible to everyone.

## Quick start

```bash
pip install klippbok[all]
```

**Scenario: Character LoRA from raw footage**

```powershell
# Set up a concepts folder with a reference image — any screenshot or photo works
mkdir concepts\character
# Copy a reference image into concepts\character\ (e.g. a screenshot of your character)

# 1. Find every scene containing your character using CLIP visual matching
python -m klippbok.video triage "C:\raw_videos" -s concepts/

# 2. Split only matching scenes into training clips
python -m klippbok.video ingest "C:\raw_videos" -o clips --triage scene_triage_manifest.json

# 3. Caption
python -m klippbok.video caption clips -p gemini -u character -a "Holly Golightly"

# 4. Extract reference frames (for I2V)
python -m klippbok.video extract clips -o clips/references

# 5. Validate
python -m klippbok.dataset validate clips
```

Output: normalized clips (16fps, 720p), `.txt` caption sidecars, `.png` reference frames — ready for any trainer.

See [docs/PIPELINES.md](docs/PIPELINES.md) for all 6 supported pipelines.

## Pipelines

Different training goals need different tools. Klippbok supports 6 pipelines:

| Your situation | Pipeline |
|---------------|----------|
| Raw footage + character reference images | **Triage-first** — find your character, skip everything else |
| Pre-cut clips of a character | **Normalize + caption** — fix specs, add captions |
| Clips selected for visual style | **Style LoRA** — captions describe content, model learns style from pixels |
| Clips selected for motion patterns | **Motion LoRA** — captions focus on movement and camera behavior |
| Existing dataset with wrong specs | **Cleanup** — re-normalize, score captions, validate |
| Raw footage + object/setting references | **Experimental triage** — works but needs manual review |

Full details: [docs/PIPELINES.md](docs/PIPELINES.md)

## Installation

**Minimal** (config + validation only):
```bash
pip install klippbok
```

**With specific features:**
```bash
pip install klippbok[video]      # + scene detection, normalization (requires ffmpeg)
pip install klippbok[caption]    # + Gemini, Replicate captioning backends
pip install klippbok[triage]     # + CLIP-based triage (torch, transformers)
pip install klippbok[dataset]    # + rich reports, file type detection
pip install klippbok[all]        # everything
```

**Requirements:**
- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on PATH (for video processing)
- API key for cloud captioning (Gemini, Replicate) — or use Ollama for free local captioning

## Caption providers

| Provider | Flag | Needs | Quality | Notes |
|----------|------|-------|---------|-------|
| **Gemini** | `-p gemini` | `GEMINI_API_KEY` | Best | Free tier available |
| **Replicate** | `-p replicate` | `REPLICATE_API_TOKEN` | Best | Pay-per-use |
| **Ollama** (local) | `-p openai` | Ollama installed | Good | Free, runs on your GPU |

Local models tend toward verbose captions. Gemini and Replicate produce the short, factual captions that train best. See [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) for provider setup.

## Command reference

```
python -m klippbok.video scan <dir>                    # probe clips, report issues
python -m klippbok.video ingest <path> -o <out>        # scene detect + split + normalize
python -m klippbok.video normalize <dir> -o <out>      # fix fps, resolution, frame count
python -m klippbok.video triage <dir> -s <concepts>    # CLIP match against references
python -m klippbok.video caption <dir> -p <provider>   # generate .txt captions
python -m klippbok.video score <dir>                   # check caption quality (local)
python -m klippbok.video extract <dir> -o <out>        # extract reference frames as PNG
python -m klippbok.video audit <dir>                   # compare captions against VLM

python -m klippbok.dataset validate <dir>              # check dataset completeness
python -m klippbok.dataset organize <dir> -o <out>     # format for specific trainers
```

Full flag reference with examples: [docs/COMMANDS.md](docs/COMMANDS.md)

## Trainer compatibility

Klippbok produces standard formats that work with:

- **musubi-tuner** — generates TOML config via `organize -t musubi`
- **ai-toolkit (ostris)** — generates YAML config via `organize -t aitoolkit`
- **kohya/sd-scripts** — flat layout with `.txt` sidecars (standard format)
- **Any trainer** that reads video + caption sidecar pairs

## Visual triage: let a reference image organize your dataset

This is one of Klippbok's most powerful features. Instead of manually scrubbing through hours of footage, you drop a reference image into a folder and Klippbok finds every scene containing that subject.

**How it works:** You create a `concepts/` folder with subfolders named by type. Put one or more reference images in each — a screenshot, a character sheet, a photo. Klippbok uses [CLIP](https://openai.com/research/clip) to compare video frames against your reference images and automatically identifies which scenes match.

```
concepts/
  character/
    luna_ref.jpg              ← just one image is enough
  setting/
    cafe_exterior.png
```

```powershell
python -m klippbok.video triage "C:\raw_videos" -s concepts/
```

That's it. Klippbok probes every video, detects scenes, samples frames from each scene, and matches them against your references. The output is a reviewable manifest where you can verify matches and flip `include: true/false` per scene before splitting.

**Why this matters:** A 2-hour film might produce 1700 clips when fully split. If you're training a character LoRA, maybe 150 of those actually contain your character. Without triage, you'd split everything, caption everything, then manually delete 1500 files. With triage, you skip all of that — only matching scenes get split.

**Auto-detection:** Klippbok automatically adapts to your source material:
- **Short clips (<30s):** Samples a few frames per clip, matches directly → `triage_manifest.json`
- **Long videos (>=30s):** Detects scenes first, samples 1-2 frames per scene → `scene_triage_manifest.json`

**Reliability:** Character triage is production-ready — CLIP recognizes human identity across angles, lighting, and distance. Object and setting triage is experimental and benefits from manual manifest review.

## Documentation

- [**COMMANDS.md**](docs/COMMANDS.md) — Full command glossary with every flag and option
- [**CAPTIONING.md**](docs/CAPTIONING.md) — Captioning methodology: use-case prompts, anchor words, provider comparison
- [**PIPELINES.md**](docs/PIPELINES.md) — Which pipeline to use for your training scenario
- [**WALKTHROUGH.md**](docs/WALKTHROUGH.md) — Step-by-step tutorial tested on real data

## Roadmap

Klippbok is actively developed. Here's what's coming:

**Organization & multi-source handling**
- Smarter reference image routing during organize (I2V references, subject references, style references as separate control signals)
- Multi-source dataset merging — combine clips from different sources with different configs into a single organized output
- Per-concept captioning configs (different anchor words, use-cases, or providers per concept folder)
- Organize from triage output directly — `organize --from-triage manifest.json` without intermediate steps

**Trainer integration**
- Additional trainer config generators (kohya/sd-scripts TOML, SimpleTuner)
- Config validation against trainer requirements (e.g. "musubi needs 4n+1 frames")
- Training-ready archive export (zip with config + data, ready to upload to cloud training)

**Caption improvements**
- Caption refinement pipeline — score, filter, and re-caption low-quality entries automatically
- Multi-pass captioning — generate captions from multiple providers and pick the best
- Caption style transfer — convert verbose captions to prompt-style without re-running VLM

**Triage & quality**
- Multi-concept triage in a single pass with conflict resolution
- Temporal consistency scoring — detect clips where the subject appears/disappears mid-clip
- Scene-level quality scoring (motion blur, shot stability, lighting consistency)

**Data management**
- Dataset versioning — track what changed between iterations
- Deduplication across datasets — find overlapping content between projects
- Dataset splitting for train/val with stratified sampling by concept

## Part of the Dimljus ecosystem

Klippbok is the standalone data preparation toolkit from the soon to be released Dimljus Trainer, a video LoRA training framework for diffusion transformer models. Klippbok handles everything before training — Dimljus handles training itself.

You don't need Dimljus to use Klippbok. The output works with any trainer.

## License

Apache 2.0
