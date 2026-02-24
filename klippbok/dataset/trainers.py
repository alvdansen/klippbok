"""Trainer config file generators.

Registry of functions that produce trainer-specific config files from
organized dataset output. Each generator takes the organized samples
and config, and writes a config file that the trainer can consume directly.

Adding a new trainer: define a function and decorate it with @register_trainer.

Currently supported:
    - musubi: musubi-tuner dataset TOML
    - aitoolkit: ai-toolkit dataset YAML

Usage:
    from klippbok.dataset.trainers import generate_trainer_config

    path = generate_trainer_config("musubi", samples, output_dir, config, layout)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from klippbok.config.data_schema import KlippbokDataConfig
from klippbok.dataset.errors import OrganizeError
from klippbok.dataset.models import OrganizeLayout, OrganizedSample


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps trainer name -> generator function
TRAINER_REGISTRY: dict[str, Callable[..., Path]] = {}


def register_trainer(name: str) -> Callable:
    """Decorator to register a trainer config generator.

    Usage:
        @register_trainer("my_trainer")
        def generate_my_trainer_config(samples, output_dir, config, layout):
            ...
    """
    def decorator(func: Callable[..., Path]) -> Callable[..., Path]:
        TRAINER_REGISTRY[name] = func
        return func
    return decorator


def get_available_trainers() -> list[str]:
    """Return sorted list of registered trainer names."""
    return sorted(TRAINER_REGISTRY.keys())


def generate_trainer_config(
    trainer_name: str,
    samples: list[OrganizedSample],
    output_dir: Path,
    config: KlippbokDataConfig,
    layout: OrganizeLayout,
    dry_run: bool = False,
) -> Path:
    """Generate a trainer config file using the named generator.

    Args:
        trainer_name: Registered trainer name (e.g. "musubi", "aitoolkit").
        samples: The organized samples (non-skipped).
        output_dir: The organize output directory.
        config: The klippbok data config.
        layout: Which layout was used for organizing.
        dry_run: If True, compute the path but don't write the file.

    Returns:
        Path where the config file was (or would be) written.

    Raises:
        OrganizeError: If the trainer name is not registered.
    """
    if trainer_name not in TRAINER_REGISTRY:
        available = ", ".join(get_available_trainers()) or "(none)"
        raise OrganizeError(
            f"Unknown trainer '{trainer_name}'. "
            f"Available trainers: {available}."
        )

    generator = TRAINER_REGISTRY[trainer_name]
    return generator(
        samples=samples,
        output_dir=output_dir,
        config=config,
        layout=layout,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_forward_slash(path: Path) -> str:
    """Convert a path to forward-slash string.

    Trainer configs need forward slashes even on Windows, because
    many trainers are Linux-first and choke on backslashes.
    """
    return str(path).replace("\\", "/")


def _resolution_to_wh(resolution: int) -> list[int]:
    """Convert a resolution tier (height) to [width, height].

    Uses 16:9 aspect ratio for standard tiers.
    480 -> [854, 480], 720 -> [1280, 720], 1080 -> [1920, 1080].
    """
    aspect_map = {
        480: [854, 480],
        720: [1280, 720],
        1080: [1920, 1080],
    }
    return aspect_map.get(resolution, [int(resolution * 16 / 9), resolution])


def _unique_frame_counts(samples: list[OrganizedSample]) -> list[int]:
    """Extract deduplicated, sorted frame counts from organized samples."""
    counts = {s.frame_count for s in samples if s.frame_count is not None}
    return sorted(counts)


def _video_directory(output_dir: Path, layout: OrganizeLayout) -> str:
    """Get the directory containing video files, in forward-slash format."""
    if layout == OrganizeLayout.DIMLJUS:
        return _to_forward_slash(output_dir / "training" / "targets")
    return _to_forward_slash(output_dir)


def _reference_directory(output_dir: Path, layout: OrganizeLayout) -> str:
    """Get the directory containing reference images, in forward-slash format."""
    if layout == OrganizeLayout.DIMLJUS:
        return _to_forward_slash(output_dir / "training" / "signals" / "references")
    return _to_forward_slash(output_dir)


# ---------------------------------------------------------------------------
# musubi-tuner
# ---------------------------------------------------------------------------

@register_trainer("musubi")
def generate_musubi_config(
    samples: list[OrganizedSample],
    output_dir: Path,
    config: KlippbokDataConfig,
    layout: OrganizeLayout,
    dry_run: bool = False,
) -> Path:
    """Generate a musubi-tuner dataset TOML file.

    musubi-tuner expects a TOML file describing where videos are,
    what resolution to use, and how to cache latents.

    The generated file includes a header comment with the regeneration
    command so the user can re-run organize if they change settings.

    Args:
        samples: Organized samples.
        output_dir: Output directory.
        config: Klippbok data config.
        layout: Which organize layout was used.
        dry_run: If True, returns the path without writing.

    Returns:
        Path to the generated musubi_dataset.toml file.
    """
    output_path = output_dir / "musubi_dataset.toml"

    wh = _resolution_to_wh(config.video.resolution)
    video_dir = _video_directory(output_dir, layout)
    cache_dir = _to_forward_slash(output_dir / "cache")
    frame_counts = _unique_frame_counts(samples)

    # Get repeats from first dataset source (default 1)
    num_repeats = 1
    if config.datasets:
        num_repeats = config.datasets[0].repeats

    # Check if any samples have reference images (I2V training)
    has_references = any(s.reference_dest is not None for s in samples)
    ref_dir = _reference_directory(output_dir, layout) if has_references else None

    lines = [
        "# musubi-tuner dataset config",
        f"# Generated by: python -m klippbok.dataset organize <source> -o {_to_forward_slash(output_dir)} -t musubi",
        "",
        "# ─── HOW TO USE THIS FILE ───",
        "# This is the dataset TOML passed to musubi-tuner via --dataset_config.",
        "# Training params (model, LoRA, optimizer) are separate CLI arguments.",
        "#",
        "# Before training, you must pre-cache latents and text embeddings:",
        "#   python wan_cache_latents.py --dataset_config musubi_dataset.toml --vae <vae_path>",
        "#   python wan_cache_text_encoder_outputs.py --dataset_config musubi_dataset.toml --t5 <t5_path>",
        "#",
        "# Then train:",
        "#   accelerate launch wan_train_network.py \\",
        "#     --task t2v-14B \\                          # or t2v-A14B, i2v-14B, i2v-A14B",
        "#     --dit <dit_path> \\                        # DiT model weights",
        "#     --dataset_config musubi_dataset.toml \\    # ← this file",
        "#     --network_module networks.lora_wan \\",
        "#     --network_dim 32 --network_alpha 32 \\     # LoRA rank and alpha",
        "#     --learning_rate 1e-4 \\",
        "#     --optimizer_type adamw8bit \\",
        "#     --max_train_epochs 50 \\",
        "#     --output_dir ./output --output_name my_lora",
        "#",
        "# See https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md",
        "# and https://github.com/alvdansen/lora-gym for full training templates.",
        "",
        "# ─── DATASET (generated — edit paths if you move files) ───",
        "",
        "[[datasets]]",
        f'resolution = [{wh[0]}, {wh[1]}]',
        f'caption_extension = ".txt"',
        f"enable_bucket = true",
        f"bucket_no_upscale = true",
        f'video_directory = "{video_dir}"',
        f'cache_directory = "{cache_dir}"',
    ]

    if ref_dir:
        lines.append(f'conditioning_image_directory = "{ref_dir}"')

    if frame_counts:
        fc_str = ", ".join(str(fc) for fc in frame_counts)
        lines.append(f"target_frames = [{fc_str}]")

    lines.append(f"num_repeats = {num_repeats}")
    lines.append("")
    lines.extend([
        "# ─── OPTIONAL: uncomment to customize ───",
        "# batch_size = 1",
        "# frame_extraction = \"head\"",
        "",
    ])

    content = "\n".join(lines)

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

    return output_path


# ---------------------------------------------------------------------------
# ai-toolkit
# ---------------------------------------------------------------------------

@register_trainer("aitoolkit")
def generate_aitoolkit_config(
    samples: list[OrganizedSample],
    output_dir: Path,
    config: KlippbokDataConfig,
    layout: OrganizeLayout,
    dry_run: bool = False,
) -> Path:
    """Generate an ai-toolkit training YAML file.

    ai-toolkit (ostris) uses a single YAML for everything: model, LoRA,
    training, dataset, and sampling. This generates a complete template
    with the dataset section filled in and TODO markers for training params.

    Args:
        samples: Organized samples.
        output_dir: Output directory.
        config: Klippbok data config.
        layout: Which organize layout was used.
        dry_run: If True, returns the path without writing.

    Returns:
        Path to the generated aitoolkit_config.yaml file.
    """
    output_path = output_dir / "aitoolkit_config.yaml"

    wh = _resolution_to_wh(config.video.resolution)
    video_dir = _video_directory(output_dir, layout)

    lines = [
        "# ai-toolkit training config",
        f"# Generated by: python -m klippbok.dataset organize <source> -o {_to_forward_slash(output_dir)} -t aitoolkit",
        "",
        "# ─── HOW TO USE THIS FILE ───",
        "# ai-toolkit uses a single YAML for everything: model, LoRA, training,",
        "# dataset, and sampling. The dataset section below is filled in for you.",
        "# Fill in the remaining sections marked with TODO before training.",
        "#",
        "# See https://github.com/ostris/ai-toolkit/tree/main/config/examples",
        "# and https://github.com/alvdansen/lora-gym for full training templates.",
        "",
        "---",
        "job: extension",
        "config:",
        "  name: \"my_lora_v1\"                              # TODO: your run name",
        "  process:",
        "    - type: 'sd_trainer'",
        "      training_folder: \"output\"",
        "      device: cuda:0",
        "",
        "      # ─── MODEL (TODO: set your model) ───",
        "      model:",
        "        name_or_path: \"Wan-AI/Wan2.1-T2V-14B-Diffusers\"  # TODO: your model",
        "        arch: 'wan21'                              # wan21 or wan22_14b",
        "        quantize: true",
        "        quantize_te: true",
        "        qtype_te: \"qfloat8\"",
        "        low_vram: true",
        "",
        "      # ─── LORA NETWORK (TODO: set rank) ───",
        "      network:",
        "        type: \"lora\"",
        "        linear: 32                                 # LoRA rank",
        "        linear_alpha: 32                           # LoRA alpha",
        "",
        "      # ─── TRAINING (TODO: set learning rate, steps) ───",
        "      train:",
        "        batch_size: 1",
        "        steps: 2000                                # TODO: training steps",
        "        gradient_accumulation: 1",
        "        train_unet: true",
        "        train_text_encoder: false",
        "        gradient_checkpointing: true",
        "        noise_scheduler: \"flowmatch\"",
        "        optimizer: \"adamw8bit\"",
        "        lr: 1e-4                                   # TODO: learning rate",
        "        dtype: bf16",
        "        cache_text_embeddings: true",
        "",
        "      # ─── SAVE ───",
        "      save:",
        "        dtype: float16",
        "        save_every: 250",
        "        max_step_saves_to_keep: 4",
        "",
        "      # ─── DATASET (generated — edit paths if you move files) ───",
        "      datasets:",
        f"        - folder_path: \"{video_dir}\"",
        f'          caption_ext: "txt"',
        f"          is_video: true",
        f"          resolution:",
        f"            - {wh[0]}",
        f"            - {wh[1]}",
        "",
        "      # ─── SAMPLING (TODO: set your prompts) ───",
        "      sample:",
        "        sampler: \"flowmatch\"",
        "        sample_every: 250",
        f"        width: {wh[0]}",
        f"        height: {wh[1]}",
        "        prompts:",
        "          - \"your sample prompt here\"              # TODO: sample prompts",
        "        seed: 42",
        "        guidance_scale: 5.0",
        "        sample_steps: 30",
        "",
    ]

    content = "\n".join(lines)

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

    return output_path
