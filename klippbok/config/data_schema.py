"""Pydantic v2 models for the Klippbok data config.

This is the schema that describes a dataset to Klippbok. It answers:
"What files do I have, how should they be handled, and what quality
standards should they meet?"

The data config lives with the dataset (typically as klippbok_data.yaml)
and works independently of any training config. The standalone data tools
(scene detection, captioning, image extraction) consume and produce this
config. It also works with external trainers like musubi-tuner or ai-toolkit.

Three tiers of complexity:
  - New user: just a path (everything else defaulted)
  - Standard: name, use_case, fps, resolution, anchor word, reference source
  - Internal: every field, including quality thresholds and bucketing
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from klippbok.config.defaults import (
    UMT5_MAX_TOKENS,
    VALID_BUCKETING_DIMENSIONS,
    VALID_DOWNSCALE_METHODS,
    VALID_FRAME_COUNTS,
    VALID_RESOLUTIONS,
    VALID_UPSCALE_POLICIES,
    VALID_SAR_POLICIES,
    VALID_TEXT_FORMATS,
    VALID_REFERENCE_SOURCES,
    VALID_USE_CASES,
    WAN_TRAINING_FPS,
)


# ─── Dataset Identity ───


class DatasetIdentityConfig(BaseModel):
    """Who this dataset is and what it's for.

    The identity block is optional — it's metadata for humans, not
    required by the pipeline. But it helps the captioner tool (Phase 2)
    make better decisions about what to include/omit in captions.
    """

    name: str | None = Field(
        default=None,
        description="Human-readable dataset name (e.g., 'annika', 'noir_style').",
    )
    use_case: Literal["character", "style", "motion", "object"] | None = Field(
        default=None,
        description=(
            "What kind of LoRA this dataset is for. Informs captioning strategy: "
            "'character' omits appearance, 'style' omits aesthetics, "
            "'motion' omits identity, 'object' omits the object."
        ),
    )
    description: str = Field(
        default="",
        description="Free-text description of what this dataset is for.",
    )

    @field_validator("use_case", mode="before")
    @classmethod
    def validate_use_case(cls, v: str | None) -> str | None:
        """Validate use_case against known values, with a helpful error."""
        if v is not None and v not in VALID_USE_CASES:
            valid_list = ", ".join(sorted(VALID_USE_CASES))
            raise ValueError(
                f"Invalid use_case '{v}'. "
                f"Valid options: {valid_list}. "
                f"Set to null if none of these apply."
            )
        return v


# ─── Dataset Sources ───


class DatasetSourceConfig(BaseModel):
    """A single dataset folder containing video clips.

    Multiple dataset sources can be combined in one config, each with
    its own settings for repeats and loss weighting. This lets you
    balance a small high-quality dataset against a larger supplementary
    one, or mark regularization data separately.
    """

    path: str = Field(
        description=(
            "Path to a folder of video clips. Can be relative (resolved "
            "from the config file's location) or absolute."
        ),
    )
    repeats: int = Field(
        default=1,
        ge=1,
        description=(
            "Times to repeat this dataset per epoch. Use this to balance "
            "datasets of different sizes (e.g., repeats=3 for a small "
            "dataset alongside a larger one)."
        ),
    )
    loss_multiplier: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Relative training weight for this dataset. 1.0 = normal. "
            "Use <1.0 to reduce influence, >1.0 to emphasize."
        ),
    )
    is_regularization: bool = Field(
        default=False,
        description=(
            "True = this is regularization data (preserves model generality). "
            "Regularization clips are trained on but don't teach the target "
            "concept — they keep the model from forgetting everything else."
        ),
    )


# ─── Video Specifications ───


class VideoConfig(BaseModel):
    """Video format requirements for training clips.

    These settings define what the pipeline expects from your video files.
    Clips that don't meet these specs will be flagged during validation.
    Defaults are tuned for Wan models (16 FPS, 480p, 4n+1 frame counts).
    """

    fps: int = Field(
        default=WAN_TRAINING_FPS,
        gt=0,
        description=(
            "Target frame rate. Default 16 = Wan's training FPS. "
            "Clips at different frame rates will be re-sampled."
        ),
    )
    resolution: int = Field(
        default=720,
        description=(
            "Target resolution tier (height in pixels). "
            "480 or 720 for Wan models."
        ),
    )
    frame_count: int | Literal["auto"] = Field(
        default="auto",
        description=(
            "Expected frame count per clip, or 'auto' to accept clip "
            "lengths as-is (each validated to 4n+1). If set explicitly, "
            "must be a 4n+1 value (1, 5, 9, 13, ..., 81, ...)."
        ),
    )
    upscale_policy: Literal["never", "warn"] = Field(
        default="never",
        description=(
            "How to handle clips below target resolution. "
            "'never' = reject (recommended — Wan was trained on downscale-only data). "
            "'warn' = allow with a warning."
        ),
    )
    sar_policy: Literal["auto_correct", "reject"] = Field(
        default="auto_correct",
        description=(
            "How to handle non-square pixel aspect ratios (SAR). "
            "'auto_correct' = resample to square pixels. "
            "'reject' = error on non-square SAR."
        ),
    )
    max_frames: int | None = Field(
        default=81,
        description=(
            "Maximum frame count per clip. Scenes longer than this are "
            "subdivided into shorter clips. Default 81 = ~5s at 16fps, "
            "which is the longest clip length for standard Wan training. "
            "Set to null for no limit."
        ),
    )
    downscale_method: Literal["lanczos", "bicubic", "bilinear", "area"] = Field(
        default="lanczos",
        description=(
            "Scaling algorithm for downscaling clips above target resolution. "
            "'lanczos' = sharp, high quality (recommended for training data). "
            "'bicubic' = smooth, fewer ringing artifacts. "
            "'bilinear' = fast but soft. "
            "'area' = pixel averaging, good for very large downscale factors."
        ),
    )

    @field_validator("resolution", mode="before")
    @classmethod
    def validate_resolution(cls, v: int) -> int:
        if v not in VALID_RESOLUTIONS:
            valid_list = ", ".join(str(r) for r in sorted(VALID_RESOLUTIONS))
            raise ValueError(
                f"Resolution {v} is not a supported tier. "
                f"Valid resolutions: {valid_list}. "
                f"Wan models were trained at these resolutions."
            )
        return v

    @field_validator("frame_count", mode="before")
    @classmethod
    def validate_frame_count(cls, v: int | str) -> int | str:
        """Frame count must be 4n+1 (Wan VAE requirement) or 'auto'."""
        if isinstance(v, str):
            if v != "auto":
                raise ValueError(
                    f"frame_count must be an integer or 'auto', got '{v}'."
                )
            return v
        if v not in VALID_FRAME_COUNTS:
            # Find the nearest valid counts to help the user
            lower = max((fc for fc in VALID_FRAME_COUNTS if fc <= v), default=1)
            upper = min((fc for fc in VALID_FRAME_COUNTS if fc >= v), default=None)
            suggestion = f"Nearest valid counts: {lower}"
            if upper is not None and upper != lower:
                suggestion += f" or {upper}"
            raise ValueError(
                f"Frame count {v} is not valid for Wan's VAE. "
                f"Must be 4n+1 (1, 5, 9, 13, ..., 81, ...). "
                f"{suggestion}."
            )
        return v

    @field_validator("upscale_policy", mode="before")
    @classmethod
    def validate_upscale_policy(cls, v: str) -> str:
        if v not in VALID_UPSCALE_POLICIES:
            valid_list = ", ".join(sorted(VALID_UPSCALE_POLICIES))
            raise ValueError(
                f"Invalid upscale_policy '{v}'. Valid options: {valid_list}."
            )
        return v

    @field_validator("sar_policy", mode="before")
    @classmethod
    def validate_sar_policy(cls, v: str) -> str:
        if v not in VALID_SAR_POLICIES:
            valid_list = ", ".join(sorted(VALID_SAR_POLICIES))
            raise ValueError(
                f"Invalid sar_policy '{v}'. Valid options: {valid_list}."
            )
        return v

    @field_validator("downscale_method", mode="before")
    @classmethod
    def validate_downscale_method(cls, v: str) -> str:
        if v not in VALID_DOWNSCALE_METHODS:
            valid_list = ", ".join(sorted(VALID_DOWNSCALE_METHODS))
            raise ValueError(
                f"Invalid downscale_method '{v}'. "
                f"Valid options: {valid_list}."
            )
        return v


# ─── Control Signals: Text ───


class TextControlConfig(BaseModel):
    """Caption / text control signal configuration.

    Captions are ONE control signal, not THE control signal. They tell the
    model what's happening in the video — the semantic content. Other control
    signals (reference images, depth maps, etc.) tell the model other things.

    Two formats supported:
      - txt: one .txt sidecar file per clip (clip_001.mp4 → clip_001.txt)
      - jsonl: all captions in one file, each line {"file": ..., "caption": ...}

    Anchor word (Minta's terminology): a natural-language term prepended to
    every caption. Reads naturally: "annika A girl walks through a garden..."
    NOT token gibberish like "sks" or "strbmn". Set to null for style LoRAs.
    """

    format: Literal["txt", "jsonl"] = Field(
        default="txt",
        description=(
            "Caption file format. 'txt' = one .txt sidecar per clip. "
            "'jsonl' = consolidated file with {\"file\": ..., \"caption\": ...} per line."
        ),
    )
    jsonl_file: str | None = Field(
        default=None,
        description=(
            "Path to JSONL caption file (when format='jsonl'). "
            "Relative to config file location. null = auto-discover "
            "a .jsonl file in the dataset folder."
        ),
    )
    anchor_word: str | None = Field(
        default=None,
        description=(
            "Natural-language anchor word prepended to every caption. "
            "Example: 'annika' → 'annika A girl walks through a garden...' "
            "Set to null for style LoRAs or when no anchor is needed."
        ),
    )
    default_caption: str = Field(
        default="",
        description=(
            "Fallback caption for clips with no caption file. "
            "Empty string = no fallback (behavior depends on 'required')."
        ),
    )
    required: bool = Field(
        default=True,
        description=(
            "True = error if any clip has no caption. "
            "False = allow uncaptioned clips (they'll use default_caption if set, "
            "otherwise train without text conditioning for those clips)."
        ),
    )
    max_tokens: int = Field(
        default=UMT5_MAX_TOKENS,
        gt=0,
        description=(
            "Maximum token count for captions. Validation warns if exceeded. "
            "Default 512 = UMT5 encoder limit for Wan models."
        ),
    )
    shuffle_tokens: bool = Field(
        default=False,
        description=(
            "Randomize comma-separated token order in captions each epoch. "
            "Prevents the model from learning caption order as a signal. "
            "Preserves the first 'keep_tokens' tokens (e.g., anchor word)."
        ),
    )
    keep_tokens: int = Field(
        default=1,
        ge=0,
        description=(
            "Number of leading comma-separated tokens to preserve when "
            "shuffle_tokens is enabled. Default 1 keeps the anchor word first."
        ),
    )
    token_dropout_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Per-token removal probability (0.0–1.0). Each comma-separated "
            "token has this chance of being dropped during training. "
            "0.0 = no dropout. Preserves keep_tokens."
        ),
    )

    @field_validator("format", mode="before")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v not in VALID_TEXT_FORMATS:
            valid_list = ", ".join(sorted(VALID_TEXT_FORMATS))
            raise ValueError(
                f"Invalid text format '{v}'. Valid options: {valid_list}."
            )
        return v


# ─── Control Signals: Images ───


class ReferenceImageConfig(BaseModel):
    """Reference image configuration for I2V training.

    Reference images tell the model "make video starting from this frame."
    In Wan I2V, the reference image is VAE-encoded and concatenated with
    the noisy latents in latent space (16 extra channels → 36 total).

    Three sourcing strategies:
      - first_frame: auto-extract from each clip (Phase 3 tool does this)
      - folder: look in a specified folder for stem-matched images
      - none: no reference images (T2V training)
    """

    source: Literal["first_frame", "folder", "none"] = Field(
        default="none",
        description=(
            "'first_frame' = extract first frame from each clip. "
            "'folder' = load from a specified folder (stem-matched filenames). "
            "'none' = no reference images."
        ),
    )
    folder: str | None = Field(
        default=None,
        description=(
            "Path to folder containing reference images (when source='folder'). "
            "Images are matched to clips by filename stem: clip_001.mp4 → clip_001.png. "
            "Relative to config file location."
        ),
    )
    required: bool = Field(
        default=False,
        description=(
            "True = error if any clip is missing a reference image. "
            "False = skip I2V conditioning for clips without references."
        ),
    )

    @field_validator("source", mode="before")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if v not in VALID_REFERENCE_SOURCES:
            valid_list = ", ".join(sorted(VALID_REFERENCE_SOURCES))
            raise ValueError(
                f"Invalid reference source '{v}'. Valid options: {valid_list}."
            )
        return v


class ImagesControlConfig(BaseModel):
    """Image-based control signals.

    Currently supports reference images for I2V conditioning. Future
    extensions will add subject references (character identity), style
    references (aesthetic guidance), and structural guides (depth, edge, pose).
    """

    reference: ReferenceImageConfig = Field(
        default_factory=ReferenceImageConfig,
        description="Reference image configuration for I2V conditioning.",
    )
    # Future extensions:
    # subject: SubjectImageConfig  — character/object identity references
    # style: StyleImageConfig      — aesthetic/mood references


class ControlsConfig(BaseModel):
    """All control signal configurations.

    Control signals are everything the model learns to OBEY — as opposed to
    video, which is what the model learns to PRODUCE. Each signal has its
    own preparation, validation, and encoding pathway.
    """

    text: TextControlConfig = Field(
        default_factory=TextControlConfig,
        description="Caption / text control signal settings.",
    )
    images: ImagesControlConfig = Field(
        default_factory=ImagesControlConfig,
        description="Image-based control signal settings.",
    )
    # Future:
    # structural: StructuralControlConfig  — depth, edge, pose maps


# ─── Quality Thresholds ───


class MotionQualityConfig(BaseModel):
    """Motion quality thresholds.

    These are optional filters for automated quality control. Set to null
    to skip. The exact metrics are defined by the scene detection tool
    (Phase 1) — here we just set the acceptance thresholds.
    """

    min_intensity: float | None = Field(
        default=None,
        description=(
            "Reject clips with motion below this intensity (static shots). "
            "null = don't filter by minimum motion."
        ),
    )
    max_intensity: float | None = Field(
        default=None,
        description=(
            "Reject clips with motion above this intensity (chaotic/unusable). "
            "null = don't filter by maximum motion."
        ),
    )


class QualityConfig(BaseModel):
    """Quality thresholds for dataset validation.

    All thresholds are optional — set to null to skip that check.
    These are applied during dataset validation (Phase 4), not during
    training. The idea: catch bad data before you burn GPU time.
    """

    min_resolution: int = Field(
        default=720,
        gt=0,
        description="Reject clips below this resolution (height in pixels).",
    )
    blur_threshold: float | None = Field(
        default=None,
        description=(
            "Laplacian variance minimum. Lower = blurrier. "
            "null = skip blur detection."
        ),
    )
    exposure_range: tuple[float, float] | None = Field(
        default=None,
        description=(
            "[low, high] acceptable brightness range (0.0–1.0). "
            "null = skip exposure check."
        ),
    )
    motion: MotionQualityConfig = Field(
        default_factory=MotionQualityConfig,
        description="Motion quality thresholds.",
    )
    check_duplicates: bool = Field(
        default=False,
        description=(
            "Enable perceptual duplicate detection via dHash. "
            "Opt-in because it requires loading every reference image. "
            "When enabled, flags near-duplicate reference images within "
            "and across dataset sources."
        ),
    )


# ─── Bucketing ───


class BucketingConfig(BaseModel):
    """Bucketing configuration for batch construction.

    Training requires batches of clips with compatible dimensions.
    Bucketing groups clips by aspect ratio, frame count, and/or resolution
    so each batch is uniform. This is 3D bucketing — more sophisticated
    than the 1D (aspect ratio only) bucketing in most image trainers.
    """

    dimensions: list[str] = Field(
        default=["aspect_ratio", "frame_count", "resolution"],
        description=(
            "Properties to group clips by. Order doesn't matter. "
            "Valid: aspect_ratio, frame_count, resolution."
        ),
    )
    aspect_ratio_tolerance: float = Field(
        default=0.1,
        gt=0.0,
        description=(
            "How close aspect ratios need to be to share a bucket. "
            "0.1 means ratios within 10% of each other are grouped."
        ),
    )
    min_bucket_size: int = Field(
        default=2,
        ge=1,
        description=(
            "Minimum clips per bucket. Buckets with fewer clips "
            "generate a warning (they'll waste GPU time with padding)."
        ),
    )

    @field_validator("dimensions", mode="before")
    @classmethod
    def validate_dimensions(cls, v: list[str]) -> list[str]:
        invalid = set(v) - VALID_BUCKETING_DIMENSIONS
        if invalid:
            valid_list = ", ".join(sorted(VALID_BUCKETING_DIMENSIONS))
            invalid_list = ", ".join(sorted(invalid))
            raise ValueError(
                f"Invalid bucketing dimensions: {invalid_list}. "
                f"Valid options: {valid_list}."
            )
        return v


# ─── Metadata ───


class MetadataConfig(BaseModel):
    """Dataset metadata for organization and provenance tracking."""

    source: str = Field(
        default="",
        description="Where this data came from (film source, client footage, etc.).",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Arbitrary tags for organization and filtering.",
    )


# ─── Top-Level Config ───


class KlippbokDataConfig(BaseModel):
    """Root config model for a Klippbok dataset.

    This is the complete schema for klippbok_data.yaml. Minimum viable
    config is just a dataset path — everything else has sensible defaults
    tuned for Wan model training.

    The config supports multiple dataset folders, each with independent
    settings for repeats, loss weighting, and regularization.

    Example minimal config::

        dataset:
          path: ./video_clips

    Example standard config::

        dataset:
          name: annika
          use_case: character
        datasets:
          - path: ./video_clips
        video:
          fps: 16
          resolution: 720
        controls:
          text:
            anchor_word: annika
          images:
            reference:
              source: first_frame
    """

    dataset: DatasetIdentityConfig = Field(
        default_factory=DatasetIdentityConfig,
        description="Dataset identity and metadata.",
    )
    datasets: list[DatasetSourceConfig] = Field(
        default_factory=list,
        description=(
            "List of dataset source folders. Each can have independent "
            "settings for repeats, loss weight, and regularization."
        ),
    )
    video: VideoConfig = Field(
        default_factory=VideoConfig,
        description="Video format requirements.",
    )
    controls: ControlsConfig = Field(
        default_factory=ControlsConfig,
        description="Control signal configurations.",
    )
    quality: QualityConfig = Field(
        default_factory=QualityConfig,
        description="Quality thresholds for validation.",
    )
    bucketing: BucketingConfig = Field(
        default_factory=BucketingConfig,
        description="Bucketing settings for batch construction.",
    )
    metadata: MetadataConfig = Field(
        default_factory=MetadataConfig,
        description="Dataset metadata and provenance.",
    )

    @model_validator(mode="after")
    def check_datasets_not_empty(self) -> KlippbokDataConfig:
        """Ensure at least one dataset source is defined.

        This runs after all field parsing, so the loader has already had
        a chance to convert dataset.path into a datasets entry.
        An empty datasets list means the user didn't provide any path at all.
        """
        if not self.datasets:
            raise ValueError(
                "No dataset paths provided. Add at least one entry to 'datasets', "
                "or set 'dataset.path' to your video clips folder.\n"
                "Example:\n"
                "  datasets:\n"
                "    - path: ./video_clips"
            )
        return self
