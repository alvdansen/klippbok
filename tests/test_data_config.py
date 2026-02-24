"""Tests for the Klippbok data config schema, loader, and validation.

Covers:
  - Loading minimal, standard, and full configs
  - Default values applied correctly
  - Multi-dataset support
  - Backwards compatibility (dataset.path shorthand)
  - All validation errors with helpful messages
  - Path resolution (relative paths)
  - Config file discovery in directories
  - Control signal settings (anchor word, JSONL, reference images)
  - Bucketing, quality, metadata
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from klippbok.config.data_schema import KlippbokDataConfig
from klippbok.config.defaults import UMT5_MAX_TOKENS, WAN_TRAINING_FPS
from klippbok.config.loader import KlippbokConfigError, load_data_config


# ─── Fixtures ───


@pytest.fixture
def tmp_dataset(tmp_path: Path) -> Path:
    """Create a minimal dataset directory with a video file placeholder."""
    dataset_dir = tmp_path / "video_clips"
    dataset_dir.mkdir()
    # Create a dummy video file (content doesn't matter for config tests)
    (dataset_dir / "clip_001.mp4").touch()
    (dataset_dir / "clip_001.txt").write_text("annika A girl walks through a garden")
    return dataset_dir


@pytest.fixture
def tmp_multi_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Create two dataset directories for multi-dataset testing."""
    main_dir = tmp_path / "main_clips"
    main_dir.mkdir()
    (main_dir / "clip_001.mp4").touch()

    supp_dir = tmp_path / "supplementary_clips"
    supp_dir.mkdir()
    (supp_dir / "extra_001.mp4").touch()

    return main_dir, supp_dir


def write_config(path: Path, data: dict) -> Path:
    """Write a YAML config dict to a file and return the file path."""
    config_path = path / "klippbok_data.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)
    return config_path


# ─── Loading: Minimal Config ───


class TestMinimalConfig:
    """Minimum viable config = just a path. Everything else defaults."""

    def test_load_minimal_yaml(self, tmp_path: Path, tmp_dataset: Path):
        """A config with only dataset.path loads with all defaults."""
        config_path = write_config(tmp_path, {
            "dataset": {"path": str(tmp_dataset)},
        })
        config = load_data_config(config_path)

        # Dataset source was created from the shorthand
        assert len(config.datasets) == 1
        assert Path(config.datasets[0].path) == tmp_dataset.resolve()
        assert config.datasets[0].repeats == 1
        assert config.datasets[0].loss_multiplier == 1.0
        assert config.datasets[0].is_regularization is False

    def test_defaults_video(self, tmp_path: Path, tmp_dataset: Path):
        """Video defaults match Wan model training parameters."""
        config_path = write_config(tmp_path, {
            "dataset": {"path": str(tmp_dataset)},
        })
        config = load_data_config(config_path)

        assert config.video.fps == WAN_TRAINING_FPS
        assert config.video.resolution == 720
        assert config.video.frame_count == "auto"
        assert config.video.upscale_policy == "never"
        assert config.video.sar_policy == "auto_correct"
        assert config.video.downscale_method == "lanczos"

    def test_defaults_text_controls(self, tmp_path: Path, tmp_dataset: Path):
        """Text control defaults are sensible for most users."""
        config_path = write_config(tmp_path, {
            "dataset": {"path": str(tmp_dataset)},
        })
        config = load_data_config(config_path)

        text = config.controls.text
        assert text.format == "txt"
        assert text.jsonl_file is None
        assert text.anchor_word is None
        assert text.default_caption == ""
        assert text.required is True
        assert text.max_tokens == UMT5_MAX_TOKENS
        assert text.shuffle_tokens is False
        assert text.keep_tokens == 1
        assert text.token_dropout_rate == 0.0

    def test_defaults_image_controls(self, tmp_path: Path, tmp_dataset: Path):
        """Image controls default to no reference images (T2V training)."""
        config_path = write_config(tmp_path, {
            "dataset": {"path": str(tmp_dataset)},
        })
        config = load_data_config(config_path)

        ref = config.controls.images.reference
        assert ref.source == "none"
        assert ref.folder is None
        assert ref.required is False

    def test_defaults_quality(self, tmp_path: Path, tmp_dataset: Path):
        """Quality thresholds default to minimal checks."""
        config_path = write_config(tmp_path, {
            "dataset": {"path": str(tmp_dataset)},
        })
        config = load_data_config(config_path)

        assert config.quality.min_resolution == 720
        assert config.quality.blur_threshold is None
        assert config.quality.exposure_range is None
        assert config.quality.motion.min_intensity is None
        assert config.quality.motion.max_intensity is None

    def test_defaults_bucketing(self, tmp_path: Path, tmp_dataset: Path):
        """Bucketing defaults to 3D bucketing with reasonable tolerance."""
        config_path = write_config(tmp_path, {
            "dataset": {"path": str(tmp_dataset)},
        })
        config = load_data_config(config_path)

        assert set(config.bucketing.dimensions) == {
            "aspect_ratio", "frame_count", "resolution"
        }
        assert config.bucketing.aspect_ratio_tolerance == 0.1
        assert config.bucketing.min_bucket_size == 2

    def test_load_from_directory(self, tmp_dataset: Path):
        """Loading a directory without a config creates a minimal config."""
        config = load_data_config(tmp_dataset)

        assert len(config.datasets) == 1
        assert Path(config.datasets[0].path) == tmp_dataset.resolve()

    def test_discover_config_in_directory(self, tmp_path: Path, tmp_dataset: Path):
        """Loading a directory finds klippbok_data.yaml inside it."""
        write_config(tmp_path, {
            "dataset": {
                "name": "test_discovery",
                "path": str(tmp_dataset),
            },
        })
        config = load_data_config(tmp_path)

        assert config.dataset.name == "test_discovery"
        assert len(config.datasets) == 1


# ─── Loading: Standard Config ───


class TestStandardConfig:
    """Standard tier: name, use_case, video settings, anchor word, references."""

    def test_standard_fields(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "dataset": {
                "name": "annika",
                "use_case": "character",
            },
            "datasets": [{"path": str(tmp_dataset)}],
            "video": {
                "fps": 16,
                "resolution": 480,
            },
            "controls": {
                "text": {"anchor_word": "annika"},
                "images": {
                    "reference": {"source": "first_frame"},
                },
            },
        })
        config = load_data_config(config_path)

        assert config.dataset.name == "annika"
        assert config.dataset.use_case == "character"
        assert config.video.fps == 16
        assert config.video.resolution == 480
        assert config.controls.text.anchor_word == "annika"
        assert config.controls.images.reference.source == "first_frame"

    def test_use_case_style(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "dataset": {"use_case": "style"},
            "datasets": [{"path": str(tmp_dataset)}],
        })
        config = load_data_config(config_path)
        assert config.dataset.use_case == "style"

    def test_use_case_motion(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "dataset": {"use_case": "motion"},
            "datasets": [{"path": str(tmp_dataset)}],
        })
        config = load_data_config(config_path)
        assert config.dataset.use_case == "motion"

    def test_use_case_object(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "dataset": {"use_case": "object"},
            "datasets": [{"path": str(tmp_dataset)}],
        })
        config = load_data_config(config_path)
        assert config.dataset.use_case == "object"

    def test_use_case_null(self, tmp_path: Path, tmp_dataset: Path):
        """null use_case is valid — not every dataset fits a category."""
        config_path = write_config(tmp_path, {
            "dataset": {"use_case": None},
            "datasets": [{"path": str(tmp_dataset)}],
        })
        config = load_data_config(config_path)
        assert config.dataset.use_case is None

    def test_anchor_word_null(self, tmp_path: Path, tmp_dataset: Path):
        """Null anchor word = no prepending (style LoRAs, etc.)."""
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "controls": {"text": {"anchor_word": None}},
        })
        config = load_data_config(config_path)
        assert config.controls.text.anchor_word is None


# ─── Loading: Full Config ───


class TestFullConfig:
    """Full tier: every field specified, including JSONL, token ops, multi-dataset."""

    def test_full_config(self, tmp_path: Path, tmp_multi_dataset: tuple[Path, Path]):
        main_dir, supp_dir = tmp_multi_dataset

        # Create a reference images folder
        ref_dir = tmp_path / "reference_images"
        ref_dir.mkdir()
        (ref_dir / "clip_001.png").touch()

        config_path = write_config(tmp_path, {
            "dataset": {
                "name": "full_test",
                "use_case": "character",
                "description": "Full config test dataset",
            },
            "datasets": [
                {
                    "path": str(main_dir),
                    "repeats": 1,
                    "loss_multiplier": 1.0,
                    "is_regularization": False,
                },
                {
                    "path": str(supp_dir),
                    "repeats": 3,
                    "loss_multiplier": 0.5,
                    "is_regularization": True,
                },
            ],
            "video": {
                "fps": 16,
                "resolution": 720,
                "frame_count": 81,
                "upscale_policy": "warn",
                "sar_policy": "reject",
            },
            "controls": {
                "text": {
                    "format": "txt",
                    "anchor_word": "annika",
                    "default_caption": "a person",
                    "required": False,
                    "max_tokens": 256,
                    "shuffle_tokens": True,
                    "keep_tokens": 2,
                    "token_dropout_rate": 0.1,
                },
                "images": {
                    "reference": {
                        "source": "folder",
                        "folder": str(ref_dir),
                        "required": True,
                    },
                },
            },
            "quality": {
                "min_resolution": 720,
                "blur_threshold": 100.0,
                "exposure_range": [0.1, 0.9],
                "motion": {
                    "min_intensity": 0.5,
                    "max_intensity": 50.0,
                },
            },
            "bucketing": {
                "dimensions": ["aspect_ratio", "frame_count"],
                "aspect_ratio_tolerance": 0.05,
                "min_bucket_size": 4,
            },
            "metadata": {
                "source": "film source",
                "tags": ["anime", "high_quality"],
            },
        })
        config = load_data_config(config_path)

        # Identity
        assert config.dataset.name == "full_test"
        assert config.dataset.use_case == "character"
        assert config.dataset.description == "Full config test dataset"

        # Multiple datasets
        assert len(config.datasets) == 2
        assert config.datasets[0].repeats == 1
        assert config.datasets[0].loss_multiplier == 1.0
        assert config.datasets[0].is_regularization is False
        assert config.datasets[1].repeats == 3
        assert config.datasets[1].loss_multiplier == 0.5
        assert config.datasets[1].is_regularization is True

        # Video
        assert config.video.resolution == 720
        assert config.video.frame_count == 81
        assert config.video.upscale_policy == "warn"
        assert config.video.sar_policy == "reject"

        # Text controls
        assert config.controls.text.anchor_word == "annika"
        assert config.controls.text.default_caption == "a person"
        assert config.controls.text.required is False
        assert config.controls.text.max_tokens == 256
        assert config.controls.text.shuffle_tokens is True
        assert config.controls.text.keep_tokens == 2
        assert config.controls.text.token_dropout_rate == 0.1

        # Image controls
        assert config.controls.images.reference.source == "folder"
        assert config.controls.images.reference.required is True

        # Quality
        assert config.quality.min_resolution == 720
        assert config.quality.blur_threshold == 100.0
        assert config.quality.exposure_range == (0.1, 0.9)
        assert config.quality.motion.min_intensity == 0.5
        assert config.quality.motion.max_intensity == 50.0

        # Bucketing
        assert config.bucketing.dimensions == ["aspect_ratio", "frame_count"]
        assert config.bucketing.aspect_ratio_tolerance == 0.05
        assert config.bucketing.min_bucket_size == 4

        # Metadata
        assert config.metadata.source == "film source"
        assert config.metadata.tags == ["anime", "high_quality"]


# ─── Multi-Dataset ───


class TestMultiDataset:
    """Multiple dataset folders with per-dataset settings."""

    def test_multi_dataset_parsed(
        self, tmp_path: Path, tmp_multi_dataset: tuple[Path, Path]
    ):
        main_dir, supp_dir = tmp_multi_dataset
        config_path = write_config(tmp_path, {
            "datasets": [
                {"path": str(main_dir), "repeats": 1},
                {"path": str(supp_dir), "repeats": 5, "loss_multiplier": 0.3},
            ],
        })
        config = load_data_config(config_path)

        assert len(config.datasets) == 2
        assert config.datasets[0].repeats == 1
        assert config.datasets[0].loss_multiplier == 1.0  # default
        assert config.datasets[1].repeats == 5
        assert config.datasets[1].loss_multiplier == 0.3

    def test_per_dataset_regularization(
        self, tmp_path: Path, tmp_multi_dataset: tuple[Path, Path]
    ):
        main_dir, supp_dir = tmp_multi_dataset
        config_path = write_config(tmp_path, {
            "datasets": [
                {"path": str(main_dir), "is_regularization": False},
                {"path": str(supp_dir), "is_regularization": True},
            ],
        })
        config = load_data_config(config_path)

        assert config.datasets[0].is_regularization is False
        assert config.datasets[1].is_regularization is True


# ─── Backwards Compatibility ───


class TestBackwardsCompat:
    """dataset.path shorthand auto-wraps into datasets list."""

    def test_dataset_path_wrapped(self, tmp_path: Path, tmp_dataset: Path):
        """dataset.path creates a single-item datasets list."""
        config_path = write_config(tmp_path, {
            "dataset": {
                "name": "annika",
                "path": str(tmp_dataset),
            },
        })
        config = load_data_config(config_path)

        assert config.dataset.name == "annika"
        assert len(config.datasets) == 1
        assert Path(config.datasets[0].path) == tmp_dataset.resolve()

    def test_dataset_path_ignored_when_datasets_exists(
        self, tmp_path: Path, tmp_dataset: Path
    ):
        """If both dataset.path and datasets[] exist, datasets[] wins."""
        config_path = write_config(tmp_path, {
            "dataset": {
                "name": "annika",
                "path": "/some/old/path",  # should be ignored
            },
            "datasets": [{"path": str(tmp_dataset)}],
        })
        config = load_data_config(config_path)

        # datasets[] takes precedence
        assert len(config.datasets) == 1
        assert Path(config.datasets[0].path) == tmp_dataset.resolve()


# ─── Path Resolution ───


class TestPathResolution:
    """Relative paths resolve from the config file's directory."""

    def test_relative_dataset_path(self, tmp_path: Path):
        """Relative dataset path resolves from config file location."""
        dataset_dir = tmp_path / "my_clips"
        dataset_dir.mkdir()
        (dataset_dir / "clip.mp4").touch()

        config_path = write_config(tmp_path, {
            "datasets": [{"path": "./my_clips"}],
        })
        config = load_data_config(config_path)

        assert Path(config.datasets[0].path) == dataset_dir.resolve()

    def test_absolute_dataset_path(self, tmp_path: Path, tmp_dataset: Path):
        """Absolute paths are kept as-is."""
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
        })
        config = load_data_config(config_path)

        assert Path(config.datasets[0].path) == tmp_dataset.resolve()

    def test_relative_reference_folder(self, tmp_path: Path, tmp_dataset: Path):
        """Reference image folder path resolves relative to config."""
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()

        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "controls": {
                "images": {
                    "reference": {"source": "folder", "folder": "./refs"},
                },
            },
        })
        config = load_data_config(config_path)

        assert Path(config.controls.images.reference.folder) == ref_dir.resolve()

    def test_relative_jsonl_path(self, tmp_path: Path, tmp_dataset: Path):
        """JSONL file path resolves relative to config."""
        jsonl_file = tmp_path / "captions.jsonl"
        jsonl_file.touch()

        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "controls": {
                "text": {"format": "jsonl", "jsonl_file": "./captions.jsonl"},
            },
        })
        config = load_data_config(config_path)

        assert Path(config.controls.text.jsonl_file) == jsonl_file.resolve()


# ─── JSONL Format ───


class TestJsonlFormat:
    """JSONL caption format handling."""

    def test_jsonl_format_accepted(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "controls": {"text": {"format": "jsonl"}},
        })
        config = load_data_config(config_path)
        assert config.controls.text.format == "jsonl"

    def test_jsonl_file_null_by_default(self, tmp_path: Path, tmp_dataset: Path):
        """When jsonl_file is null, auto-discovery will happen at runtime."""
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "controls": {"text": {"format": "jsonl"}},
        })
        config = load_data_config(config_path)
        assert config.controls.text.jsonl_file is None


# ─── Validation Errors ───


class TestValidationErrors:
    """Every validation error should tell you what's wrong AND how to fix it."""

    def test_invalid_frame_count(self, tmp_path: Path, tmp_dataset: Path):
        """Frame count not 4n+1 gives helpful error with nearest valid values."""
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "video": {"frame_count": 80},
        })
        with pytest.raises(KlippbokConfigError, match="4n.1"):
            load_data_config(config_path)

    def test_invalid_frame_count_suggests_nearest(
        self, tmp_path: Path, tmp_dataset: Path
    ):
        """Error message includes nearest valid frame counts."""
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "video": {"frame_count": 80},
        })
        with pytest.raises(KlippbokConfigError, match="77.*81"):
            load_data_config(config_path)

    def test_invalid_resolution(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "video": {"resolution": 1080},
        })
        with pytest.raises(KlippbokConfigError, match="480.*720"):
            load_data_config(config_path)

    def test_invalid_use_case(self, tmp_path: Path, tmp_dataset: Path):
        """Invalid use_case lists valid options."""
        config_path = write_config(tmp_path, {
            "dataset": {"use_case": "landscape"},
            "datasets": [{"path": str(tmp_dataset)}],
        })
        with pytest.raises(KlippbokConfigError, match="character.*motion.*object.*style"):
            load_data_config(config_path)

    def test_invalid_text_format(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "controls": {"text": {"format": "csv"}},
        })
        with pytest.raises(KlippbokConfigError, match="jsonl.*txt"):
            load_data_config(config_path)

    def test_invalid_upscale_policy(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "video": {"upscale_policy": "always"},
        })
        with pytest.raises(KlippbokConfigError, match="never.*warn"):
            load_data_config(config_path)

    def test_invalid_sar_policy(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "video": {"sar_policy": "ignore"},
        })
        with pytest.raises(KlippbokConfigError, match="auto_correct.*reject"):
            load_data_config(config_path)

    def test_invalid_token_dropout_rate_too_high(
        self, tmp_path: Path, tmp_dataset: Path
    ):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "controls": {"text": {"token_dropout_rate": 1.5}},
        })
        with pytest.raises(KlippbokConfigError, match="token_dropout_rate"):
            load_data_config(config_path)

    def test_invalid_token_dropout_rate_negative(
        self, tmp_path: Path, tmp_dataset: Path
    ):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "controls": {"text": {"token_dropout_rate": -0.1}},
        })
        with pytest.raises(KlippbokConfigError, match="token_dropout_rate"):
            load_data_config(config_path)

    def test_invalid_loss_multiplier_zero(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset), "loss_multiplier": 0.0}],
        })
        with pytest.raises(KlippbokConfigError, match="loss_multiplier"):
            load_data_config(config_path)

    def test_invalid_loss_multiplier_negative(
        self, tmp_path: Path, tmp_dataset: Path
    ):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset), "loss_multiplier": -1.0}],
        })
        with pytest.raises(KlippbokConfigError, match="loss_multiplier"):
            load_data_config(config_path)

    def test_invalid_repeats_zero(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset), "repeats": 0}],
        })
        with pytest.raises(KlippbokConfigError, match="repeats"):
            load_data_config(config_path)

    def test_missing_dataset_path(self, tmp_path: Path):
        """No datasets at all gives a helpful error."""
        config_path = write_config(tmp_path, {
            "dataset": {"name": "orphan"},
        })
        with pytest.raises(KlippbokConfigError, match="No dataset paths"):
            load_data_config(config_path)

    def test_nonexistent_dataset_path(self, tmp_path: Path):
        """Path that doesn't exist on disk gives a clear error."""
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_path / "does_not_exist")}],
        })
        with pytest.raises(KlippbokConfigError, match="not found"):
            load_data_config(config_path)

    def test_nonexistent_config_file(self):
        """Trying to load a file that doesn't exist."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_data_config("/nonexistent/path/config.yaml")

    def test_invalid_reference_source(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "controls": {
                "images": {"reference": {"source": "magic"}},
            },
        })
        with pytest.raises(KlippbokConfigError, match="first_frame.*folder.*none"):
            load_data_config(config_path)

    def test_invalid_bucketing_dimension(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "bucketing": {"dimensions": ["aspect_ratio", "color"]},
        })
        with pytest.raises(KlippbokConfigError, match="color"):
            load_data_config(config_path)

    def test_invalid_downscale_method(self, tmp_path: Path, tmp_dataset: Path):
        """Invalid downscale_method lists valid options."""
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "video": {"downscale_method": "nearest"},
        })
        with pytest.raises(
            KlippbokConfigError, match="area.*bicubic.*bilinear.*lanczos"
        ):
            load_data_config(config_path)

    def test_invalid_fps_zero(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "video": {"fps": 0},
        })
        with pytest.raises(KlippbokConfigError, match="fps"):
            load_data_config(config_path)


# ─── SAR Policy ───


class TestDownscaleMethod:
    """All downscale method values are accepted correctly."""

    @pytest.mark.parametrize("method", ["lanczos", "bicubic", "bilinear", "area"])
    def test_valid_methods(
        self, tmp_path: Path, tmp_dataset: Path, method: str
    ):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "video": {"downscale_method": method},
        })
        config = load_data_config(config_path)
        assert config.video.downscale_method == method

    def test_default_is_lanczos(self, tmp_path: Path, tmp_dataset: Path):
        """Default downscale method is lanczos (best quality for training data)."""
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
        })
        config = load_data_config(config_path)
        assert config.video.downscale_method == "lanczos"


class TestSarPolicy:
    """Both SAR policy values are accepted correctly."""

    def test_auto_correct(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "video": {"sar_policy": "auto_correct"},
        })
        config = load_data_config(config_path)
        assert config.video.sar_policy == "auto_correct"

    def test_reject(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "video": {"sar_policy": "reject"},
        })
        config = load_data_config(config_path)
        assert config.video.sar_policy == "reject"


# ─── Valid Frame Counts ───


class TestFrameCounts:
    """Frame count validation enforces the 4n+1 rule."""

    @pytest.mark.parametrize("fc", [1, 5, 9, 13, 17, 21, 41, 81, 161])
    def test_valid_frame_counts(
        self, tmp_path: Path, tmp_dataset: Path, fc: int
    ):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "video": {"frame_count": fc},
        })
        config = load_data_config(config_path)
        assert config.video.frame_count == fc

    @pytest.mark.parametrize("fc", [2, 3, 4, 10, 80, 82, 100])
    def test_invalid_frame_counts(
        self, tmp_path: Path, tmp_dataset: Path, fc: int
    ):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "video": {"frame_count": fc},
        })
        with pytest.raises(KlippbokConfigError, match="4n.1"):
            load_data_config(config_path)

    def test_frame_count_auto(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "video": {"frame_count": "auto"},
        })
        config = load_data_config(config_path)
        assert config.video.frame_count == "auto"


# ─── Empty / Edge Cases ───


class TestEdgeCases:
    """Edge cases and unusual but valid configs."""

    def test_empty_yaml_file(self, tmp_path: Path):
        """An empty YAML file has no dataset path → error."""
        config_path = tmp_path / "klippbok_data.yaml"
        config_path.write_text("")
        with pytest.raises(KlippbokConfigError, match="No dataset paths"):
            load_data_config(config_path)

    def test_empty_tags(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "metadata": {"tags": []},
        })
        config = load_data_config(config_path)
        assert config.metadata.tags == []

    def test_description_field(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "dataset": {"description": "My test dataset"},
            "datasets": [{"path": str(tmp_dataset)}],
        })
        config = load_data_config(config_path)
        assert config.dataset.description == "My test dataset"

    def test_exposure_range_tuple(self, tmp_path: Path, tmp_dataset: Path):
        """exposure_range accepts a two-element list → tuple."""
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "quality": {"exposure_range": [0.2, 0.8]},
        })
        config = load_data_config(config_path)
        assert config.quality.exposure_range == (0.2, 0.8)

    def test_keep_tokens_zero(self, tmp_path: Path, tmp_dataset: Path):
        """keep_tokens=0 is valid — no tokens preserved during shuffle."""
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "controls": {"text": {"keep_tokens": 0}},
        })
        config = load_data_config(config_path)
        assert config.controls.text.keep_tokens == 0

    def test_token_dropout_rate_boundaries(
        self, tmp_path: Path, tmp_dataset: Path
    ):
        """0.0 and 1.0 are both valid token_dropout_rate values."""
        for rate in [0.0, 1.0]:
            config_path = write_config(tmp_path, {
                "datasets": [{"path": str(tmp_dataset)}],
                "controls": {"text": {"token_dropout_rate": rate}},
            })
            config = load_data_config(config_path)
            assert config.controls.text.token_dropout_rate == rate

    def test_resolution_720(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "video": {"resolution": 720},
        })
        config = load_data_config(config_path)
        assert config.video.resolution == 720

    def test_metadata_source(self, tmp_path: Path, tmp_dataset: Path):
        config_path = write_config(tmp_path, {
            "datasets": [{"path": str(tmp_dataset)}],
            "metadata": {"source": "Client footage - Project X"},
        })
        config = load_data_config(config_path)
        assert config.metadata.source == "Client footage - Project X"


# ─── Real Dataset: Sample Character 5-Clip Subset ───


SAMPLE_FIXTURE = Path(__file__).parent / "fixtures" / "sample_character"


class TestSampleCharacterSubset:
    """Integration tests using a real 5-clip character dataset subset.

    These tests verify that the config schema works with a real-world
    dataset layout: .mov files with .txt sidecar captions, anchor word,
    character use_case, first_frame reference source.
    """

    def test_load_from_directory(self):
        """Load the config by pointing at its directory."""
        config = load_data_config(SAMPLE_FIXTURE)

        assert config.dataset.name == "sample_character"
        assert config.dataset.use_case == "character"
        assert config.dataset.description == "Sample character LoRA — 5-clip test subset"

    def test_dataset_source(self):
        """The single dataset source points at the fixture directory."""
        config = load_data_config(SAMPLE_FIXTURE)

        assert len(config.datasets) == 1
        assert Path(config.datasets[0].path) == SAMPLE_FIXTURE.resolve()
        assert config.datasets[0].repeats == 1
        assert config.datasets[0].loss_multiplier == 1.0
        assert config.datasets[0].is_regularization is False

    def test_video_settings(self):
        """Video settings match Wan defaults."""
        config = load_data_config(SAMPLE_FIXTURE)

        assert config.video.fps == 16
        assert config.video.resolution == 480
        assert config.video.frame_count == "auto"

    def test_anchor_word(self):
        """Anchor word is 'Luna' for this character LoRA."""
        config = load_data_config(SAMPLE_FIXTURE)

        assert config.controls.text.anchor_word == "Luna"
        assert config.controls.text.format == "txt"
        assert config.controls.text.required is True

    def test_reference_images(self):
        """Reference source is first_frame for I2V training."""
        config = load_data_config(SAMPLE_FIXTURE)

        assert config.controls.images.reference.source == "first_frame"
        assert config.controls.images.reference.required is False

    def test_all_defaults_populated(self):
        """Every section of the config has defaults, nothing is None unexpectedly."""
        config = load_data_config(SAMPLE_FIXTURE)

        # Quality defaults
        assert config.quality.min_resolution == 720
        assert config.quality.blur_threshold is None
        assert config.quality.motion.min_intensity is None

        # Bucketing defaults
        assert len(config.bucketing.dimensions) == 3
        assert config.bucketing.aspect_ratio_tolerance == 0.1

        # Metadata defaults
        assert config.metadata.tags == []
