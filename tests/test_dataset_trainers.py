"""Tests for klippbok.dataset.trainers — trainer config generators.

Tests cover: registry, musubi TOML generation, aitoolkit YAML generation,
resolution mapping, frame counts, forward-slash paths, dry-run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from klippbok.config.data_schema import KlippbokDataConfig, VideoConfig
from klippbok.dataset.errors import OrganizeError
from klippbok.dataset.models import OrganizeLayout, OrganizedSample
from klippbok.dataset.trainers import (
    TRAINER_REGISTRY,
    _resolution_to_wh,
    _to_forward_slash,
    _unique_frame_counts,
    _video_directory,
    generate_trainer_config,
    get_available_trainers,
)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_musubi_registered(self):
        assert "musubi" in TRAINER_REGISTRY

    def test_aitoolkit_registered(self):
        assert "aitoolkit" in TRAINER_REGISTRY

    def test_get_available_trainers(self):
        available = get_available_trainers()
        assert "musubi" in available
        assert "aitoolkit" in available
        # Sorted alphabetically
        assert available == sorted(available)

    def test_unknown_trainer_raises(self, tmp_path: Path):
        with pytest.raises(OrganizeError, match="Unknown trainer"):
            generate_trainer_config(
                trainer_name="nonexistent",
                samples=[],
                output_dir=tmp_path,
                config=KlippbokDataConfig(datasets=[{"path": "."}]),
                layout=OrganizeLayout.FLAT,
            )


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_forward_slash_windows(self):
        assert _to_forward_slash(Path("C:\\Users\\minta\\data")) == "C:/Users/minta/data"

    def test_forward_slash_unix(self):
        assert _to_forward_slash(Path("/home/user/data")) == "/home/user/data"

    def test_resolution_480(self):
        assert _resolution_to_wh(480) == [854, 480]

    def test_resolution_720(self):
        assert _resolution_to_wh(720) == [1280, 720]

    def test_resolution_1080(self):
        assert _resolution_to_wh(1080) == [1920, 1080]

    def test_resolution_custom(self):
        """Unknown resolution uses 16:9 calculation."""
        wh = _resolution_to_wh(360)
        assert wh[1] == 360
        assert wh[0] == int(360 * 16 / 9)

    def test_unique_frame_counts(self):
        samples = [
            OrganizedSample(stem="a", frame_count=17),
            OrganizedSample(stem="b", frame_count=81),
            OrganizedSample(stem="c", frame_count=17),
            OrganizedSample(stem="d", frame_count=None),
        ]
        counts = _unique_frame_counts(samples)
        assert counts == [17, 81]

    def test_unique_frame_counts_empty(self):
        assert _unique_frame_counts([]) == []

    def test_video_directory_flat(self, tmp_path: Path):
        result = _video_directory(tmp_path, OrganizeLayout.FLAT)
        assert "\\" not in result
        assert result == _to_forward_slash(tmp_path)

    def test_video_directory_klippbok(self, tmp_path: Path):
        result = _video_directory(tmp_path, OrganizeLayout.DIMLJUS)
        assert result.endswith("training/targets")


# ---------------------------------------------------------------------------
# musubi-tuner config tests
# ---------------------------------------------------------------------------

class TestMusubiConfig:
    def _make_samples(self) -> list[OrganizedSample]:
        return [
            OrganizedSample(stem="a", frame_count=17, width=1280, height=720,
                            target_dest=Path("/out/a.mp4")),
            OrganizedSample(stem="b", frame_count=81, width=1280, height=720,
                            target_dest=Path("/out/b.mp4")),
            OrganizedSample(stem="c", frame_count=17, width=1280, height=720,
                            target_dest=Path("/out/c.mp4")),
        ]

    def _default_config(self) -> KlippbokDataConfig:
        return KlippbokDataConfig(
            datasets=[{"path": "."}],
            video=VideoConfig(resolution=720),
        )

    def test_generates_toml_file(self, tmp_path: Path):
        path = generate_trainer_config(
            "musubi", self._make_samples(), tmp_path,
            self._default_config(), OrganizeLayout.FLAT,
        )
        assert path.name == "musubi_dataset.toml"
        assert path.exists()

    def test_toml_content_valid(self, tmp_path: Path):
        path = generate_trainer_config(
            "musubi", self._make_samples(), tmp_path,
            self._default_config(), OrganizeLayout.FLAT,
        )
        content = path.read_text(encoding="utf-8")
        assert "[[datasets]]" in content
        assert "resolution = [1280, 720]" in content
        assert 'caption_extension = ".txt"' in content
        assert "enable_bucket = true" in content
        assert "bucket_no_upscale = true" in content

    def test_toml_has_video_directory(self, tmp_path: Path):
        path = generate_trainer_config(
            "musubi", self._make_samples(), tmp_path,
            self._default_config(), OrganizeLayout.FLAT,
        )
        content = path.read_text(encoding="utf-8")
        assert "video_directory" in content
        # Forward slashes
        assert "\\\\" not in content

    def test_toml_has_cache_directory(self, tmp_path: Path):
        path = generate_trainer_config(
            "musubi", self._make_samples(), tmp_path,
            self._default_config(), OrganizeLayout.FLAT,
        )
        content = path.read_text(encoding="utf-8")
        assert "cache_directory" in content

    def test_toml_frame_counts_deduplicated(self, tmp_path: Path):
        path = generate_trainer_config(
            "musubi", self._make_samples(), tmp_path,
            self._default_config(), OrganizeLayout.FLAT,
        )
        content = path.read_text(encoding="utf-8")
        # 17 and 81 but 17 only once
        assert "target_frames = [17, 81]" in content

    def test_toml_num_repeats(self, tmp_path: Path):
        path = generate_trainer_config(
            "musubi", self._make_samples(), tmp_path,
            self._default_config(), OrganizeLayout.FLAT,
        )
        content = path.read_text(encoding="utf-8")
        assert "num_repeats = 1" in content

    def test_toml_custom_repeats(self, tmp_path: Path):
        config = KlippbokDataConfig(
            datasets=[{"path": ".", "repeats": 3}],
            video=VideoConfig(resolution=720),
        )
        path = generate_trainer_config(
            "musubi", self._make_samples(), tmp_path,
            config, OrganizeLayout.FLAT,
        )
        content = path.read_text(encoding="utf-8")
        assert "num_repeats = 3" in content

    def test_toml_480_resolution(self, tmp_path: Path):
        config = KlippbokDataConfig(
            datasets=[{"path": "."}],
            video=VideoConfig(resolution=480),
        )
        path = generate_trainer_config(
            "musubi", self._make_samples(), tmp_path,
            config, OrganizeLayout.FLAT,
        )
        content = path.read_text(encoding="utf-8")
        assert "resolution = [854, 480]" in content

    def test_toml_has_header_comment(self, tmp_path: Path):
        path = generate_trainer_config(
            "musubi", self._make_samples(), tmp_path,
            self._default_config(), OrganizeLayout.FLAT,
        )
        content = path.read_text(encoding="utf-8")
        assert content.startswith("# musubi-tuner dataset config")
        assert "Generated by" in content

    def test_toml_klippbok_layout_path(self, tmp_path: Path):
        """Klippbok layout points video_directory at training/targets/."""
        path = generate_trainer_config(
            "musubi", self._make_samples(), tmp_path,
            self._default_config(), OrganizeLayout.DIMLJUS,
        )
        content = path.read_text(encoding="utf-8")
        assert "training/targets" in content

    def test_dry_run_no_file(self, tmp_path: Path):
        path = generate_trainer_config(
            "musubi", self._make_samples(), tmp_path,
            self._default_config(), OrganizeLayout.FLAT,
            dry_run=True,
        )
        assert path.name == "musubi_dataset.toml"
        assert not path.exists()


# ---------------------------------------------------------------------------
# ai-toolkit config tests
# ---------------------------------------------------------------------------

class TestAitoolkitConfig:
    def _make_samples(self) -> list[OrganizedSample]:
        return [
            OrganizedSample(stem="a", frame_count=17, width=1280, height=720,
                            target_dest=Path("/out/a.mp4")),
        ]

    def _default_config(self) -> KlippbokDataConfig:
        return KlippbokDataConfig(
            datasets=[{"path": "."}],
            video=VideoConfig(resolution=720),
        )

    def test_generates_yaml_file(self, tmp_path: Path):
        path = generate_trainer_config(
            "aitoolkit", self._make_samples(), tmp_path,
            self._default_config(), OrganizeLayout.FLAT,
        )
        assert path.name == "aitoolkit_config.yaml"
        assert path.exists()

    def test_yaml_content(self, tmp_path: Path):
        path = generate_trainer_config(
            "aitoolkit", self._make_samples(), tmp_path,
            self._default_config(), OrganizeLayout.FLAT,
        )
        content = path.read_text(encoding="utf-8")
        assert "folder_path" in content
        assert 'caption_ext: "txt"' in content
        assert "resolution:" in content
        assert "1280" in content
        assert "720" in content

    def test_yaml_has_network(self, tmp_path: Path):
        path = generate_trainer_config(
            "aitoolkit", self._make_samples(), tmp_path,
            self._default_config(), OrganizeLayout.FLAT,
        )
        content = path.read_text(encoding="utf-8")
        assert "network:" in content
        assert "lora" in content

    def test_yaml_has_header_comment(self, tmp_path: Path):
        path = generate_trainer_config(
            "aitoolkit", self._make_samples(), tmp_path,
            self._default_config(), OrganizeLayout.FLAT,
        )
        content = path.read_text(encoding="utf-8")
        assert content.startswith("# ai-toolkit training config")

    def test_yaml_forward_slashes(self, tmp_path: Path):
        path = generate_trainer_config(
            "aitoolkit", self._make_samples(), tmp_path,
            self._default_config(), OrganizeLayout.FLAT,
        )
        content = path.read_text(encoding="utf-8")
        # No backslashes in paths
        assert "\\\\" not in content

    def test_dry_run_no_file(self, tmp_path: Path):
        path = generate_trainer_config(
            "aitoolkit", self._make_samples(), tmp_path,
            self._default_config(), OrganizeLayout.FLAT,
            dry_run=True,
        )
        assert not path.exists()
