"""Config loader for Klippbok data configs.

Handles YAML loading, path resolution, backwards compatibility, and
human-readable error formatting. This is the main entry point for
consuming a Klippbok data config.

Usage::

    from klippbok.config import load_data_config

    # From a YAML file
    config = load_data_config("path/to/klippbok_data.yaml")

    # From a directory (looks for klippbok_data.yaml inside)
    config = load_data_config("path/to/dataset_folder")

    # From just a path (creates minimal config with that path as dataset source)
    config = load_data_config("path/to/video_clips")
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from klippbok.config.data_schema import KlippbokDataConfig

# The filename we look for when given a directory
CONFIG_FILENAME = "klippbok_data.yaml"


def load_data_config(path: str | Path) -> KlippbokDataConfig:
    """Load and validate a Klippbok data config.

    This is the main entry point. It accepts three kinds of input:

    1. **A YAML file path**: loads and parses it directly.
    2. **A directory containing klippbok_data.yaml**: discovers and loads it.
    3. **A directory without a config file**: creates a minimal config
       using that directory as the single dataset source.

    All relative paths in the config are resolved relative to the config
    file's location (or the directory itself, if no config file exists).

    Args:
        path: Path to a YAML config file, or a directory.

    Returns:
        A fully validated KlippbokDataConfig with all defaults applied
        and all relative paths resolved.

    Raises:
        FileNotFoundError: If the path doesn't exist.
        KlippbokConfigError: If the config has validation errors.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Path not found: {path}\n"
            f"Check that the path exists and is spelled correctly."
        )

    # Determine the config data and the base directory for path resolution
    if path.is_file():
        config_data = _load_yaml(path)
        base_dir = path.parent
    elif (path / CONFIG_FILENAME).is_file():
        config_data = _load_yaml(path / CONFIG_FILENAME)
        base_dir = path
    else:
        # No config file found — create a minimal config from the directory
        config_data = {
            "datasets": [{"path": str(path)}],
        }
        base_dir = path

    # Backwards compatibility: if the user put a path in dataset.path
    # (singular shorthand) and has no datasets list, wrap it
    config_data = _apply_backwards_compat(config_data)

    # Resolve relative paths against the config file's directory
    config_data = _resolve_paths(config_data, base_dir)

    # Parse and validate through Pydantic
    try:
        config = KlippbokDataConfig.model_validate(config_data)
    except ValidationError as e:
        raise KlippbokConfigError(_format_validation_error(e)) from e

    # Check that dataset paths exist on disk
    _check_dataset_paths(config)

    return config


class KlippbokConfigError(Exception):
    """Raised when a Klippbok data config has validation errors.

    The error message is formatted for humans — it tells you what's wrong
    and how to fix it. This is the only exception type that load_data_config
    raises for config content problems (FileNotFoundError is used for
    missing files).
    """

    pass


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict.

    Returns an empty dict if the file is empty or contains only comments.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _apply_backwards_compat(data: dict) -> dict:
    """Handle the dataset.path → datasets[] shorthand.

    If the user wrote::

        dataset:
          path: ./video_clips
          name: annika

    and there's no 'datasets' key, we auto-wrap the path into a
    single-item datasets list. The 'path' key is removed from
    'dataset' since it's not part of DatasetIdentityConfig.
    """
    dataset_block = data.get("dataset", {})
    if not isinstance(dataset_block, dict):
        return data

    shorthand_path = dataset_block.pop("path", None)

    if shorthand_path is not None and "datasets" not in data:
        data["datasets"] = [{"path": shorthand_path}]

    return data


def _resolve_paths(data: dict, base_dir: Path) -> dict:
    """Resolve all relative paths in the config against base_dir.

    Paths that are already absolute are left as-is. This ensures the
    config works regardless of the current working directory.
    """
    # Resolve dataset source paths
    for ds in data.get("datasets", []):
        if "path" in ds:
            ds["path"] = str(_resolve_one(ds["path"], base_dir))

    # Resolve JSONL file path
    controls = data.get("controls", {})
    text = controls.get("text", {})
    if isinstance(text, dict) and text.get("jsonl_file"):
        text["jsonl_file"] = str(_resolve_one(text["jsonl_file"], base_dir))

    # Resolve reference image folder
    images = controls.get("images", {})
    ref = images.get("reference", {})
    if isinstance(ref, dict) and ref.get("folder"):
        ref["folder"] = str(_resolve_one(ref["folder"], base_dir))

    return data


def _resolve_one(path_str: str, base_dir: Path) -> Path:
    """Resolve a single path string against a base directory.

    Absolute paths are returned as-is. Relative paths are resolved
    against base_dir and fully resolved (symlinks, .., etc.).
    """
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def _check_dataset_paths(config: KlippbokDataConfig) -> None:
    """Verify that all dataset source paths exist on disk.

    This is an existence check only — we don't validate file formats
    or contents here. Format validation happens in Phase 1.
    """
    missing = []
    for ds in config.datasets:
        p = Path(ds.path)
        if not p.exists():
            missing.append(str(p))

    if missing:
        paths_list = "\n  - ".join(missing)
        raise KlippbokConfigError(
            f"Dataset path(s) not found:\n  - {paths_list}\n\n"
            f"Check that these directories exist. Relative paths are resolved "
            f"from the config file's location."
        )


def _format_validation_error(error: ValidationError) -> str:
    """Convert Pydantic's ValidationError into human-readable messages.

    Pydantic's default error output is developer-oriented and can be
    confusing for users who don't know what "value_error" means.
    We reformat each error into: where it is → what's wrong → how to fix it.
    """
    lines = ["Data config validation failed:\n"]

    for err in error.errors():
        # Build the location string (e.g., "video → resolution")
        loc_parts = [str(part) for part in err["loc"]]
        location = " > ".join(loc_parts) if loc_parts else "(root)"

        # The error message from Pydantic (or our custom validators)
        message = err["msg"]

        # Clean up Pydantic's "Value error, " prefix from custom validators
        if message.startswith("Value error, "):
            message = message[len("Value error, "):]

        lines.append(f"  {location}: {message}")

    lines.append(
        "\nSee examples/ in klippbok-kit for valid config files."
    )
    return "\n".join(lines)
