"""YAML loading for project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ask_agent_builder.exceptions import ConfigLoadError, ConfigSchemaError
from ask_agent_builder.models import ProjectConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a mapping."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigLoadError(
            f"config file does not exist: {config_path}",
            hint="Check the path passed to the CLI or Python API.",
        ) from exc
    except IsADirectoryError as exc:
        raise ConfigLoadError(f"config path is a directory, not a file: {config_path}") from exc
    except PermissionError as exc:
        raise ConfigLoadError(
            f"permission denied while reading config: {config_path}",
            hint="Check file permissions or choose a readable config path.",
        ) from exc
    except UnicodeDecodeError as exc:
        raise ConfigLoadError(
            f"config file is not valid UTF-8: {config_path}",
            hint="Save YAML config files using UTF-8 encoding.",
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"invalid YAML in {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigLoadError(f"failed to read config {config_path}: {exc}") from exc

    if raw is None:
        raise ConfigLoadError(
            f"config file is empty: {config_path}",
            hint="Add app and agents sections.",
        )
    if not isinstance(raw, dict):
        raise ConfigLoadError(
            f"config file must contain a YAML mapping: {config_path}",
            hint="The top-level YAML value must be an object with app and agents keys.",
        )
    return raw


def load_project_config(path: str | Path) -> ProjectConfig:
    """Load and parse a project config YAML file."""

    raw = load_yaml(path)
    try:
        return ProjectConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigSchemaError(str(exc)) from exc
