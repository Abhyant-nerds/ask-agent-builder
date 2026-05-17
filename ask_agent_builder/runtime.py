"""Runtime loading helpers for generated ADK configs."""

from __future__ import annotations

from pathlib import Path

from ask_agent_builder.exceptions import AdkLoadError
from ask_agent_builder.generator import BuildResult, build_adk_project
from ask_agent_builder.loader import load_project_config


def build_and_load_agent(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    strict_imports: bool | None = None,
):
    """Build ADK YAML from a project config and load the root ADK agent."""

    project_path = Path(config_path)
    project = load_project_config(project_path)
    build = build_adk_project(
        project,
        output_dir=output_dir,
        base_dir=project_path.parent,
        strict_imports=strict_imports,
    )
    return load_adk_agent(build.root_config_path)


def load_adk_agent(root_config_path: str | Path):
    """Load an ADK agent from an ADK Agent Config YAML file."""

    try:
        from google.adk.agents import config_agent_utils
    except ImportError as exc:
        raise AdkLoadError(
            "google-adk is required to load agents.",
            hint="Install the project dependencies and run inside the project virtual environment.",
        ) from exc

    try:
        return config_agent_utils.from_config(str(root_config_path))
    except Exception as exc:
        raise AdkLoadError(f"ADK failed to load root config {root_config_path}: {exc}") from exc


def build_project_from_file(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    strict_imports: bool | None = None,
) -> BuildResult:
    """Load project YAML and generate ADK YAML."""

    project_path = Path(config_path)
    project = load_project_config(project_path)
    return build_adk_project(
        project,
        output_dir=output_dir,
        base_dir=project_path.parent,
        strict_imports=strict_imports,
    )
