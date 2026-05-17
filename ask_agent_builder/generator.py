"""Generate ADK Agent Config YAML from builder project YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import yaml

from ask_agent_builder.exceptions import AtomicWriteError, BuildError
from ask_agent_builder.models import (
    AgentConfig,
    AgentType,
    CodeRefConfig,
    ProjectConfig,
    SkillConfig,
    ToolConfig,
)
from ask_agent_builder.validator import validate_project_config


@dataclass(frozen=True)
class BuildResult:
    """Result of generating an ADK config project."""

    app_name: str
    output_dir: Path
    root_config_path: Path
    generated_files: tuple[Path, ...]


def build_adk_project(
    config: ProjectConfig,
    output_dir: str | Path | None = None,
    base_dir: str | Path | None = None,
    *,
    strict_imports: bool | None = None,
) -> BuildResult:
    """Validate and generate ADK-compatible YAML config files."""

    source_base = Path(base_dir or ".")
    validate_project_config(config, source_base, strict_imports=strict_imports)

    target_root = Path(output_dir or config.runtime.generated_dir).resolve()
    app_dir = (target_root / _root_agent_directory_name(config)).resolve()
    try:
        target_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BuildError(f"failed to create output directory {target_root}: {exc}") from exc

    filename_map = _agent_filename_map(config)
    _assert_safe_generated_paths(app_dir, filename_map)

    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{app_dir.name}.",
            suffix=".tmp",
            dir=target_root,
        )
    )
    backup_dir: Path | None = None
    try:
        generated_files = _write_project_files(tmp_dir, config, source_base, filename_map)
        if app_dir.exists():
            if not app_dir.is_dir():
                raise AtomicWriteError(f"output path exists and is not a directory: {app_dir}")
            backup_dir = target_root / f".{app_dir.name}.backup"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            app_dir.replace(backup_dir)
        tmp_dir.replace(app_dir)
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if backup_dir is not None and backup_dir.exists() and not app_dir.exists():
            backup_dir.replace(app_dir)
        if isinstance(exc, BuildError):
            raise
        raise AtomicWriteError(f"failed to generate ADK config project: {exc}") from exc

    final_files = tuple(app_dir / file_path.relative_to(tmp_dir) for file_path in generated_files)

    return BuildResult(
        app_name=_root_agent_runtime_name(config),
        output_dir=app_dir,
        root_config_path=app_dir / filename_map[config.app.root_agent],
        generated_files=final_files,
    )


def _write_project_files(
    app_dir: Path,
    config: ProjectConfig,
    source_base: Path,
    filename_map: dict[str, str],
) -> list[Path]:
    app_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []
    for key, agent in config.agents.items():
        file_path = app_dir / filename_map[key]
        file_path.write_text(
            _dump_yaml(_build_adk_agent_config(key, agent, config, source_base, filename_map)),
            encoding="utf-8",
        )
        generated_files.append(file_path)

    env_example = app_dir / ".env.example"
    env_example.write_text(
        "\n".join(
            [
                "GOOGLE_GENAI_USE_VERTEXAI=0",
                "GOOGLE_API_KEY=<your-google-gemini-api-key>",
                "",
                "# For Vertex AI instead:",
                "# GOOGLE_GENAI_USE_VERTEXAI=1",
                "# GOOGLE_CLOUD_PROJECT=<your-gcp-project>",
                "# GOOGLE_CLOUD_LOCATION=us-central1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    generated_files.append(env_example)
    return generated_files


def _build_adk_agent_config(
    key: str,
    agent: AgentConfig,
    project: ProjectConfig,
    source_base: Path,
    filename_map: dict[str, str],
) -> dict[str, Any]:
    adk_config: dict[str, Any] = {
        "agent_class": agent.adk_agent_class,
        "name": agent.name or key,
    }

    if agent.description:
        adk_config["description"] = agent.description

    if agent.sub_agents:
        adk_config["sub_agents"] = [
            {"config_path": filename_map[sub_agent]}
            for sub_agent in agent.sub_agents
        ]

    _add_callbacks(adk_config, agent)

    if agent.type == AgentType.LLM:
        _add_llm_fields(adk_config, agent, project, source_base)

    if agent.type == AgentType.LOOP and agent.max_iterations is not None:
        adk_config["max_iterations"] = agent.max_iterations

    return adk_config


def _add_llm_fields(
    adk_config: dict[str, Any],
    agent: AgentConfig,
    project: ProjectConfig,
    source_base: Path,
) -> None:
    model = agent.model or project.app.default_model
    if model:
        adk_config["model"] = model

    instruction_parts: list[str] = []
    for skill_name in agent.skills:
        skill = project.skills[skill_name]
        skill_instruction = _read_skill_instruction(skill, source_base)
        if skill_instruction:
            instruction_parts.append(skill_instruction)

    direct_instruction = _read_instruction(agent, source_base)
    if direct_instruction:
        instruction_parts.append(direct_instruction)
    adk_config["instruction"] = "\n\n".join(instruction_parts)

    tools = _merged_tools(agent, project)
    if tools:
        adk_config["tools"] = [_tool_to_adk(tool) for tool in tools]

    optional_values = {
        "output_key": agent.output_key,
        "include_contents": agent.include_contents,
        "input_schema": _code_ref_to_adk(agent.input_schema),
        "output_schema": _code_ref_to_adk(agent.output_schema),
        "generate_content_config": _model_to_dict(agent.generate_content_config),
        "disallow_transfer_to_parent": agent.disallow_transfer_to_parent,
        "disallow_transfer_to_peers": agent.disallow_transfer_to_peers,
    }
    for key, value in optional_values.items():
        if value is not None:
            adk_config[key] = value


def _add_callbacks(adk_config: dict[str, Any], agent: AgentConfig) -> None:
    callback_fields = {
        "before_agent_callbacks": agent.before_agent_callbacks,
        "after_agent_callbacks": agent.after_agent_callbacks,
        "before_model_callbacks": agent.before_model_callbacks,
        "after_model_callbacks": agent.after_model_callbacks,
        "before_tool_callbacks": agent.before_tool_callbacks,
        "after_tool_callbacks": agent.after_tool_callbacks,
    }
    for key, callbacks in callback_fields.items():
        if callbacks:
            adk_config[key] = [_code_ref_to_adk(callback) for callback in callbacks]


def _read_instruction(agent: AgentConfig, source_base: Path) -> str:
    if agent.instruction_file is None:
        return agent.instruction or ""
    return _read_text(source_base / agent.instruction_file)


def _read_skill_instruction(skill: SkillConfig, source_base: Path) -> str:
    if skill.instruction_file is not None:
        return _read_text(source_base / skill.instruction_file)
    return skill.instruction or ""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise BuildError(f"failed to read referenced file {path}: {exc}") from exc


def _merged_tools(agent: AgentConfig, project: ProjectConfig) -> list[str | ToolConfig]:
    tools: list[str | ToolConfig] = []
    for skill_name in agent.skills:
        tools.extend(project.skills[skill_name].tools)
    tools.extend(agent.tools)
    return _dedupe_tools(tools)


def _dedupe_tools(tools: list[str | ToolConfig]) -> list[str | ToolConfig]:
    seen: set[str] = set()
    deduped: list[str | ToolConfig] = []
    for tool in tools:
        name = tool if isinstance(tool, str) else tool.name
        if name in seen:
            continue
        seen.add(name)
        deduped.append(tool)
    return deduped


def _tool_to_adk(tool: str | ToolConfig) -> dict[str, Any]:
    if isinstance(tool, str):
        return {"name": tool}
    value: dict[str, Any] = {"name": tool.name}
    if tool.args is not None:
        value["args"] = tool.args
    return value


def _code_ref_to_adk(code_ref: CodeRefConfig | None) -> dict[str, Any] | None:
    if code_ref is None:
        return None
    value: dict[str, Any] = {"name": code_ref.name}
    if code_ref.args is not None:
        value["args"] = [
            {"name": arg_name, "value": arg_value}
            for arg_name, arg_value in code_ref.args.items()
        ]
    return value


def _model_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=True)
    return value


def _agent_filename_map(config: ProjectConfig) -> dict[str, str]:
    filenames: dict[str, str] = {}
    for agent_key in config.agents:
        filenames[agent_key] = "root_agent.yaml" if agent_key == config.app.root_agent else f"{agent_key}.yaml"
    return filenames


def _root_agent_runtime_name(config: ProjectConfig) -> str:
    root_agent = config.agents[config.app.root_agent]
    return root_agent.name or config.app.root_agent


def _root_agent_directory_name(config: ProjectConfig) -> str:
    root_name = _root_agent_runtime_name(config)
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", root_name).strip("._-")
    return sanitized or config.app.root_agent


def _assert_safe_generated_paths(app_dir: Path, filename_map: dict[str, str]) -> None:
    for agent_key, filename in filename_map.items():
        if "/" in filename or "\\" in filename or filename in {"", ".", ".."}:
            raise BuildError(f"unsafe generated filename for agent {agent_key!r}: {filename!r}")
        file_path = (app_dir / filename).resolve()
        if file_path.parent != app_dir:
            raise BuildError(f"generated path escapes output directory: {file_path}")


def _dump_yaml(data: dict[str, Any]) -> str:
    return (
        "# yaml-language-server: $schema=https://raw.githubusercontent.com/google/adk-python/refs/heads/main/src/google/adk/agents/config_schemas/AgentConfig.json\n"
        + yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
    )
