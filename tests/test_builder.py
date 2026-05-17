"""Backward-compatible test entrypoints for cached VS Code test selections."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ask_agent_builder.exceptions import ConfigValidationError
from ask_agent_builder.generator import build_adk_project
from ask_agent_builder.loader import load_project_config
from ask_agent_builder.validator import validate_project_config


def test_builds_adk_yaml_from_example(example_config_path: Path, tmp_path: Path) -> None:
    project = load_project_config(example_config_path)

    result = build_adk_project(project, output_dir=tmp_path, base_dir=example_config_path.parent)

    assert result.output_dir == tmp_path / "research_pipeline"
    assert result.app_name == "research_pipeline"
    assert result.root_config_path.name == "root_agent.yaml"

    root = yaml.safe_load(result.root_config_path.read_text(encoding="utf-8"))
    assert root["agent_class"] == "SequentialAgent"
    assert root["sub_agents"] == [
        {"config_path": "parallel_research.yaml"},
        {"config_path": "critic.yaml"},
        {"config_path": "final_writer.yaml"},
    ]


def test_rejects_missing_sub_agent(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)
    project.agents["root"].sub_agents.append("missing")

    with pytest.raises(ConfigValidationError, match="AGENT_SUB_AGENT_MISSING"):
        validate_project_config(project, example_config_path.parent)

