from __future__ import annotations

from pathlib import Path

import pytest

from ask_agent_builder.exceptions import ConfigLoadError, ConfigSchemaError
from ask_agent_builder.loader import load_project_config, load_yaml


def test_load_project_config_parses_example(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)

    assert project.app.name == "research_team"
    assert project.app.root_agent == "root"
    assert project.agents["root"].name == "research_pipeline"


def test_load_yaml_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoadError, match="does not exist"):
        load_yaml(tmp_path / "missing.yaml")


def test_load_yaml_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="empty"):
        load_yaml(path)


def test_load_yaml_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="YAML mapping"):
        load_yaml(path)


def test_load_project_config_rejects_schema_errors(tmp_path: Path) -> None:
    path = tmp_path / "project.yaml"
    path.write_text("app:\n  name: broken\n", encoding="utf-8")

    with pytest.raises(ConfigSchemaError, match="agents"):
        load_project_config(path)

