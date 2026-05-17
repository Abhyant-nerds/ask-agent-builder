from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ask_agent_builder.exceptions import BuildError, ConfigValidationError
from ask_agent_builder.generator import build_adk_project
from ask_agent_builder.loader import load_project_config
from ask_agent_builder.models import AgentType, CodeRefConfig, ToolConfig


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

    web_researcher = yaml.safe_load(
        (result.output_dir / "web_researcher.yaml").read_text(encoding="utf-8")
    )
    assert web_researcher["agent_class"] == "LlmAgent"
    assert web_researcher["model"] == "gemini-flash-latest"
    assert web_researcher["tools"] == [{"name": "google_search"}]
    assert "Use web search" in web_researcher["instruction"]


def test_build_uses_default_agents_output_dir(minimal_project_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(minimal_project_dir)
    project = load_project_config("project.yaml")

    result = build_adk_project(project, base_dir=minimal_project_dir)

    assert result.output_dir == (minimal_project_dir / "Agents" / "assistant").resolve()
    assert result.root_config_path.exists()


def test_build_rejects_unsafe_root_agent_folder_name(minimal_project_dir: Path, tmp_path: Path) -> None:
    project = load_project_config(minimal_project_dir / "project.yaml")
    project.agents["root"].name = "Assistant Team"

    with pytest.raises(ConfigValidationError, match="AGENT_NAME_UNSAFE"):
        build_adk_project(project, output_dir=tmp_path, base_dir=minimal_project_dir)


def test_build_writes_env_example(example_config_path: Path, tmp_path: Path) -> None:
    project = load_project_config(example_config_path)

    result = build_adk_project(project, output_dir=tmp_path, base_dir=example_config_path.parent)

    env_example = result.output_dir / ".env.example"
    assert env_example.exists()
    assert "GOOGLE_API_KEY" in env_example.read_text(encoding="utf-8")


def test_build_preserves_existing_output_when_validation_fails(
    example_config_path: Path,
    tmp_path: Path,
) -> None:
    project = load_project_config(example_config_path)
    result = build_adk_project(project, output_dir=tmp_path, base_dir=example_config_path.parent)
    marker = result.output_dir / "marker.txt"
    marker.write_text("keep", encoding="utf-8")

    project.agents["bad/key"] = project.agents.pop("critic")
    project.agents["root"].sub_agents[1] = "bad/key"

    with pytest.raises(ConfigValidationError):
        build_adk_project(project, output_dir=tmp_path, base_dir=example_config_path.parent)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_build_rejects_output_path_that_is_file(
    example_config_path: Path,
    tmp_path: Path,
) -> None:
    project = load_project_config(example_config_path)
    output_file = tmp_path / "out-file"
    output_file.write_text("not a dir", encoding="utf-8")

    with pytest.raises(BuildError):
        build_adk_project(project, output_dir=output_file, base_dir=example_config_path.parent)


def test_build_emits_loop_agent_max_iterations(example_config_path: Path, tmp_path: Path) -> None:
    project = load_project_config(example_config_path)
    project.agents["root"].type = AgentType.LOOP
    project.agents["root"].max_iterations = 3

    result = build_adk_project(project, output_dir=tmp_path, base_dir=example_config_path.parent)
    root = yaml.safe_load(result.root_config_path.read_text(encoding="utf-8"))

    assert root["agent_class"] == "LoopAgent"
    assert root["max_iterations"] == 3


def test_build_emits_optional_llm_fields(example_config_path: Path, tmp_path: Path) -> None:
    project = load_project_config(example_config_path)
    agent = project.agents["critic"]
    agent.include_contents = "none"
    agent.input_schema = CodeRefConfig(name="schemas.Input", args={"kind": "test"})
    agent.output_schema = CodeRefConfig(name="schemas.Output")
    agent.generate_content_config = {"temperature": 0.1}
    agent.disallow_transfer_to_parent = True
    agent.disallow_transfer_to_peers = False
    agent.before_agent_callbacks = [CodeRefConfig(name="callbacks.before", args={"sink": "stdout"})]
    agent.tools = [
        ToolConfig(name="tools.with_args", args={"index": "docs"}),
        ToolConfig(name="tools.with_args"),
    ]

    result = build_adk_project(project, output_dir=tmp_path, base_dir=example_config_path.parent)
    critic = yaml.safe_load((result.output_dir / "critic.yaml").read_text(encoding="utf-8"))

    assert critic["include_contents"] == "none"
    assert critic["input_schema"]["args"] == [{"name": "kind", "value": "test"}]
    assert critic["generate_content_config"] == {"temperature": 0.1}
    assert critic["disallow_transfer_to_parent"] is True
    assert critic["disallow_transfer_to_peers"] is False
    assert critic["before_agent_callbacks"][0]["args"] == [{"name": "sink", "value": "stdout"}]
    assert critic["tools"] == [{"name": "tools.with_args", "args": {"index": "docs"}}]


def test_build_wraps_prompt_read_failure(
    example_config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = load_project_config(example_config_path)

    original_read_text = Path.read_text

    def patched_read_text(self: Path, encoding: str = "utf-8") -> str:
        if self.name == "critic.md":
            raise OSError("cannot read")
        return original_read_text(self, encoding=encoding)

    monkeypatch.setattr(Path, "read_text", patched_read_text)

    with pytest.raises(BuildError, match="failed to read referenced file"):
        build_adk_project(project, output_dir=tmp_path, base_dir=example_config_path.parent)
