from __future__ import annotations

from pathlib import Path

import pytest

from ask_agent_builder.exceptions import ConfigValidationError
from ask_agent_builder.loader import load_project_config
from ask_agent_builder.models import AgentType
from ask_agent_builder.validator import collect_validation_issues, validate_project_config


def _issue_codes(path: str | Path) -> set[str]:
    project_path = Path(path)
    project = load_project_config(project_path)
    return {issue.code for issue in collect_validation_issues(project, project_path.parent).issues}


def test_example_config_validates_without_strict_imports(example_config_path: Path) -> None:
    validate_project_config(load_project_config(example_config_path), example_config_path.parent)


def test_agent_type_aliases_are_normalized(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)

    assert project.agents["root"].type == AgentType.SEQUENTIAL
    assert project.agents["web_researcher"].type == AgentType.LLM


def test_missing_sub_agent_raises_structured_validation_error(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)
    project.agents["root"].sub_agents.append("missing")

    with pytest.raises(ConfigValidationError) as exc_info:
        validate_project_config(project, example_config_path.parent)

    assert "AGENT_SUB_AGENT_MISSING" in str(exc_info.value)
    assert any(issue.code == "AGENT_SUB_AGENT_MISSING" for issue in exc_info.value.report.issues)


def test_rejects_missing_skill(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)
    project.agents["web_researcher"].skills.append("missing_skill")

    report = collect_validation_issues(project, example_config_path.parent)

    assert any(issue.code == "AGENT_SKILL_MISSING" for issue in report.issues)


def test_rejects_missing_instruction_file(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)
    project.agents["critic"].instruction_file = Path("prompts/missing.md")

    report = collect_validation_issues(project, example_config_path.parent)

    assert any(issue.code == "AGENT_INSTRUCTION_FILE_MISSING" for issue in report.issues)


def test_rejects_duplicate_agent_names(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)
    project.agents["critic"].name = project.agents["final_writer"].name

    report = collect_validation_issues(project, example_config_path.parent)

    assert any(issue.code == "AGENT_NAME_DUPLICATE" for issue in report.issues)


def test_rejects_unsafe_agent_key(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)
    project.agents["../escape"] = project.agents.pop("critic")
    project.agents["root"].sub_agents[1] = "../escape"

    report = collect_validation_issues(project, example_config_path.parent)

    assert any(issue.code == "AGENT_KEY_UNSAFE" for issue in report.issues)


def test_rejects_agent_graph_cycle(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)
    project.agents["final_writer"].sub_agents.append("root")

    report = collect_validation_issues(project, example_config_path.parent)

    assert any(issue.code == "AGENT_GRAPH_CYCLE" for issue in report.issues)


def test_rejects_empty_workflow_agent(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)
    project.agents["parallel_research"].sub_agents = []

    report = collect_validation_issues(project, example_config_path.parent)

    assert any(issue.code == "WORKFLOW_SUB_AGENTS_EMPTY" for issue in report.issues)


def test_rejects_parallel_duplicate_output_key(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)
    project.agents["docs_researcher"].output_key = project.agents["web_researcher"].output_key

    report = collect_validation_issues(project, example_config_path.parent)

    assert any(issue.code == "PARALLEL_OUTPUT_KEY_DUPLICATE" for issue in report.issues)


def test_rejects_loop_without_max_iterations_or_exit_tool(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)
    project.agents["root"].type = AgentType.LOOP

    report = collect_validation_issues(project, example_config_path.parent)

    assert any(issue.code == "LOOP_EXIT_MISSING" for issue in report.issues)


def test_rejects_llm_without_model(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)
    project.app.default_model = None
    project.agents["critic"].model = None

    report = collect_validation_issues(project, example_config_path.parent)

    assert any(issue.code == "LLM_MODEL_MISSING" for issue in report.issues)


def test_rejects_llm_without_instruction(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)
    project.agents["critic"].instruction_file = None
    project.agents["critic"].instruction = None

    report = collect_validation_issues(project, example_config_path.parent)

    assert any(issue.code == "LLM_INSTRUCTION_MISSING" for issue in report.issues)


def test_rejects_custom_tools_when_disabled(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)
    project.security.allow_custom_tools = False

    report = collect_validation_issues(project, example_config_path.parent)

    assert any(issue.code == "CUSTOM_TOOL_DISABLED" for issue in report.issues)


def test_rejects_custom_tool_prefix_not_allowlisted(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)
    project.security.allowed_tool_import_prefixes = ["allowed"]

    report = collect_validation_issues(project, example_config_path.parent)

    assert any(issue.code == "CUSTOM_TOOL_PREFIX_DENIED" for issue in report.issues)


def test_strict_imports_reject_missing_custom_tool(example_config_path: Path) -> None:
    project = load_project_config(example_config_path)

    report = collect_validation_issues(
        project,
        example_config_path.parent,
        strict_imports=True,
    )

    assert any(issue.code == "CUSTOM_TOOL_IMPORT_FAILED" for issue in report.issues)

