from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ask_agent_builder.agents import GeneratedAgentLoadResult
from ask_agent_builder.cli import app
from ask_agent_builder.exceptions import BuildError


def test_validate_command_success(example_config_path: Path) -> None:
    result = CliRunner().invoke(app, ["validate", str(example_config_path)])

    assert result.exit_code == 0
    assert "valid:" in result.output


def test_validate_command_json_success(example_config_path: Path) -> None:
    result = CliRunner().invoke(app, ["validate", str(example_config_path), "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {"issues": [], "valid": True}


def test_validate_command_strict_imports_fails(example_config_path: Path) -> None:
    result = CliRunner().invoke(app, ["validate", str(example_config_path), "--strict-imports"])

    assert result.exit_code == 1
    assert "CUSTOM_TOOL_IMPORT_FAILED" in result.output


def test_build_command_creates_output(example_config_path: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["build", str(example_config_path), "--out", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "generated:" in result.output
    assert (tmp_path / "research_pipeline" / "root_agent.yaml").exists()


def test_build_command_defaults_to_agents_folder(example_config_path: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = example_config_path.resolve()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["build", str(config_path)])

        assert result.exit_code == 0
        assert Path("Agents/research_pipeline/root_agent.yaml").exists()


def test_inspect_command_outputs_normalized_json(example_config_path: Path) -> None:
    result = CliRunner().invoke(app, ["inspect", str(example_config_path)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["app"]["name"] == "research_team"
    assert payload["agents"]["root"]["type"] == "sequential"


def test_doctor_command_reports_tools() -> None:
    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "python:" in result.output
    assert "adk:" in result.output
    assert "uv:" in result.output


def test_load_agent_command_prints_generated_agent_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    agent_dir = tmp_path / "Agents" / "research_pipeline"
    agent_dir.mkdir(parents=True)
    root_agent = agent_dir / "root_agent.yaml"
    root_agent.write_text("agent_class: SequentialAgent\nname: research_pipeline\n", encoding="utf-8")
    load_result = GeneratedAgentLoadResult(
        agent_dir=agent_dir,
        root_agent_config=root_agent,
        agent=SimpleNamespace(name="research_pipeline"),
        name="research_pipeline",
        type_name="SequentialAgent",
        sub_agent_count=3,
    )

    monkeypatch.setattr("ask_agent_builder.cli.smoke_load_generated_agent", lambda path: load_result)

    result = CliRunner().invoke(app, ["load-agent", str(agent_dir)])

    assert result.exit_code == 0
    assert "loaded: research_pipeline" in result.output
    assert "type: SequentialAgent" in result.output


def test_run_agent_command_invokes_generated_agent_runner(tmp_path: Path, monkeypatch) -> None:
    agent_dir = tmp_path / "Agents" / "research_pipeline"
    agent_dir.mkdir(parents=True)
    calls: list[Path] = []

    monkeypatch.setattr("ask_agent_builder.cli.run_generated_agent", lambda path: calls.append(path))

    result = CliRunner().invoke(app, ["run-agent", str(agent_dir)])

    assert result.exit_code == 0
    assert calls == [agent_dir]


def test_validate_command_reports_missing_config(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["validate", str(tmp_path / "missing.yaml")])

    assert result.exit_code == 1
    assert "CONFIG_LOAD_ERROR" in result.output


def test_build_command_reports_builder_error(example_config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_build(*args, **kwargs):
        raise BuildError("boom")

    monkeypatch.setattr("ask_agent_builder.cli.build_project_from_file", fail_build)

    result = CliRunner().invoke(app, ["build", str(example_config_path)])

    assert result.exit_code == 1
    assert "BUILD_ERROR" in result.output


def test_build_command_debug_prints_traceback(example_config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_build(*args, **kwargs):
        raise RuntimeError("debug boom")

    monkeypatch.setattr("ask_agent_builder.cli.build_project_from_file", fail_build)

    result = CliRunner().invoke(app, ["build", str(example_config_path), "--debug"])

    assert result.exit_code == 1
    assert "RuntimeError: debug boom" in result.output


def test_inspect_command_reports_validation_error(tmp_path: Path) -> None:
    config = tmp_path / "project.yaml"
    config.write_text(
        "app:\n  name: demo\nagents:\n  root:\n    instruction: hi\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["inspect", str(config)])

    assert result.exit_code == 1
    assert "LLM_MODEL_MISSING" in result.output


def test_run_command_invokes_adk_run(example_config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        calls.append(command)
        assert check is True

    monkeypatch.setattr("ask_agent_builder.cli.subprocess.run", fake_run)

    result = CliRunner().invoke(app, ["run", str(example_config_path), "--out", str(tmp_path)])

    assert result.exit_code == 0
    assert calls == [["adk", "run", str((tmp_path / "research_pipeline").resolve())]]


def test_run_command_reports_missing_adk(example_config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], check: bool) -> None:
        raise FileNotFoundError("adk")

    monkeypatch.setattr("ask_agent_builder.cli.subprocess.run", fake_run)

    result = CliRunner().invoke(app, ["run", str(example_config_path), "--out", str(tmp_path)])

    assert result.exit_code == 1
    assert "adk CLI was not found" in result.output


def test_run_command_propagates_adk_exit_code(
    example_config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], check: bool) -> None:
        raise subprocess.CalledProcessError(returncode=7, cmd=command)

    monkeypatch.setattr("ask_agent_builder.cli.subprocess.run", fake_run)

    result = CliRunner().invoke(app, ["run", str(example_config_path), "--out", str(tmp_path)])

    assert result.exit_code == 7


def test_load_agent_command_reports_loader_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["load-agent", str(tmp_path / "missing")])

    assert result.exit_code == 1
    assert "ADK_LOAD_ERROR" in result.output


def test_run_agent_command_reports_runner_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["run-agent", str(tmp_path / "missing")])

    assert result.exit_code == 1
    assert "ADK_LOAD_ERROR" in result.output
