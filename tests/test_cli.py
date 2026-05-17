from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ask_agent_builder.cli import app


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
