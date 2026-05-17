from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ask_agent_builder.agents import (
    GeneratedAgentLoadResult,
    describe_generated_agent,
    find_root_agent_config,
    load_generated_agent,
    run_generated_agent,
    smoke_load_generated_agent,
)
from ask_agent_builder.exceptions import AdkLoadError, ExternalCommandError


def _make_generated_agent_dir(tmp_path: Path) -> Path:
    agent_dir = tmp_path / "Agents" / "research_pipeline"
    agent_dir.mkdir(parents=True)
    (agent_dir / "root_agent.yaml").write_text(
        "agent_class: SequentialAgent\nname: research_pipeline\n",
        encoding="utf-8",
    )
    return agent_dir


def test_find_root_agent_config_from_generated_agent_folder(tmp_path: Path) -> None:
    agent_dir = _make_generated_agent_dir(tmp_path)

    assert find_root_agent_config(agent_dir) == (agent_dir / "root_agent.yaml").resolve()


def test_find_root_agent_config_from_direct_root_agent_yaml(tmp_path: Path) -> None:
    agent_dir = _make_generated_agent_dir(tmp_path)
    root_agent = agent_dir / "root_agent.yaml"

    assert find_root_agent_config(root_agent) == root_agent.resolve()


def test_find_root_agent_config_rejects_non_root_yaml_path(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    config.write_text("name: bad\n", encoding="utf-8")

    with pytest.raises(AdkLoadError, match="generated agent path"):
        find_root_agent_config(config)


def test_find_root_agent_config_rejects_folder_without_root_agent_yaml(tmp_path: Path) -> None:
    agent_dir = tmp_path / "Agents" / "empty"
    agent_dir.mkdir(parents=True)

    with pytest.raises(AdkLoadError, match="root config does not exist"):
        find_root_agent_config(agent_dir)


def test_load_generated_agent_uses_root_agent_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_dir = _make_generated_agent_dir(tmp_path)
    loaded_agent = SimpleNamespace(name="research_pipeline")
    calls: list[Path] = []

    def fake_load_adk_agent(path: Path):
        calls.append(path)
        return loaded_agent

    monkeypatch.setattr("ask_agent_builder.agents.loader.load_adk_agent", fake_load_adk_agent)

    assert load_generated_agent(agent_dir) is loaded_agent
    assert calls == [(agent_dir / "root_agent.yaml").resolve()]


def test_smoke_load_generated_agent_returns_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_dir = _make_generated_agent_dir(tmp_path)
    loaded_agent = SimpleNamespace(name="research_pipeline", sub_agents=[object(), object(), object()])

    monkeypatch.setattr(
        "ask_agent_builder.agents.loader.load_adk_agent",
        lambda path: loaded_agent,
    )

    result = smoke_load_generated_agent(agent_dir)

    assert result.agent is loaded_agent
    assert result.name == "research_pipeline"
    assert result.type_name == "SimpleNamespace"
    assert result.sub_agent_count == 3


def test_describe_generated_agent_formats_summary(tmp_path: Path) -> None:
    result = GeneratedAgentLoadResult(
        agent_dir=tmp_path / "Agents" / "research_pipeline",
        root_agent_config=tmp_path / "Agents" / "research_pipeline" / "root_agent.yaml",
        agent=object(),
        name="research_pipeline",
        type_name="SequentialAgent",
        sub_agent_count=3,
    )

    output = describe_generated_agent(result)

    assert "loaded: research_pipeline" in output
    assert "type: SequentialAgent" in output
    assert "sub_agents: 3" in output
    assert "root_agent:" in output


def test_run_generated_agent_invokes_adk_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_dir = _make_generated_agent_dir(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        calls.append(command)
        assert check is True

    monkeypatch.setattr("ask_agent_builder.agents.loader.subprocess.run", fake_run)

    run_generated_agent(agent_dir)

    assert calls == [["adk", "run", str(agent_dir.resolve())]]


def test_run_generated_agent_wraps_missing_adk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_dir = _make_generated_agent_dir(tmp_path)

    def fake_run(command: list[str], check: bool) -> None:
        raise FileNotFoundError("adk")

    monkeypatch.setattr("ask_agent_builder.agents.loader.subprocess.run", fake_run)

    with pytest.raises(ExternalCommandError, match="adk CLI"):
        run_generated_agent(agent_dir)

