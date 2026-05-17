from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ask_agent_builder.exceptions import AdkLoadError
from ask_agent_builder.runtime import build_and_load_agent, build_project_from_file, load_adk_agent


def test_build_project_from_file_returns_build_result(example_config_path: Path, tmp_path: Path) -> None:
    result = build_project_from_file(example_config_path, output_dir=tmp_path)

    assert result.root_config_path == tmp_path.resolve() / "research_pipeline" / "root_agent.yaml"
    assert result.root_config_path.exists()


def test_load_adk_agent_wraps_adk_errors(tmp_path: Path) -> None:
    bad_config = tmp_path / "missing-root.yaml"

    with pytest.raises(AdkLoadError):
        load_adk_agent(bad_config)


def test_load_adk_agent_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loaded = SimpleNamespace(name="root")
    config = tmp_path / "root_agent.yaml"
    config.write_text("agent_class: LlmAgent\nname: root\n", encoding="utf-8")

    class FakeConfigAgentUtils:
        @staticmethod
        def from_config(path: str):
            assert path == str(config)
            return loaded

    monkeypatch.setattr(
        "google.adk.agents.config_agent_utils",
        FakeConfigAgentUtils,
    )

    assert load_adk_agent(config) is loaded


def test_build_and_load_agent_builds_then_loads(
    example_config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = SimpleNamespace(name="research_pipeline")

    monkeypatch.setattr("ask_agent_builder.runtime.load_adk_agent", lambda path: loaded)

    assert build_and_load_agent(example_config_path, output_dir=tmp_path) is loaded
    assert (tmp_path / "research_pipeline" / "root_agent.yaml").exists()
