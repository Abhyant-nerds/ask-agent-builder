from __future__ import annotations

from pathlib import Path

import pytest

from ask_agent_builder.exceptions import AdkLoadError
from ask_agent_builder.runtime import build_project_from_file, load_adk_agent


def test_build_project_from_file_returns_build_result(example_config_path: Path, tmp_path: Path) -> None:
    result = build_project_from_file(example_config_path, output_dir=tmp_path)

    assert result.root_config_path == tmp_path.resolve() / "research_pipeline" / "root_agent.yaml"
    assert result.root_config_path.exists()


def test_load_adk_agent_wraps_adk_errors(tmp_path: Path) -> None:
    bad_config = tmp_path / "missing-root.yaml"

    with pytest.raises(AdkLoadError):
        load_adk_agent(bad_config)
