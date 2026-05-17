from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest


@pytest.fixture
def example_config_path() -> Path:
    return Path("examples/research_team/project.yaml")


@pytest.fixture
def minimal_project_dir(tmp_path: Path) -> Path:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "root.md").write_text("Answer the user.", encoding="utf-8")
    (tmp_path / "project.yaml").write_text(
        dedent(
            """
            app:
              name: minimal
              root_agent: root
              default_model: gemini-flash-latest

            agents:
              root:
                type: llm
                name: assistant
                instruction_file: prompts/root.md
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return tmp_path

