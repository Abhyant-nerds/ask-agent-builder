"""Create ADK agent instances from generated Agents/<name> folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

from ask_agent_builder.exceptions import AdkLoadError, ExternalCommandError
from ask_agent_builder.runtime import load_adk_agent


ROOT_AGENT_FILE = "root_agent.yaml"


@dataclass(frozen=True)
class GeneratedAgentLoadResult:
    """Summary for a generated agent that was loaded into memory."""

    agent_dir: Path
    root_agent_config: Path
    agent: Any
    name: str
    type_name: str
    sub_agent_count: int


def find_root_agent_config(agent_path: str | Path) -> Path:
    """Find root_agent.yaml from Agents/<name> or a direct root_agent.yaml path."""

    path = Path(agent_path)
    if path.is_dir():
        root_agent_config = path / ROOT_AGENT_FILE
    elif path.name == ROOT_AGENT_FILE:
        root_agent_config = path
    else:
        raise AdkLoadError(
            f"generated agent path must be an agent folder or {ROOT_AGENT_FILE}: {path}",
            hint="Use a path like Agents/research_pipeline or Agents/research_pipeline/root_agent.yaml.",
        )

    if not root_agent_config.exists():
        raise AdkLoadError(
            f"generated agent root config does not exist: {root_agent_config}",
            hint="Run `ask-agent-builder build <project.yaml>` first.",
        )
    if not root_agent_config.is_file():
        raise AdkLoadError(f"generated agent root config is not a file: {root_agent_config}")
    return root_agent_config.resolve()


def load_generated_agent(agent_path: str | Path):
    """Create the actual ADK root agent instance from generated YAML artifacts."""

    return load_adk_agent(find_root_agent_config(agent_path))


def smoke_load_generated_agent(agent_path: str | Path) -> GeneratedAgentLoadResult:
    """Create an ADK root agent instance and return a compact summary."""

    root_agent_config = find_root_agent_config(agent_path)
    agent = load_adk_agent(root_agent_config)
    return GeneratedAgentLoadResult(
        agent_dir=root_agent_config.parent,
        root_agent_config=root_agent_config,
        agent=agent,
        name=getattr(agent, "name", "<unknown>"),
        type_name=type(agent).__name__,
        sub_agent_count=len(getattr(agent, "sub_agents", []) or []),
    )


def run_generated_agent(agent_path: str | Path) -> None:
    """Run an already-generated agent folder with the ADK CLI."""

    root_agent_config = find_root_agent_config(agent_path)
    agent_dir = root_agent_config.parent
    try:
        subprocess.run(["adk", "run", str(agent_dir)], check=True)
    except FileNotFoundError as exc:
        raise ExternalCommandError(
            "adk CLI was not found.",
            hint="Install google-adk and ensure `adk` is on PATH.",
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ExternalCommandError(f"adk run failed with exit code {exc.returncode}") from exc


def describe_generated_agent(result: GeneratedAgentLoadResult) -> str:
    """Format a generated-agent load result for humans."""

    return "\n".join(
        [
            f"loaded: {result.name}",
            f"type: {result.type_name}",
            f"sub_agents: {result.sub_agent_count}",
            f"root_agent: {result.root_agent_config}",
        ]
    )

