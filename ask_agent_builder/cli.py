"""Command line interface for ask-agent-builder."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Annotated

import typer

from ask_agent_builder.agents import (
    describe_generated_agent,
    run_generated_agent,
    smoke_load_generated_agent,
)
from ask_agent_builder.exceptions import AgentBuilderError
from ask_agent_builder.loader import load_project_config
from ask_agent_builder.runtime import build_project_from_file
from ask_agent_builder.validator import collect_validation_issues, validate_project_config

app = typer.Typer(help="Build Google ADK agents from production-grade YAML configs.")
DEFAULT_OUTPUT_DIR = Path("Agents")


def _handle_error(exc: Exception, *, debug: bool) -> None:
    if debug:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    elif isinstance(exc, AgentBuilderError):
        typer.echo(str(exc), err=True)
    else:
        typer.echo(f"[UNEXPECTED_ERROR] {exc}", err=True)
    raise typer.Exit(1) from exc


@app.command()
def validate(
    config: Annotated[Path, typer.Argument(help="Path to builder project YAML.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print structured validation JSON.")] = False,
    strict_imports: Annotated[
        bool,
        typer.Option("--strict-imports", help="Verify custom import paths can be imported."),
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Show tracebacks for failures.")] = False,
) -> None:
    """Validate a builder project config."""

    try:
        project = load_project_config(config)
        report = collect_validation_issues(project, config.parent, strict_imports=strict_imports)
        if json_output:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        elif report.has_errors:
            typer.echo(report.format(), err=True)
        else:
            typer.echo(f"valid: {config}")
        if report.has_errors:
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_error(exc, debug=debug)


@app.command()
def build(
    config: Annotated[Path, typer.Argument(help="Path to builder project YAML.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Generated output dir.")] = DEFAULT_OUTPUT_DIR,
    strict_imports: Annotated[
        bool,
        typer.Option("--strict-imports", help="Verify custom import paths can be imported."),
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Show tracebacks for failures.")] = False,
) -> None:
    """Generate ADK Agent Config YAML files."""

    try:
        result = build_project_from_file(config, output_dir=out, strict_imports=strict_imports)
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_error(exc, debug=debug)

    typer.echo(f"generated: {result.output_dir}")
    typer.echo(f"root: {result.root_config_path}")
    for file_path in result.generated_files:
        typer.echo(f"- {file_path}")


@app.command()
def inspect(
    config: Annotated[Path, typer.Argument(help="Path to builder project YAML.")],
    debug: Annotated[bool, typer.Option("--debug", help="Show tracebacks for failures.")] = False,
) -> None:
    """Print normalized project config as JSON."""

    try:
        project = load_project_config(config)
        validate_project_config(project, config.parent)
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_error(exc, debug=debug)

    typer.echo(json.dumps(project.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def run(
    config: Annotated[Path, typer.Argument(help="Path to builder project YAML.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Generated output dir.")] = DEFAULT_OUTPUT_DIR,
    strict_imports: Annotated[
        bool,
        typer.Option("--strict-imports", help="Verify custom import paths can be imported."),
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Show tracebacks for failures.")] = False,
) -> None:
    """Build the config and run it with the ADK CLI."""

    try:
        result = build_project_from_file(config, output_dir=out, strict_imports=strict_imports)
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_error(exc, debug=debug)

    typer.echo(f"running ADK app: {result.output_dir}")
    try:
        subprocess.run(["adk", "run", str(result.output_dir)], check=True)
    except FileNotFoundError as exc:
        typer.echo("adk CLI was not found. Install google-adk and ensure `adk` is on PATH.", err=True)
        raise typer.Exit(1) from exc
    except subprocess.CalledProcessError as exc:
        raise typer.Exit(exc.returncode) from exc


@app.command("load-agent")
def load_agent(
    agent_path: Annotated[
        Path,
        typer.Argument(help="Generated agent folder or root_agent.yaml path."),
    ],
    debug: Annotated[bool, typer.Option("--debug", help="Show tracebacks for failures.")] = False,
) -> None:
    """Create an actual ADK agent instance from Agents/<name>/root_agent.yaml."""

    try:
        result = smoke_load_generated_agent(agent_path)
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_error(exc, debug=debug)

    typer.echo(describe_generated_agent(result))


@app.command("run-agent")
def run_agent(
    agent_path: Annotated[
        Path,
        typer.Argument(help="Generated agent folder or root_agent.yaml path."),
    ],
    debug: Annotated[bool, typer.Option("--debug", help="Show tracebacks for failures.")] = False,
) -> None:
    """Run an already-generated agent folder with the ADK CLI."""

    try:
        run_generated_agent(agent_path)
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_error(exc, debug=debug)


@app.command()
def doctor() -> None:
    """Check local development/runtime prerequisites."""

    checks = {
        "python": sys.executable,
        "adk": shutil.which("adk") or "missing",
        "uv": shutil.which("uv") or "missing",
    }
    for name, value in checks.items():
        status = "ok" if value != "missing" else "missing"
        typer.echo(f"{name}: {status} ({value})")


if __name__ == "__main__":
    app()
