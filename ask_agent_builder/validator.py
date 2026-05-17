"""Semantic validation for project configuration."""

from __future__ import annotations

from collections import Counter
from importlib import import_module
from pathlib import Path
import re

from ask_agent_builder.exceptions import ConfigValidationError, ValidationIssue, ValidationReport
from ask_agent_builder.models import AgentConfig, AgentType, CodeRefConfig, ProjectConfig, ToolConfig


SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

BUILT_IN_TOOL_NAMES = {
    "AgentTool",
    "ExampleTool",
    "LongRunningFunctionTool",
    "McpToolset",
    "enterprise_web_search",
    "exit_loop",
    "get_user_choice",
    "google_maps_grounding",
    "google_search",
    "load_artifacts",
    "load_memory",
    "load_web_page",
    "preload_memory",
    "url_context",
}


def validate_project_config(
    config: ProjectConfig,
    base_dir: str | Path | None = None,
    *,
    strict_imports: bool | None = None,
) -> None:
    """Validate project config beyond Pydantic field-level checks."""

    report = collect_validation_issues(config, base_dir, strict_imports=strict_imports)
    if report.has_errors:
        raise ConfigValidationError(report)


def collect_validation_issues(
    config: ProjectConfig,
    base_dir: str | Path | None = None,
    *,
    strict_imports: bool | None = None,
) -> ValidationReport:
    """Collect project validation diagnostics without raising."""

    issues: list[ValidationIssue] = []
    root_key = config.app.root_agent
    base_path = Path(base_dir or ".")
    should_check_imports = False if strict_imports is None else strict_imports

    _validate_safe_name("app.name", config.app.name, "APP_NAME_UNSAFE", issues)
    _validate_safe_name("app.root_agent", config.app.root_agent, "ROOT_AGENT_KEY_UNSAFE", issues)

    if root_key not in config.agents:
        _issue(
            issues,
            "ROOT_AGENT_MISSING",
            "app.root_agent",
            f"references missing agent key: {root_key}",
            "Add a matching key under agents or change app.root_agent.",
        )

    runtime_names = [
        agent.name or key
        for key, agent in config.agents.items()
    ]
    duplicate_names = sorted(name for name, count in Counter(runtime_names).items() if count > 1)
    if duplicate_names:
        _issue(
            issues,
            "AGENT_NAME_DUPLICATE",
            "agents",
            f"duplicate agent names: {', '.join(duplicate_names)}",
            "Use unique runtime names for every generated ADK agent.",
        )

    for skill_name, skill in config.skills.items():
        _validate_safe_name(f"skills.{skill_name}", skill_name, "SKILL_KEY_UNSAFE", issues)
        if skill.instruction_file is not None:
            skill_instruction_path = base_path / skill.instruction_file
            if not skill_instruction_path.exists():
                _issue(
                    issues,
                    "SKILL_INSTRUCTION_FILE_MISSING",
                    f"skills.{skill_name}.instruction_file",
                    f"does not exist: {skill_instruction_path}",
                    "Create the prompt file or update instruction_file.",
                )

    for key, agent in config.agents.items():
        _validate_agent(key, agent, config, base_path, issues, should_check_imports)

    _validate_graph(config, issues)

    return ValidationReport(tuple(issues))


def _validate_agent(
    key: str,
    agent: AgentConfig,
    config: ProjectConfig,
    base_path: Path,
    issues: list[ValidationIssue],
    strict_imports: bool,
) -> None:
    agent_label = f"agents.{key}"
    _validate_safe_name(agent_label, key, "AGENT_KEY_UNSAFE", issues)
    if agent.name is not None:
        _validate_safe_name(f"{agent_label}.name", agent.name, "AGENT_NAME_UNSAFE", issues)

    for sub_agent in agent.sub_agents:
        if sub_agent not in config.agents:
            _issue(
                issues,
                "AGENT_SUB_AGENT_MISSING",
                f"{agent_label}.sub_agents",
                f"references missing agent: {sub_agent}",
                "Define the referenced agent key under agents.",
            )

    for skill in agent.skills:
        if skill not in config.skills:
            _issue(
                issues,
                "AGENT_SKILL_MISSING",
                f"{agent_label}.skills",
                f"references missing skill: {skill}",
                "Define the skill under skills or remove the reference.",
            )

    if agent.instruction_file is not None:
        instruction_path = base_path / agent.instruction_file
        if not instruction_path.exists():
            _issue(
                issues,
                "AGENT_INSTRUCTION_FILE_MISSING",
                f"{agent_label}.instruction_file",
                f"does not exist: {instruction_path}",
                "Create the prompt file or update instruction_file.",
            )

    if agent.type == AgentType.LLM:
        _validate_llm_agent(key, agent, config, issues)
    else:
        if not agent.sub_agents:
            _issue(
                issues,
                "WORKFLOW_SUB_AGENTS_EMPTY",
                f"{agent_label}.sub_agents",
                f"{agent.type.value} workflow agent has no sub_agents",
                "Workflow agents should define at least one sub-agent.",
            )

    _validate_imports(agent, config, issues, agent_label, strict_imports)


def _validate_llm_agent(
    key: str,
    agent: AgentConfig,
    config: ProjectConfig,
    issues: list[ValidationIssue],
) -> None:
    agent_label = f"agents.{key}"

    if not agent.model and not config.app.default_model:
        _issue(
            issues,
            "LLM_MODEL_MISSING",
            f"{agent_label}.model",
            "llm agent has no model and app.default_model is not set",
            "Set agents.<name>.model or app.default_model.",
        )
    if not agent.instruction and not agent.instruction_file and not agent.skills:
        _issue(
            issues,
            "LLM_INSTRUCTION_MISSING",
            f"{agent_label}.instruction",
            "llm agent has no instruction, instruction_file, or skills",
            "Provide direct instruction text, instruction_file, or reusable skills.",
        )


def _validate_imports(
    agent: AgentConfig,
    config: ProjectConfig,
    issues: list[ValidationIssue],
    agent_label: str,
    strict_imports: bool,
) -> None:
    tools = list(agent.tools)
    for skill_name in agent.skills:
        if skill_name in config.skills:
            tools.extend(config.skills[skill_name].tools)

    if not config.security.allow_custom_tools:
        for tool in tools:
            tool_name = _tool_name(tool)
            if tool_name not in BUILT_IN_TOOL_NAMES:
                _issue(
                    issues,
                    "CUSTOM_TOOL_DISABLED",
                    f"{agent_label}.tools",
                    f"contains custom tool while disabled: {tool_name}",
                    "Enable security.allow_custom_tools or remove the custom tool.",
                )

    allowed_tool_prefixes = config.security.allowed_tool_import_prefixes
    if allowed_tool_prefixes:
        for tool in tools:
            tool_name = _tool_name(tool)
            if tool_name in BUILT_IN_TOOL_NAMES:
                continue
            if not _has_allowed_prefix(tool_name, allowed_tool_prefixes):
                _issue(
                    issues,
                    "CUSTOM_TOOL_PREFIX_DENIED",
                    f"{agent_label}.tools",
                    f"contains import outside allowed prefixes: {tool_name}",
                    "Add the package prefix to security.allowed_tool_import_prefixes.",
                )
            elif strict_imports:
                _validate_importable(
                    tool_name,
                    f"{agent_label}.tools",
                    "CUSTOM_TOOL_IMPORT_FAILED",
                    issues,
                )
    elif strict_imports:
        for tool in tools:
            tool_name = _tool_name(tool)
            if tool_name not in BUILT_IN_TOOL_NAMES:
                _validate_importable(
                    tool_name,
                    f"{agent_label}.tools",
                    "CUSTOM_TOOL_IMPORT_FAILED",
                    issues,
                )

    allowed_callback_prefixes = config.security.allowed_callback_import_prefixes
    if allowed_callback_prefixes:
        callback_names = [
            callback.name
            for callback in [
                *agent.before_agent_callbacks,
                *agent.after_agent_callbacks,
                *agent.before_model_callbacks,
                *agent.after_model_callbacks,
                *agent.before_tool_callbacks,
                *agent.after_tool_callbacks,
            ]
        ]
        for callback_name in callback_names:
            if not _has_allowed_prefix(callback_name, allowed_callback_prefixes):
                _issue(
                    issues,
                    "CALLBACK_PREFIX_DENIED",
                    agent_label,
                    f"contains callback outside allowed prefixes: {callback_name}",
                    "Add the package prefix to security.allowed_callback_import_prefixes.",
                )
            elif strict_imports:
                _validate_importable(
                    callback_name,
                    agent_label,
                    "CALLBACK_IMPORT_FAILED",
                    issues,
                )
    elif strict_imports:
        for callback_name in [
            callback.name
            for callback in [
                *agent.before_agent_callbacks,
                *agent.after_agent_callbacks,
                *agent.before_model_callbacks,
                *agent.after_model_callbacks,
                *agent.before_tool_callbacks,
                *agent.after_tool_callbacks,
            ]
        ]:
            _validate_importable(
                callback_name,
                agent_label,
                "CALLBACK_IMPORT_FAILED",
                issues,
            )

    if strict_imports:
        for code_ref, path, code in [
            (agent.input_schema, f"{agent_label}.input_schema", "INPUT_SCHEMA_IMPORT_FAILED"),
            (agent.output_schema, f"{agent_label}.output_schema", "OUTPUT_SCHEMA_IMPORT_FAILED"),
        ]:
            if code_ref is not None:
                _validate_importable(code_ref.name, path, code, issues)


def _validate_graph(config: ProjectConfig, issues: list[ValidationIssue]) -> None:
    visited: set[str] = set()
    visiting: list[str] = []

    def visit(key: str) -> None:
        if key in visited:
            return
        if key in visiting:
            cycle = " -> ".join([*visiting[visiting.index(key) :], key])
            _issue(
                issues,
                "AGENT_GRAPH_CYCLE",
                "agents",
                f"agent graph contains a cycle: {cycle}",
                "Remove the cycle from sub_agents references.",
            )
            return
        visiting.append(key)
        agent = config.agents.get(key)
        if agent is not None:
            for child in agent.sub_agents:
                if child in config.agents:
                    visit(child)
        visiting.pop()
        visited.add(key)

    for agent_key in config.agents:
        visit(agent_key)

    for key, agent in config.agents.items():
        if agent.type == AgentType.PARALLEL:
            output_keys = [
                config.agents[child].output_key
                for child in agent.sub_agents
                if child in config.agents and config.agents[child].output_key
            ]
            duplicates = sorted(name for name, count in Counter(output_keys).items() if count > 1)
            if duplicates:
                _issue(
                    issues,
                    "PARALLEL_OUTPUT_KEY_DUPLICATE",
                    f"agents.{key}.sub_agents",
                    "parallel sub-agents have duplicate output_key values: " + ", ".join(duplicates),
                    "Give each parallel branch a unique output_key.",
                )

        if agent.type == AgentType.LOOP and agent.max_iterations is None:
            child_tools = [
                _tool_name(tool)
                for child in agent.sub_agents
                if child in config.agents
                for tool in config.agents[child].tools
            ]
            if "exit_loop" not in child_tools:
                _issue(
                    issues,
                    "LOOP_EXIT_MISSING",
                    f"agents.{key}",
                    "loop agent should define max_iterations or include exit_loop in a child tool",
                    "Set max_iterations or add the exit_loop tool to a loop child agent.",
                )


def _tool_name(tool: str | ToolConfig) -> str:
    return tool if isinstance(tool, str) else tool.name


def _has_allowed_prefix(import_name: str, prefixes: list[str]) -> bool:
    return any(import_name == prefix or import_name.startswith(f"{prefix}.") for prefix in prefixes)


def _validate_safe_name(
    path: str,
    value: str,
    code: str,
    issues: list[ValidationIssue],
) -> None:
    if not value or not SAFE_IDENTIFIER_RE.fullmatch(value) or "/" in value or "\\" in value:
        _issue(
            issues,
            code,
            path,
            f"contains unsafe name: {value!r}",
            "Use only letters, numbers, underscores, dots, and hyphens.",
        )


def _validate_importable(
    import_name: str,
    path: str,
    code: str,
    issues: list[ValidationIssue],
) -> None:
    if "." not in import_name:
        return
    module_name, attribute_name = import_name.rsplit(".", 1)
    try:
        module = import_module(module_name)
    except Exception as exc:
        _issue(
            issues,
            code,
            path,
            f"cannot import module {module_name!r} for {import_name!r}: {exc}",
            "Install the package or correct the import path.",
        )
        return
    if not hasattr(module, attribute_name):
        _issue(
            issues,
            code,
            path,
            f"module {module_name!r} has no attribute {attribute_name!r}",
            "Export the referenced function/class or correct the import path.",
        )


def _issue(
    issues: list[ValidationIssue],
    code: str,
    path: str,
    message: str,
    hint: str | None = None,
) -> None:
    issues.append(ValidationIssue(code=code, path=path, message=message, hint=hint))
