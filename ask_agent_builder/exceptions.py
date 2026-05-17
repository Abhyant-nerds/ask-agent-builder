"""Project-specific exceptions and diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    """Structured validation diagnostic."""

    code: str
    path: str
    message: str
    severity: Severity = "error"
    hint: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)

    def format(self) -> str:
        hint = f" Hint: {self.hint}" if self.hint else ""
        return f"[{self.code}] {self.path}: {self.message}{hint}"


@dataclass(frozen=True)
class ValidationReport:
    """Collection of validation diagnostics."""

    issues: tuple[ValidationIssue, ...]

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": not self.has_errors,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def format(self) -> str:
        return "\n".join(f"- {issue.format()}" for issue in self.issues)


class AgentBuilderError(Exception):
    """Base exception for builder errors."""

    code = "AGENT_BUILDER_ERROR"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        self.message = message
        self.hint = hint
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.hint:
            return f"[{self.code}] {self.message} Hint: {self.hint}"
        return f"[{self.code}] {self.message}"


class ConfigLoadError(AgentBuilderError):
    """Raised when a YAML config cannot be loaded."""

    code = "CONFIG_LOAD_ERROR"


class ConfigSchemaError(ConfigLoadError):
    """Raised when YAML loads but fails schema parsing."""

    code = "CONFIG_SCHEMA_ERROR"


class ConfigValidationError(AgentBuilderError):
    """Raised when project config is invalid."""

    code = "CONFIG_VALIDATION_ERROR"

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(report.format())


class UnsafePathError(ConfigValidationError):
    """Raised when generated paths would be unsafe."""


class ImportResolutionError(ConfigValidationError):
    """Raised when strict import validation fails."""


class BuildError(AgentBuilderError):
    """Raised when ADK config generation fails."""

    code = "BUILD_ERROR"


class AtomicWriteError(BuildError):
    """Raised when atomic output generation fails."""

    code = "ATOMIC_WRITE_ERROR"


class AdkLoadError(AgentBuilderError):
    """Raised when ADK cannot load generated config."""

    code = "ADK_LOAD_ERROR"


class ExternalCommandError(AgentBuilderError):
    """Raised when an external command fails."""

    code = "EXTERNAL_COMMAND_ERROR"
