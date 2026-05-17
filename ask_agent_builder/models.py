"""Typed configuration models for the higher-level builder YAML."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentType(str, Enum):
    """Agent types supported by the builder."""

    LLM = "llm"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    LOOP = "loop"


ADK_AGENT_CLASS_BY_TYPE: dict[AgentType, str] = {
    AgentType.LLM: "LlmAgent",
    AgentType.SEQUENTIAL: "SequentialAgent",
    AgentType.PARALLEL: "ParallelAgent",
    AgentType.LOOP: "LoopAgent",
}

AGENT_TYPE_ALIASES: dict[str, AgentType] = {
    "agent": AgentType.LLM,
    "llm": AgentType.LLM,
    "llmagent": AgentType.LLM,
    "LlmAgent": AgentType.LLM,
    "sequential": AgentType.SEQUENTIAL,
    "sequentialagent": AgentType.SEQUENTIAL,
    "SequentialAgent": AgentType.SEQUENTIAL,
    "parallel": AgentType.PARALLEL,
    "parallelagent": AgentType.PARALLEL,
    "ParallelAgent": AgentType.PARALLEL,
    "loop": AgentType.LOOP,
    "loopagent": AgentType.LOOP,
    "LoopAgent": AgentType.LOOP,
}


class StrictModel(BaseModel):
    """Base model that rejects unknown config keys."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class BuilderSettings(BaseSettings):
    """Environment-driven runtime settings for production deployments."""

    model_config = SettingsConfigDict(
        env_prefix="ASK_AGENT_BUILDER_",
        env_file=".env",
        extra="ignore",
    )

    environment: str = "dev"
    strict_import_validation: bool = False
    tool_allowlist: list[str] = Field(default_factory=list)
    generated_dir: Path = Path("Agents")


class AppConfig(StrictModel):
    """Top-level application metadata."""

    name: str
    root_agent: str = "root"
    default_model: str | None = None
    environment: str = "dev"


class RuntimeConfig(StrictModel):
    """Runtime-facing settings that are not emitted into ADK YAML."""

    generated_dir: Path = Path("Agents")
    strict: bool = False


class SecurityConfig(StrictModel):
    """Security controls for production config validation."""

    allow_custom_tools: bool = True
    allowed_tool_import_prefixes: list[str] = Field(default_factory=list)
    allowed_callback_import_prefixes: list[str] = Field(default_factory=list)


class ObservabilityConfig(StrictModel):
    """Operational metadata for deployments using this builder."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    service_name: str | None = None


class CodeRefConfig(StrictModel):
    """Reference to Python code used by ADK for tools, schemas, callbacks, etc."""

    name: str
    args: dict[str, Any] | None = None


class ToolConfig(StrictModel):
    """Tool reference accepted by the builder."""

    name: str
    args: dict[str, Any] | None = None


class SkillConfig(StrictModel):
    """Reusable bundle of instruction text and tools."""

    description: str = ""
    instruction: str | None = None
    instruction_file: Path | None = None
    tools: list[str | ToolConfig] = Field(default_factory=list)
    callbacks: list[CodeRefConfig] = Field(default_factory=list)


class GenerateContentConfig(StrictModel):
    """Subset wrapper for Gemini generation config."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_output_tokens: int | None = None
    response_mime_type: str | None = None
    response_schema: dict[str, Any] | None = None
    response_json_schema: dict[str, Any] | None = Field(
        default=None,
        alias="responseJsonSchema",
    )


class AgentConfig(StrictModel):
    """One agent definition in the builder YAML."""

    type: AgentType = AgentType.LLM
    name: str | None = None
    description: str = ""
    model: str | None = None
    instruction: str | None = None
    instruction_file: Path | None = None
    tools: list[str | ToolConfig] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    sub_agents: list[str] = Field(default_factory=list)
    max_iterations: int | None = None
    output_key: str | None = None
    include_contents: Literal["default", "none"] | None = None
    input_schema: CodeRefConfig | None = None
    output_schema: CodeRefConfig | None = None
    generate_content_config: GenerateContentConfig | dict[str, Any] | None = None
    before_agent_callbacks: list[CodeRefConfig] = Field(default_factory=list)
    after_agent_callbacks: list[CodeRefConfig] = Field(default_factory=list)
    before_model_callbacks: list[CodeRefConfig] = Field(default_factory=list)
    after_model_callbacks: list[CodeRefConfig] = Field(default_factory=list)
    before_tool_callbacks: list[CodeRefConfig] = Field(default_factory=list)
    after_tool_callbacks: list[CodeRefConfig] = Field(default_factory=list)
    disallow_transfer_to_parent: bool | None = None
    disallow_transfer_to_peers: bool | None = None

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: Any) -> AgentType:
        if isinstance(value, AgentType):
            return value
        if value is None:
            return AgentType.LLM
        if isinstance(value, str):
            normalized = value.strip()
            key = normalized.replace("_", "").replace("-", "").lower()
            if normalized in AGENT_TYPE_ALIASES:
                return AGENT_TYPE_ALIASES[normalized]
            if key in AGENT_TYPE_ALIASES:
                return AGENT_TYPE_ALIASES[key]
        raise ValueError(f"unsupported agent type: {value!r}")

    @model_validator(mode="after")
    def validate_type_specific_fields(self) -> AgentConfig:
        if self.type != AgentType.LLM:
            forbidden = {
                "model": self.model,
                "instruction": self.instruction,
                "instruction_file": self.instruction_file,
                "tools": self.tools or None,
                "skills": self.skills or None,
                "output_key": self.output_key,
                "include_contents": self.include_contents,
                "input_schema": self.input_schema,
                "output_schema": self.output_schema,
                "generate_content_config": self.generate_content_config,
            }
            present = [key for key, value in forbidden.items() if value is not None]
            if present:
                joined = ", ".join(sorted(present))
                raise ValueError(f"{self.type.value} agents cannot define: {joined}")
        if self.type == AgentType.LOOP and self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError("loop agents require max_iterations >= 1")
        if self.type != AgentType.LOOP and self.max_iterations is not None:
            raise ValueError("max_iterations is only valid for loop agents")
        return self

    @property
    def adk_agent_class(self) -> str:
        return ADK_AGENT_CLASS_BY_TYPE[self.type]


class ProjectConfig(StrictModel):
    """Root config loaded from project YAML."""

    app: AppConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    skills: dict[str, SkillConfig] = Field(default_factory=dict)
    agents: dict[str, AgentConfig]
