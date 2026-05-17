# ask-agent-builder

A configuration-driven builder for Google ADK agents. It reads one production-grade
project YAML file, validates the agent graph, expands prompts and skills, and
generates ADK Agent Config YAML files that can be loaded by ADK.

## Supported ADK Agent Types

The builder supports the Agent Config types currently suitable for YAML-driven
creation:

- `llm` -> `LlmAgent`
- `sequential` -> `SequentialAgent`
- `parallel` -> `ParallelAgent`
- `loop` -> `LoopAgent`

`LangGraphAgent` and `A2aAgent` are intentionally not supported because ADK
Agent Config does not support them yet.

## Example

```yaml
app:
  name: research_team
  root_agent: root
  default_model: gemini-flash-latest

agents:
  root:
    type: sequential
    name: research_pipeline
    sub_agents:
      - researcher
      - writer

  researcher:
    type: llm
    name: researcher
    instruction: Research the user request.
    output_key: findings
    tools:
      - google_search

  writer:
    type: llm
    name: writer
    instruction: Write the final answer from {findings}.
```

Generated ADK YAML uses ADK's native `agent_class` fields and `config_path`
sub-agent references.

## Commands

Use these commands from the repository root after installing the project into
your virtual environment.

```bash
ask-agent-builder validate examples/research_team/project.yaml
ask-agent-builder validate examples/research_team/project.yaml --json
ask-agent-builder validate examples/research_team/project.yaml --strict-imports
ask-agent-builder build examples/research_team/project.yaml
ask-agent-builder inspect examples/research_team/project.yaml
ask-agent-builder run examples/research_team/project.yaml
ask-agent-builder doctor
```

### `validate`

```bash
ask-agent-builder validate examples/research_team/project.yaml
```

Checks the builder YAML without generating files. Use this while editing agent
configs to catch missing agents, missing prompt files, duplicate names, unsafe
names, graph cycles, invalid workflow structure, and model/instruction issues.

### `validate --json`

```bash
ask-agent-builder validate examples/research_team/project.yaml --json
```

Runs the same validation but prints structured JSON diagnostics. Use this in CI,
editor integrations, API responses, or any place where another tool needs to
read validation results.

### `validate --strict-imports`

```bash
ask-agent-builder validate examples/research_team/project.yaml --strict-imports
```

Runs normal validation and also imports custom tools, callbacks, and schema
references. Use this before production builds or in CI. During early config
drafting, plain `validate` is faster and does not require every custom tool
module to exist yet.

### `build`

```bash
ask-agent-builder build examples/research_team/project.yaml
```

Generates ADK-compatible Agent Config YAML files. Use this when the builder YAML
is ready and you want a runnable ADK project. The command writes through a
temporary directory before replacing the final output folder.

By default, output is written under `Agents`. To use another base folder, pass
`--out <folder>`:

```bash
ask-agent-builder build examples/research_team/project.yaml --out generated
```

The `build` command creates:

```text
Agents/research_pipeline/
  root_agent.yaml
  parallel_research.yaml
  web_researcher.yaml
  docs_researcher.yaml
  critic.yaml
  final_writer.yaml
  .env.example
```

The generated folder is named after the root agent's `name` value. If the root
agent has no explicit `name`, the builder uses the root agent key from
`app.root_agent`.

### `inspect`

```bash
ask-agent-builder inspect examples/research_team/project.yaml
```

Prints the normalized parsed config as JSON. Use this when you want to debug how
the YAML was interpreted after defaults, type normalization, and schema parsing.
It does not generate ADK files.

### `run`

```bash
ask-agent-builder run examples/research_team/project.yaml
```

Builds the ADK config and then runs it with the ADK CLI. Use this for local
manual testing after validation passes. For the example config, this runs the
generated `Agents/research_pipeline` ADK app.

### `doctor`

```bash
ask-agent-builder doctor
```

Checks the local environment and reports the Python executable, ADK CLI, and
`uv` availability. Use this when tests, imports, or ADK execution are using the
wrong interpreter or missing tools.

## Python API

```python
from ask_agent_builder.runtime import build_and_load_agent

root_agent = build_and_load_agent("examples/research_team/project.yaml")
```

Validation and build commands do not import ADK at runtime. ADK is imported
lazily only when loading or running an actual agent.

## Production Safety

The builder validates unsafe agent names before generating files, rejects
sub-agent cycles, can verify custom Python imports with `--strict-imports`, and
writes generated ADK projects through a temporary directory before replacing the
final output folder. `validate --json` returns structured diagnostics for CI or
editor integrations.
