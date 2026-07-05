# OpenCode Training Data Exporter

A utility to export agent chat sessions, tool trajectories, and model outputs
from the OpenCode local database into a training-ready JSONL format.

## Features
- **Full agent trajectories**: Each turn captures the assistant's steps in the
  order they happened — reasoning, tool calls, tool results, and text —
  interleaved so the trajectory can be reconstructed exactly.
- **Tool calls *and* results**: Both the tool input (the call) and the tool
  output (the result), including error results, are exported.
- **Idempotent by default**: Re-running rebuilds the output atomically, so the
  same database always yields the same file with no duplicate rows. An optional
  `--append` mode adds only new, deduplicated rows to an existing file.
- **Clean data**: Reads OpenCode's materialized `message` and `part` tables (the
  final consolidated state), not the streaming event log, so text and tool calls
  are never duplicated by streaming updates. Turns with no usable assistant
  content are dropped.
- **Metadata**: Each row records the `agent` and `model` that produced it.
- **Configurable**: Paths and options via CLI flags or environment variables.

## Installation
The exporter is a standalone Python 3 script with no external dependencies
beyond the standard library.

```bash
git clone https://github.com/imrija95/opencode-training-exporter
cd opencode-training-exporter
```

## Usage
Run the script directly with Python:

```bash
python3 exporter.py
```

By default this rebuilds the output file from scratch (idempotent). To instead
append only new, deduplicated rows to an existing output:

```bash
python3 exporter.py --append
```

### Options and configuration
CLI flags take precedence over environment variables, which take precedence over
the defaults.

| CLI flag | Env variable | Description | Default |
|----------|--------------|-------------|---------|
| `--db` | `OPENCODE_DB_PATH` | Path to `opencode.db` | `~/.local/share/opencode/opencode.db` |
| `--output` | `OPENCODE_OUTPUT_PATH` | Output JSONL path | `~/opencode_training_data.jsonl` |
| `--prompts-dir` | `OPENCODE_PROMPTS_DIR` | Directory of `{agent}.md` system-prompt files (see below) | unset |
| `--append` | — | Append new deduplicated rows instead of rebuilding | rebuild |

Example:
```bash
OPENCODE_OUTPUT_PATH="/path/to/my/dataset.jsonl" python3 exporter.py
```

The database is opened **read-only and immutable**, so the exporter can run
safely while OpenCode is in use; it never locks or writes the database.

## Data Format
The exporter produces a `.jsonl` file where each line is a JSON object
representing one turn (one user prompt and the assistant trajectory answering
it):

```json
{
  "session_id": "ses_...",
  "agent": "coder",
  "model": "provider/model-id",
  "system": "",
  "user": "The user's prompt",
  "assistant": "The concatenated final assistant text",
  "trajectory": [
    {"kind": "reasoning", "text": "..."},
    {"kind": "tool_call", "tool": "bash", "input": {"command": "lsusb"}},
    {"kind": "tool_result", "tool": "bash", "status": "completed", "output": "..."},
    {"kind": "text", "text": "..."}
  ]
}
```

Field notes:
- `trajectory` is the ordered source of truth for the assistant turn. Step kinds
  are `reasoning`, `text`, `tool_call`, and `tool_result`. A `tool_result` carries
  a `status` (`completed` or `error`) and the tool `output`.
- `assistant` is a convenience field: the assistant's final text parts
  concatenated. It may be empty for tool-only turns; such turns are still kept
  because their trajectory contains usable content.
- Structural markers (`step-start`, `step-finish`, `compaction`) are omitted.

### The `system` field
The system/agent prompt is **not stored in the OpenCode database** — OpenCode
injects it at request time. As a result `system` is empty by default. As a
best-effort convenience, if you point `--prompts-dir` (or `OPENCODE_PROMPTS_DIR`)
at a directory of `{agent}.md` files, the exporter attaches the matching agent's
prompt to each row. This only works for agents whose prompt files exist on disk
(typically custom agents); built-in agents' prompts live inside OpenCode and
cannot be recovered. This is best-effort and reflects the current prompt files,
which may differ from the prompt used when the session was recorded.

## Privacy & Security
This tool reads a local `opencode.db` and, optionally, local prompt files.
Handle the exported `.jsonl` files securely, as they contain your private chat
history. The exporter never reads OpenCode's `opencode.json` (which may contain
API keys) — only the database and any prompt directory you explicitly pass.
