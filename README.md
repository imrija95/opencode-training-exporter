# OpenCode Training Data Exporter

A utility to export agent chat sessions, tool trajectories, and model outputs
from the OpenCode local database into a training-ready JSONL dataset in the
standard Hugging Face chat format.

## Features
- **Standard `messages` format**: One record per session, with a `messages`
  list of `{role, content}` objects — the field trainers consume via a chat
  template (`tokenizer.apply_chat_template`, TRL `SFTTrainer`, etc.).
- **Tool calls *and* results (OpenAI convention)**: Assistant tool use is
  emitted as `tool_calls` (with `arguments` as a JSON string), and each tool
  result is a `role: "tool"` message paired back by `tool_call_id`. Error
  results are included.
- **Inline reasoning**: The assistant's reasoning is inlined as
  `<think>...</think>` at the start of the assistant content — the convention
  used by OLMo 3 / Dolci-Think SFT data.
- **Idempotent by default**: Re-running rebuilds the output atomically, so the
  same database always yields the same file with no duplicate rows. An optional
  `--append` mode adds only new, deduplicated rows to an existing file.
- **Clean data**: Reads OpenCode's materialized `message` and `part` tables (the
  final consolidated state), not the streaming event log, so text and tool calls
  are never duplicated by streaming updates. Sessions with no usable assistant
  content are dropped.
- **Metadata**: Each record carries `id` (session id), `source`, and the `agent`
  and `model` that produced it.
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
| `--source` | `OPENCODE_SOURCE` | Value for each record's `source` field | `opencode` |
| `--prompts-dir` | `OPENCODE_PROMPTS_DIR` | Directory of `{agent}.md` system-prompt files (see below) | unset |
| `--append` | — | Append new deduplicated rows instead of rebuilding | rebuild |

Example:
```bash
OPENCODE_OUTPUT_PATH="/path/to/my/dataset.jsonl" python3 exporter.py
```

The database is opened **read-only and immutable**, so the exporter can run
safely while OpenCode is in use; it never locks or writes the database.

## Data Format
The exporter produces a `.jsonl` file where each line is one session:

```json
{
  "id": "ses_...",
  "source": "opencode",
  "agent": "coder",
  "model": "provider/model-id",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "please list usb"},
    {"role": "assistant", "content": "<think>I should run lsusb</think>",
     "tool_calls": [
       {"id": "c1", "type": "function",
        "function": {"name": "bash", "arguments": "{\"command\": \"lsusb\"}"}}
     ]},
    {"role": "tool", "tool_call_id": "c1", "content": "Bus 001 ..."},
    {"role": "assistant", "content": "Found the device."}
  ]
}
```

Field notes:
- `messages` is the training field, in the standard Hugging Face chat shape.
  Roles are `system`, `user`, `assistant`, and `tool`.
- **Reasoning** is inlined as `<think>...</think>` at the start of the assistant
  `content`; the answer text follows after `</think>` (OLMo 3 / Dolci-Think
  convention). There are no `<answer>` tags.
- **Tool calls** use the OpenAI convention: an assistant message carries
  `tool_calls` (each with a JSON-string `arguments`), and every tool result is a
  `role: "tool"` message referring back via `tool_call_id`. A tool-only
  assistant message has an empty `content`.
- `id`, `source`, `agent`, and `model` are metadata columns; trainers that read
  only `messages` ignore them.
- Structural markers (`step-start`, `step-finish`, `compaction`) are omitted.

### The `system` message
The system/agent prompt is **not stored in the OpenCode database** — OpenCode
injects it at request time. As a result no `system` message is emitted by
default. As a best-effort convenience, if you point `--prompts-dir` (or
`OPENCODE_PROMPTS_DIR`) at a directory of `{agent}.md` files, the exporter
prepends the matching agent's prompt as a `system` message. This only works for
agents whose prompt files exist on disk (typically custom agents); built-in
agents' prompts live inside OpenCode and cannot be recovered. It is best-effort
and reflects the current prompt files, which may differ from the prompt used
when the session was recorded.

Note: the top-level OpenAI `tools` (function schemas) list is not emitted — the
schemas are not present in the OpenCode database.

## Privacy & Security
This tool reads a local `opencode.db` and, optionally, local prompt files.
Handle the exported `.jsonl` files securely, as they contain your private chat
history. The exporter never reads OpenCode's `opencode.json` (which may contain
API keys) — only the database and any prompt directory you explicitly pass.
