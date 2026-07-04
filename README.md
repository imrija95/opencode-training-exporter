# OpenCode Training Data Exporter

A utility to incrementally export chat sessions, prompts, and model outputs from the OpenCode local database into a training-ready JSONL format.

## Features
- **Incremental Export**: Tracks the last exported event per session to avoid duplicates.
- **Thinking Block Capture**: Specifically extracts `reasoning` blocks (thinking process) separately from the final assistant output.
- **Context Preservation**: Captures the associated `system` context (Baseline) for each session.
- **Configurable**: Supports environment variables for custom database and output paths.

## Installation
The exporter is a standalone Python script with no external dependencies beyond the standard library.

```bash
git clone https://github.com/imrija95/opencode-training-exporter
cd opencode-training-exporter
```

## Usage
Run the script directly with Python:

```bash
python3 exporter.py
```

### Configuration
You can override the default paths using environment variables:

| Variable | Description | Default |
|-----------|-------------|---------|
| `OPencode_DB_PATH` | Path to `opencode.db` | `~/.local/share/opencode/opencode.db` |
| `OPencode_STATE_PATH` | Path to export state file | `~/.opencode/export_state.json` |
| `OPencode_OUTPUT_PATH` | Path to the resulting JSONL file | `~/opencode_training_data.jsonl` |

Example:
```bash
export OPencode_OUTPUT_PATH="/path/to/my/dataset.jsonl"
python3 exporter.py
```

## Data Format
The exporter produces a `.jsonl` file where each line is a JSON object representing one turn:

```json
{
  "session_id": "ses_...",
  "system": "The system context/baseline for this session",
  "user": "The user's prompt",
  "thought": "The model's internal reasoning block",
  "assistant": "The final response text"
}
```

## Privacy & Security
This tool accesses your local `opencode.db`. Ensure you handle the exported `.jsonl` files securely, as they contain your private chat history.
