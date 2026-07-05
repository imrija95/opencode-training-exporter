#!/usr/bin/env python3
"""Export OpenCode chat sessions into a training-ready JSONL dataset.

The exporter reads OpenCode's local SQLite database and emits one JSON object
per session in the widely used Hugging Face chat format: a `messages` list of
`{role, content}` objects (the field trainers consume via a chat template).
Tool use follows the OpenAI convention — assistant `tool_calls` plus
`role: "tool"` results paired by `tool_call_id`. Assistant reasoning is inlined
as `<think>...</think>` at the start of the assistant content, the convention
used by OLMo 3 / Dolci-Think SFT data.

It reads the *materialized* `message` and `part` tables (the final consolidated
state of each message and part), not the streaming `event` log where each part
is rewritten many times as it streams. This keeps text and tool calls from
being duplicated by streaming updates.

Record schema (one line per session):

    {
      "id": "ses_...",                 # session id
      "source": "opencode",            # origin identifier (see --source)
      "agent": "coder",                # metadata: the agent that produced it
      "model": "provider/model-id",    # metadata
      "messages": [
        {"role": "system", "content": "..."},        # only if --prompts-dir resolves it
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "<think>...</think>",
         "tool_calls": [{"id": "...", "type": "function",
                         "function": {"name": "bash", "arguments": "{...}"}}]},
        {"role": "tool", "tool_call_id": "...", "content": "..."},
        {"role": "assistant", "content": "Final answer."}
      ]
    }

Standard library only; no external dependencies.
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# --- Configuration (environment variables override the defaults) -------------
DB_PATH = os.getenv("OPENCODE_DB_PATH", os.path.expanduser("~/.local/share/opencode/opencode.db"))
OUTPUT_FILE = os.getenv("OPENCODE_OUTPUT_PATH", os.path.expanduser("~/opencode_training_data.jsonl"))
# Value written to each record's `source` field (dataset-origin identifier).
SOURCE = os.getenv("OPENCODE_SOURCE", "opencode")
# Optional: a directory of `{agent}.md` system-prompt files. When set, the
# exporter prepends the matching agent prompt as a `system` message on a
# best-effort basis. The system prompt is NOT stored in the OpenCode DB (see
# README), so no system message is emitted unless this directory is provided
# and contains a file for the agent.
PROMPTS_DIR = os.getenv("OPENCODE_PROMPTS_DIR", "")

# Part types that carry no training content; they are structural markers.
_SKIP_PART_TYPES = {"step-start", "step-finish", "compaction"}


def _connect_readonly(db_path):
    """Open the database read-only and immutable so a running OpenCode instance
    is never disturbed (no locks taken, no writes possible)."""
    uri = f"file:{Path(db_path).as_posix()}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def _model_id(model):
    """Normalize an OpenCode model object to a short string."""
    if isinstance(model, dict):
        provider = model.get("providerID") or ""
        mid = model.get("modelID") or model.get("id") or ""
        if provider and mid:
            return f"{provider}/{mid}"
        return mid or provider
    if isinstance(model, str):
        return model
    return ""


def _load_prompt(agent, cache):
    """Best-effort system prompt for an agent from PROMPTS_DIR/{agent}.md."""
    if not PROMPTS_DIR or not agent:
        return ""
    if agent in cache:
        return cache[agent]
    path = os.path.join(os.path.expanduser(PROMPTS_DIR), f"{agent}.md")
    text = ""
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except OSError:
            text = ""
    cache[agent] = text
    return text


def _tool_arguments(value):
    """Tool-call arguments as a JSON string (the OpenAI function-calling form)."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _steps_from_parts(parts):
    """Turn an assistant message's ordered parts into trajectory steps.

    A single OpenCode tool part holds BOTH the call (state.input) and the result
    (state.output), so it becomes two steps: a `tool_call` and a `tool_result`.
    """
    steps = []
    for part in parts:
        ptype = part.get("type")
        if ptype in _SKIP_PART_TYPES:
            continue
        if ptype == "reasoning":
            text = (part.get("text") or "").strip()
            if text:
                steps.append({"kind": "reasoning", "text": text})
        elif ptype == "text":
            text = (part.get("text") or "").strip()
            if text:
                steps.append({"kind": "text", "text": text})
        elif ptype == "tool":
            tool = part.get("tool") or ""
            call_id = part.get("callID") or part.get("id") or ""
            state = part.get("state") or {}
            status = state.get("status")
            tool_input = state.get("input")
            if tool_input is None:
                tool_input = {}
            steps.append({"kind": "tool_call", "tool": tool, "call_id": call_id, "input": tool_input})
            output = state.get("output")
            if output:
                steps.append({"kind": "tool_result", "tool": tool, "call_id": call_id,
                              "status": status or "completed", "output": output})
            elif status == "error":
                steps.append({"kind": "tool_result", "tool": tool, "call_id": call_id,
                              "status": "error", "output": state.get("error") or ""})
    return steps


def _user_text_from_parts(parts):
    chunks = []
    for part in parts:
        if part.get("type") == "text":
            text = (part.get("text") or "").strip()
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _assistant_content(think_texts, body_texts):
    """Assemble assistant content: inline `<think>...</think>` then the answer."""
    parts = []
    think = "\n\n".join(t for t in think_texts if t).strip()
    if think:
        parts.append(f"<think>{think}</think>")
    body = "\n\n".join(t for t in body_texts if t).strip()
    if body:
        parts.append(body)
    return "\n".join(parts)


def _steps_to_messages(steps):
    """Convert an ordered assistant trajectory into OpenAI-style chat messages.

    reasoning/text accumulate into an assistant message's content; a `tool_call`
    attaches to that message's `tool_calls`; a `tool_result` becomes a
    `role: "tool"` message. A new assistant/tool exchange begins when fresh
    reasoning/text/tool_call arrives after a tool result was produced.
    """
    messages = []
    think_buf, text_buf, tool_calls, tool_results = [], [], [], []
    synth = [0]

    def flush():
        content = _assistant_content(think_buf, text_buf)
        if content or tool_calls:
            msg = {"role": "assistant", "content": content}
            if tool_calls:
                msg["tool_calls"] = list(tool_calls)
            messages.append(msg)
        for tid, out in tool_results:
            messages.append({"role": "tool", "tool_call_id": tid, "content": out})
        think_buf.clear()
        text_buf.clear()
        tool_calls.clear()
        tool_results.clear()

    def new_id():
        synth[0] += 1
        return f"call_{synth[0]}"

    for step in steps:
        kind = step["kind"]
        if kind in ("reasoning", "text", "tool_call") and tool_results:
            flush()
        if kind == "reasoning":
            think_buf.append(step["text"])
        elif kind == "text":
            text_buf.append(step["text"])
        elif kind == "tool_call":
            cid = step.get("call_id") or new_id()
            tool_calls.append({
                "id": cid,
                "type": "function",
                "function": {"name": step.get("tool") or "", "arguments": _tool_arguments(step.get("input"))},
            })
        elif kind == "tool_result":
            cid = step.get("call_id") or (tool_calls[-1]["id"] if tool_calls else new_id())
            tool_results.append((cid, step.get("output") or ""))
    flush()
    return messages


def build_records(conn, source):
    """Yield one training record (a full session conversation) per session."""
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # messages grouped by session, in chronological order (preserve first-seen order)
    messages_by_session = {}
    session_order = []
    for row in cur.execute(
        "SELECT id, session_id, data FROM message ORDER BY time_created ASC, id ASC"
    ):
        try:
            data = json.loads(row["data"])
        except (json.JSONDecodeError, TypeError):
            continue
        sid = row["session_id"]
        if sid not in messages_by_session:
            messages_by_session[sid] = []
            session_order.append(sid)
        messages_by_session[sid].append((row["id"], data))

    # parts grouped by message, in order
    parts_by_message = {}
    for row in cur.execute(
        "SELECT message_id, data FROM part ORDER BY time_created ASC, id ASC"
    ):
        try:
            data = json.loads(row["data"])
        except (json.JSONDecodeError, TypeError):
            continue
        parts_by_message.setdefault(row["message_id"], []).append(data)

    prompt_cache = {}

    for session_id in session_order:
        out_messages = []
        pending = []          # assistant steps across consecutive assistant messages
        agent = ""
        model = ""

        for msg_id, data in messages_by_session[session_id]:
            role = data.get("role")
            parts = parts_by_message.get(msg_id, [])
            if role == "user":
                if pending:
                    out_messages.extend(_steps_to_messages(pending))
                    pending = []
                utext = _user_text_from_parts(parts)
                if utext:
                    out_messages.append({"role": "user", "content": utext})
            elif role == "assistant":
                if not agent:
                    agent = data.get("agent") or ""
                    model = _model_id(data.get("model"))
                pending.extend(_steps_from_parts(parts))
        if pending:
            out_messages.extend(_steps_to_messages(pending))

        # keep only sessions with usable assistant content
        has_assistant = any(
            m["role"] == "assistant" and (m.get("content") or m.get("tool_calls"))
            for m in out_messages
        )
        if not has_assistant:
            continue

        system = _load_prompt(agent, prompt_cache)
        if system:
            out_messages.insert(0, {"role": "system", "content": system})

        yield {
            "id": session_id,
            "source": source,
            "agent": agent,
            "model": model,
            "messages": out_messages,
        }


def export_data(append=False, source="opencode"):
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}", file=sys.stderr)
        return 1

    conn = _connect_readonly(DB_PATH)
    try:
        required = {"message", "part"}
        have = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing = required - have
        if missing:
            print(
                f"Database at {DB_PATH} is missing required table(s): "
                f"{', '.join(sorted(missing))}. Is this an OpenCode database?",
                file=sys.stderr,
            )
            return 1
        seen = set()
        records = []
        for rec in build_records(conn, source):
            key = json.dumps(rec, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            records.append(json.dumps(rec, ensure_ascii=False))
    finally:
        conn.close()

    out_dir = os.path.dirname(OUTPUT_FILE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if append:
        # Append only records not already present; the output file is the state.
        existing = set()
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        existing.add(json.dumps(json.loads(line), ensure_ascii=False, sort_keys=True))
                    except json.JSONDecodeError:
                        continue
        written = 0
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for line in records:
                key = json.dumps(json.loads(line), ensure_ascii=False, sort_keys=True)
                if key in existing:
                    continue
                existing.add(key)
                f.write(line + "\n")
                written += 1
        print(f"Appended {written} new sessions to {OUTPUT_FILE} ({len(records)} unique sessions in DB)")
    else:
        # Clean rebuild: atomic replace so a running OpenCode/readers never see
        # a half-written file, and re-running yields an identical output.
        tmp = OUTPUT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for line in records:
                f.write(line + "\n")
        os.replace(tmp, OUTPUT_FILE)
        print(f"Exported {len(records)} sessions to {OUTPUT_FILE}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export OpenCode sessions to training JSONL.")
    parser.add_argument("--db", help="Path to opencode.db (overrides OPENCODE_DB_PATH).")
    parser.add_argument("--output", help="Output JSONL path (overrides OPENCODE_OUTPUT_PATH).")
    parser.add_argument("--source", help="Value for the record 'source' field (overrides OPENCODE_SOURCE).")
    parser.add_argument("--prompts-dir", help="Directory of {agent}.md system prompts (overrides OPENCODE_PROMPTS_DIR).")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new, deduplicated rows to the existing output instead of rebuilding it.",
    )
    args = parser.parse_args(argv)

    global DB_PATH, OUTPUT_FILE, PROMPTS_DIR
    if args.db:
        DB_PATH = os.path.expanduser(args.db)
    if args.output:
        OUTPUT_FILE = os.path.expanduser(args.output)
    if args.prompts_dir:
        PROMPTS_DIR = args.prompts_dir
    source = args.source or SOURCE

    return export_data(append=args.append, source=source)


if __name__ == "__main__":
    raise SystemExit(main())
