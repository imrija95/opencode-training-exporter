#!/usr/bin/env python3
"""Export OpenCode chat sessions into training-ready JSONL.

The exporter reads OpenCode's local SQLite database and emits one JSON object
per turn (a user prompt plus the assistant trajectory that answered it). It
reads the *materialized* `message` and `part` tables, which hold the final,
consolidated state of every message and part. This avoids the streaming event
log (`event`), where each part is written many times as it streams, so tool
calls, tool results and text are captured exactly once and in order.

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
# Optional: a directory of `{agent}.md` system-prompt files. When set, the
# exporter attaches the matching agent prompt as `system` on a best-effort
# basis. The system prompt is NOT stored in the OpenCode DB (see README), so it
# is empty unless this directory is provided and contains a file for the agent.
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


def _steps_from_parts(parts):
    """Turn an assistant message's ordered parts into trajectory steps."""
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
            state = part.get("state") or {}
            status = state.get("status")
            tool_input = state.get("input")
            if tool_input is None:
                tool_input = {}
            steps.append({"kind": "tool_call", "tool": tool, "input": tool_input})
            output = state.get("output")
            if output:
                steps.append({
                    "kind": "tool_result",
                    "tool": tool,
                    "status": status or "completed",
                    "output": output,
                })
            elif status == "error":
                steps.append({
                    "kind": "tool_result",
                    "tool": tool,
                    "status": "error",
                    "output": state.get("error") or "",
                })
    return steps


def _user_text_from_parts(parts):
    chunks = []
    for part in parts:
        if part.get("type") == "text":
            text = (part.get("text") or "").strip()
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def build_records(conn):
    """Yield one training record per turn across all sessions."""
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # messages grouped by session, in chronological order
    messages_by_session = {}
    for row in cur.execute(
        "SELECT id, session_id, data FROM message ORDER BY time_created ASC, id ASC"
    ):
        try:
            data = json.loads(row["data"])
        except (json.JSONDecodeError, TypeError):
            continue
        messages_by_session.setdefault(row["session_id"], []).append((row["id"], data))

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

    for session_id, messages in messages_by_session.items():
        turn = None

        def flush(t):
            if t is None:
                return None
            has_content = any(
                s["kind"] in ("text", "tool_call", "tool_result")
                for s in t["trajectory"]
            )
            if not t["user"].strip() or not has_content:
                return None
            return t

        for msg_id, data in messages:
            role = data.get("role")
            parts = parts_by_message.get(msg_id, [])
            if role == "user":
                # A new user message after a completed exchange starts a new turn.
                if turn is not None and turn["trajectory"]:
                    ready = flush(turn)
                    if ready:
                        yield ready
                    turn = None
                if turn is None:
                    turn = {
                        "session_id": session_id,
                        "agent": None,
                        "model": None,
                        "system": "",
                        "user": "",
                        "assistant": "",
                        "trajectory": [],
                    }
                utext = _user_text_from_parts(parts)
                if utext:
                    turn["user"] = (turn["user"] + "\n" + utext).strip() if turn["user"] else utext
            elif role == "assistant":
                if turn is None:
                    turn = {
                        "session_id": session_id,
                        "agent": None,
                        "model": None,
                        "system": "",
                        "user": "",
                        "assistant": "",
                        "trajectory": [],
                    }
                if turn["agent"] is None:
                    turn["agent"] = data.get("agent") or ""
                    turn["model"] = _model_id(data.get("model"))
                turn["trajectory"].extend(_steps_from_parts(parts))

        ready = flush(turn)
        if ready:
            yield ready


def finalize(record, prompt_cache):
    """Fill in the convenience `assistant` field and best-effort `system`."""
    record["assistant"] = "\n".join(
        s["text"] for s in record["trajectory"] if s["kind"] == "text"
    )
    record["system"] = _load_prompt(record.get("agent"), prompt_cache)
    return record


def export_data(append=False):
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
        prompt_cache = {}
        seen = set()
        records = []
        for rec in build_records(conn):
            rec = finalize(rec, prompt_cache)
            line = json.dumps(rec, ensure_ascii=False, sort_keys=True)
            if line in seen:
                continue
            seen.add(line)
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
        print(f"Appended {written} new rows to {OUTPUT_FILE} ({len(records)} unique turns in DB)")
    else:
        # Clean rebuild: atomic replace so a running OpenCode/readers never see
        # a half-written file, and re-running yields an identical output.
        tmp = OUTPUT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for line in records:
                f.write(line + "\n")
        os.replace(tmp, OUTPUT_FILE)
        print(f"Exported {len(records)} turns to {OUTPUT_FILE}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export OpenCode sessions to training JSONL.")
    parser.add_argument("--db", help="Path to opencode.db (overrides OPENCODE_DB_PATH).")
    parser.add_argument("--output", help="Output JSONL path (overrides OPENCODE_OUTPUT_PATH).")
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

    return export_data(append=args.append)


if __name__ == "__main__":
    raise SystemExit(main())
