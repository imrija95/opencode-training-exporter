"""Tests for the OpenCode training exporter.

All data here is SYNTHETIC. The test builds a tiny database with the same shape
as OpenCode's `message`/`part` tables - no real chat content is used.
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORTER = os.path.join(REPO_ROOT, "exporter.py")


def build_fake_db(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("CREATE TABLE message (id TEXT, session_id TEXT, time_created INTEGER, time_updated INTEGER, data TEXT)")
    c.execute("CREATE TABLE part (id TEXT, message_id TEXT, session_id TEXT, time_created INTEGER, time_updated INTEGER, data TEXT)")
    S = "ses_test1"
    model = {"providerID": "p", "modelID": "x"}
    messages = [
        ("m1", 100, {"role": "user", "agent": "coder", "model": model}),
        # one user turn answered by TWO consecutive assistant messages (multi-step)
        ("m2", 110, {"role": "assistant", "agent": "coder", "model": model}),
        ("m3", 120, {"role": "assistant", "agent": "coder", "model": model}),
        ("m4", 200, {"role": "user", "agent": "coder", "model": model}),
        ("m5", 210, {"role": "assistant", "agent": "coder", "model": model}),
        # a turn whose assistant produced no usable content -> must be filtered
        ("m6", 300, {"role": "user", "agent": "coder", "model": model}),
        ("m7", 310, {"role": "assistant", "agent": "coder", "model": model}),
    ]
    for mid, t, d in messages:
        c.execute("INSERT INTO message VALUES (?,?,?,?,?)", (mid, S, t, t, json.dumps(d)))
    parts = [
        ("p1", "m1", 101, {"type": "text", "text": "please list usb"}),
        ("p2", "m2", 111, {"type": "reasoning", "text": "I should run lsusb"}),
        ("p3", "m2", 112, {"type": "step-start"}),  # structural, dropped
        ("p4", "m2", 113, {"type": "tool", "tool": "bash", "callID": "c1",
                            "state": {"status": "completed", "input": {"command": "lsusb"}, "output": "Bus 001 ..."}}),
        ("p5", "m3", 121, {"type": "text", "text": "Found the device."}),
        ("p6", "m4", 201, {"type": "text", "text": "now read the file"}),
        ("p7", "m5", 211, {"type": "tool", "tool": "read", "callID": "c2",
                            "state": {"status": "error", "input": {"filePath": "x"}, "error": "not found"}}),
        ("p8", "m7", 311, {"type": "step-finish"}),  # only a structural marker
    ]
    for pid, mid, t, d in parts:
        c.execute("INSERT INTO part VALUES (?,?,?,?,?,?)", (pid, mid, S, t, t, json.dumps(d)))
    conn.commit()
    conn.close()


def run_exporter(db, out, *extra):
    subprocess.run(
        [sys.executable, EXPORTER, "--db", db, "--output", out, *extra],
        check=True, capture_output=True, text=True,
    )
    with open(out, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class ExporterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "fake.db")
        self.out = os.path.join(self.tmp, "out.jsonl")
        build_fake_db(self.db)

    def test_schema_and_metadata(self):
        rows = run_exporter(self.db, self.out)
        # One record per session (the whole conversation), not per turn.
        self.assertEqual(len(rows), 1)
        rec = rows[0]
        self.assertEqual(rec["id"], "ses_test1")
        self.assertEqual(rec["source"], "opencode")
        self.assertEqual(rec["agent"], "coder")
        self.assertEqual(rec["model"], "p/x")
        self.assertIn("messages", rec)

    def test_messages_and_tools(self):
        rec = run_exporter(self.db, self.out)[0]
        msgs = rec["messages"]
        roles = [m["role"] for m in msgs]
        # user -> assistant(tool_call) -> tool -> assistant(text) -> user -> assistant(tool_call) -> tool
        self.assertEqual(
            roles,
            ["user", "assistant", "tool", "assistant", "user", "assistant", "tool"],
        )

        self.assertEqual(msgs[0], {"role": "user", "content": "please list usb"})

        # assistant tool-call turn: reasoning inlined as <think>, tool_call attached
        a1 = msgs[1]
        self.assertEqual(a1["content"], "<think>I should run lsusb</think>")
        call = a1["tool_calls"][0]
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["id"], "c1")
        self.assertEqual(call["function"]["name"], "bash")
        # arguments is a JSON *string* (OpenAI convention)
        self.assertIsInstance(call["function"]["arguments"], str)
        self.assertEqual(json.loads(call["function"]["arguments"]), {"command": "lsusb"})

        # tool result pairs back by tool_call_id
        self.assertEqual(msgs[2], {"role": "tool", "tool_call_id": "c1", "content": "Bus 001 ..."})

        # the follow-up text becomes a fresh assistant message (no tool_calls, no <think>)
        self.assertEqual(msgs[3], {"role": "assistant", "content": "Found the device."})

        self.assertEqual(msgs[4], {"role": "user", "content": "now read the file"})

        # error tool result is still captured
        self.assertEqual(msgs[5]["tool_calls"][0]["function"]["name"], "read")
        self.assertEqual(msgs[5]["content"], "")
        self.assertEqual(msgs[6], {"role": "tool", "tool_call_id": "c2", "content": "not found"})

    def test_idempotent_rebuild(self):
        run_exporter(self.db, self.out)
        with open(self.out, "rb") as f:
            first = f.read()
        run_exporter(self.db, self.out)
        with open(self.out, "rb") as f:
            second = f.read()
        self.assertEqual(first, second)  # byte-identical on re-run

    def test_append_dedups(self):
        run_exporter(self.db, self.out)
        with open(self.out, encoding="utf-8") as f:
            before = [l for l in f if l.strip()]
        run_exporter(self.db, self.out, "--append")
        with open(self.out, encoding="utf-8") as f:
            after = [l for l in f if l.strip()]
        self.assertEqual(len(before), len(after))  # append adds no duplicates

    def test_system_prompt_best_effort(self):
        prompts = os.path.join(self.tmp, "prompts")
        os.makedirs(prompts)
        with open(os.path.join(prompts, "coder.md"), "w", encoding="utf-8") as f:
            f.write("You are the coder agent.")
        rec = run_exporter(self.db, self.out, "--prompts-dir", prompts)[0]
        self.assertEqual(rec["messages"][0], {"role": "system", "content": "You are the coder agent."})

    def test_source_override(self):
        rec = run_exporter(self.db, self.out, "--source", "opencode-local")[0]
        self.assertEqual(rec["source"], "opencode-local")


if __name__ == "__main__":
    unittest.main()
