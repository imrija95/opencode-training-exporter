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
        ("p4", "m2", 113, {"type": "tool", "tool": "bash",
                            "state": {"status": "completed", "input": {"command": "lsusb"}, "output": "Bus 001 ..."}}),
        ("p5", "m3", 121, {"type": "text", "text": "Found the device."}),
        ("p6", "m4", 201, {"type": "text", "text": "now read the file"}),
        ("p7", "m5", 211, {"type": "tool", "tool": "read",
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

    def test_records(self):
        rows = run_exporter(self.db, self.out)
        # Empty-content turn (m6/m7) is filtered; two usable turns remain.
        self.assertEqual(len(rows), 2)

        first = rows[0]
        self.assertEqual(first["user"], "please list usb")
        self.assertEqual(first["agent"], "coder")
        self.assertEqual(first["model"], "p/x")
        self.assertEqual(first["assistant"], "Found the device.")
        kinds = [s["kind"] for s in first["trajectory"]]
        # Multi-message assistant turn merged, in order, no structural markers.
        self.assertEqual(kinds, ["reasoning", "tool_call", "tool_result", "text"])
        result = first["trajectory"][2]
        self.assertEqual(result["output"], "Bus 001 ...")
        self.assertEqual(result["status"], "completed")

        # Tool-only turn with an error result is kept (empty assistant text).
        second = rows[1]
        self.assertEqual(second["assistant"], "")
        self.assertEqual([s["kind"] for s in second["trajectory"]], ["tool_call", "tool_result"])
        self.assertEqual(second["trajectory"][1]["status"], "error")

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
        rows = run_exporter(self.db, self.out, "--prompts-dir", prompts)
        self.assertTrue(all(r["system"] == "You are the coder agent." for r in rows))


if __name__ == "__main__":
    unittest.main()
