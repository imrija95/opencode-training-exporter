import sqlite3
import json
import os
from pathlib import Path

# Configuration via environment variables or defaults
DB_PATH = os.getenv("OPencode_DB_PATH", os.path.expanduser("~/.local/share/opencode/opencode.db"))
STATE_FILE = os.getenv("OPencode_STATE_PATH", os.path.expanduser("~/.opencode/export_state.json"))
OUTPUT_FILE = os.getenv("OPencode_OUTPUT_PATH", os.path.expanduser("~/opencode_training_data.jsonl"))

def get_last_sequences():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_sequences(sequences):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(sequences, f)

def export_data():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    last_sequences = get_last_sequences()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT id FROM session")
        sessions = [row["id"] for row in cursor.execute("SELECT id FROM session")]
    except sqlite3.OperationalError as e:
        print(f"Error accessing session table: {e}")
        return
    
    new_entries_count = 0
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:
        for session_id in sessions:
            last_seq = last_sequences.get(session_id, -1)
            
            # 1. Pre-calculate Role Map for this session
            role_map = {}
            cursor.execute("SELECT data FROM event WHERE aggregate_id = ? AND type = 'message.updated.1'", (session_id,))
            for row in cursor.fetchall():
                try:
                    data = json.loads(row["data"])
                    info = data.get("info", {})
                    msg_id = info.get("id")
                    role = info.get("role")
                    if msg_id and role:
                        role_map[msg_id] = role
                except:
                    continue

            # 2. Pre-calculate Context Epochs for this session
            # Mapping: sequence_number -> baseline_text
            epochs = {}
            cursor.execute("SELECT baseline_seq, baseline FROM session_context_epoch WHERE session_id = ? ORDER BY baseline_seq ASC", (session_id,))
            for row in cursor.fetchall():
                epochs[row["baseline_seq"]] = row["baseline"]

            # Fetch events after the last exported sequence
            cursor.execute("SELECT * FROM event WHERE aggregate_id = ? AND seq > ? ORDER BY seq ASC", (session_id, last_seq))
            events = cursor.fetchall()
            
            if not events:
                continue

            # Determine initial system prompt (the baseline active at the start of the exported window)
            # We find the epoch with the highest sequence <= the first event's sequence
            first_seq = events[0]["seq"]
            active_baseline = ""
            if epochs:
                # Find the most recent epoch that occurred before or at first_seq
                applicable_epochs = [seq for seq in epochs.keys() if seq <= first_seq]
                if applicable_epochs:
                    active_baseline = epochs[max(applicable_epochs)]
                else:
                    # If no epoch is found before the first event, take the very first epoch
                    active_baseline = epochs[min(epochs.keys())] if epochs else ""

            current_turn = {"session_id": session_id, "system": active_baseline, "user": "", "thought": "", "assistant": ""}
            
            for event in events:
                etype = event["type"]
                seq = event["seq"]
                try:
                    data = json.loads(event["data"])
                except:
                    continue

                # Check if we've crossed into a new epoch
                if etype == "session.next.epoch.admitted.1" or etype == "session.next.epoch.updated.1":
                    # The event data contains the new baseline or a reference to it.
                    # To be most accurate, we check if the current seq has a corresponding epoch entry
                    if seq in epochs:
                        active_baseline = epochs[seq]
                        current_turn["system"] = active_baseline

                if etype == "message.part.updated.1":
                    part = data.get("part", {})
                    msg_id = part.get("messageID")
                    part_type = part.get("type")
                    text = part.get("text", "")

                    role = role_map.get(msg_id, "unknown")

                    if role == "user":
                        if current_turn["user"] and current_turn["assistant"]:
                            outfile.write(json.dumps(current_turn) + "\n")
                            new_entries_count += 1
                            current_turn = {"session_id": session_id, "system": active_baseline, "user": "", "thought": "", "assistant": ""}
                        
                        current_turn["user"] += text
                    elif role == "assistant":
                        if part_type == "reasoning":
                            current_turn["thought"] += text
                        elif part_type == "text":
                            current_turn["assistant"] += text

            # Final turn for the session
            if current_turn["user"]:
                outfile.write(json.dumps(current_turn) + "\n")
                new_entries_count += 1

            if events:
                last_sequences[session_id] = events[-1]["seq"]

    save_sequences(last_sequences)
    conn.close()
    print(f"Exported {new_entries_count} new turns to {OUTPUT_FILE}")

if __name__ == "__main__":
    export_data()
