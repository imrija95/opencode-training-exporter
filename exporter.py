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
    # Ensure the directory for the state file exists
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

    # Get all sessions
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
            
            # Fetch all events for this session after the last exported sequence
            cursor.execute("SELECT * FROM event WHERE aggregate_id = ? AND seq > ? ORDER BY seq ASC", (session_id, last_seq))
            events = cursor.fetchall()
            
            if not events:
                continue

            # Group events into turns
            # A turn starts with a user message and ends with an assistant response (including its reasoning)
            current_turn = {"session_id": session_id, "system": "", "user": "", "thought": "", "assistant": ""}
            
            # We need to fetch the system context epoch for the session (simplified: take the latest one)
            cursor.execute("SELECT baseline FROM session_context_epoch WHERE session_id = ? ORDER BY baseline_seq DESC LIMIT 1", (session_id,))
            epoch_row = cursor.fetchone()
            if epoch_row:
                try:
                    current_turn["system"] = epoch_row["baseline"]
                except:
                    pass

            for event in events:
                etype = event["type"]
                try:
                    data = json.loads(event["data"])
                except:
                    continue

                if etype == "message.part.updated.1":
                    part = data.get("part", {})
                    msg_id = part.get("messageID")
                    part_type = part.get("type")
                    text = part.get("text", "")

                    # Determine role by looking up message info
                    cursor.execute("SELECT data FROM event WHERE aggregate_id = ? AND type = 'message.updated.1' AND data LIKE ?", 
                                   (session_id, f'%"{msg_id}"%'))
                    msg_row = cursor.fetchone()
                    role = "unknown"
                    if msg_row:
                        try:
                            msg_info = json.loads(msg_row["data"])
                            # The event data for message.updated.1 contains a dict with 'info'
                            role = msg_info.get("info", {}).get("role", "unknown")
                        except:
                            pass

                    if role == "user":
                        current_turn["user"] += text
                    elif role == "assistant":
                        if part_type == "reasoning":
                            current_turn["thought"] += text
                        elif part_type == "text":
                            current_turn["assistant"] += text

                # If we have both user and assistant content, and we hit a new user prompt or session end,
                # we can consider the turn complete. 
                if etype == "message.updated.1" and "user" in json.loads(event["data"]).get("info", {}).get("role", ""):
                    if current_turn["user"] and current_turn["assistant"]:
                        outfile.write(json.dumps(current_turn) + "\n")
                        new_entries_count += 1
                        current_turn = {"session_id": session_id, "system": current_turn["system"], "user": "", "thought": "", "assistant": ""}
            
            # Final turn for the session
            if current_turn["user"] and current_turn["assistant"]:
                outfile.write(json.dumps(current_turn) + "\n")
                new_entries_count += 1

            # Update last sequence for this session
            if events:
                last_sequences[session_id] = events[-1]["seq"]

    save_sequences(last_sequences)
    conn.close()
    print(f"Exported {new_entries_count} new turns to {OUTPUT_FILE}")

if __name__ == "__main__":
    export_data()
