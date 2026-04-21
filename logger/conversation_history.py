"""
logger/conversation_history.py

Stores a clean conversation history for each session.
Separate from the detailed session logger — this is just the dialogue.

Structure of conversation_history.json:
[
  {
    "patient_id": "patient_001",
    "session_id": "20260315_180911",
    "log_file": "logs/patient_001_20260315_180911.json",
    "started_at": "2026-03-15T18:09:11",
    "conversation": [
      {"role": "patient", "message": "..."},
      {"role": "therapist", "message": "..."},
      ...
    ]
  },
  ...
]
"""

import json
import os
from datetime import datetime


HISTORY_FILE = "logs/conversation_history.json"


def _load():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def _save(data):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _find_session(data, patient_id, session_id):
    for entry in data:
        if entry["patient_id"] == patient_id and str(entry["session_id"]) == str(session_id):
            return entry
    return None


def add_turn(patient_id: str, session_id: str, patient_message: str, therapist_response: str):
    """
    Append a patient + therapist turn to the conversation history.
    Creates a new session entry if one does not exist yet.
    """
    data = _load()
    session = _find_session(data, patient_id, session_id)

    if session is None:
        session = {
            "patient_id": patient_id,
            "session_id": session_id,
            "log_file": f"logs/{patient_id}_{session_id}.json",
            "started_at": datetime.now().isoformat(),
            "conversation": []
        }
        data.append(session)

    session["conversation"].append({
        "role": "patient",
        "message": patient_message
    })
    session["conversation"].append({
        "role": "therapist",
        "message": therapist_response
    })

    _save(data)