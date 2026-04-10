"""
logger/session_logger.py

Logs every turn of the therapy session as structured JSON.
Captures all intermediate steps so you can compare models, prompts, and responses later.
"""

import json
import os
from datetime import datetime


class SessionLogger:

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def _session_path(self, patient_id, session_id):
        return os.path.join(self.log_dir, f"{patient_id}_{session_id}.json")

    def _load_session(self, patient_id, session_id):
        path = self._session_path(patient_id, session_id)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        # New session skeleton
        return {
            "session_id": session_id,
            "patient_id": patient_id,
            "started_at": datetime.now().isoformat(),
            "model": None,
            "turns": []
        }

    def _save_session(self, session_data, patient_id, session_id):
        path = self._session_path(patient_id, session_id)
        with open(path, "w") as f:
            json.dump(session_data, f, indent=2)

    def _to_lines(self, text: str) -> list:
        """Convert a multiline string to a list of lines for readable JSON storage."""
        if not text:
            return []
        return [line for line in text.splitlines() if line.strip()]

    def log_turn(
        self,
        patient_id: str,
        session_id: str,
        turn: int,
        model_name: str,
        user_message: str,
        extracted: dict,
        context_keywords: list,
        kg_subgraph: dict,
        rag_chunks: list,
        triplets: list,
        session_phase: dict,
        prompt: str,
        cot_reasoning: str,
        final_response: str,
    ):
        session = self._load_session(patient_id, session_id)

        if not session["model"]:
            session["model"] = model_name

        turn_log = {
            "turn": turn,
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "step1_extraction": extracted,
            "step2_rag_chunks": rag_chunks,
            "step3_context_keywords": context_keywords,
            "step3_kg_subgraph": kg_subgraph,
            "step4_session_phase": session_phase,
            "step5_triplets": triplets,
            "step6_prompt": self._to_lines(prompt),
            "step7_response": self._to_lines(final_response),
        }

        session["turns"].append(turn_log)
        self._save_session(session, patient_id, session_id)
        print(f"[Logger] Turn {turn} saved → {self._session_path(patient_id, session_id)}")