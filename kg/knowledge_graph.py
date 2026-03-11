"""
kg/knowledge_graph.py

JSON-backed Knowledge Graph for patient profiles.
Interface is storage-agnostic — swap JSON for Neo4j later without touching the pipeline.
"""

import json
import os


class KnowledgeGraph:

    def __init__(self, storage_path="kg/patient_profiles.json"):
        self.storage_path = storage_path
        self.profiles = {}
        self.load()

    # ------------------------------------------------------------------
    # Storage 
    # ------------------------------------------------------------------

    def load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                self.profiles = json.load(f)

    def save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(self.profiles, f, indent=2)

    def _get_or_create(self, patient_id):
        if patient_id not in self.profiles:
            self.profiles[patient_id] = {
                "patient_id": patient_id,
                "smoking_status": None,
                "quit_goal": None,
                "triggers": [],          # [{trigger, intensity, context}]
                "past_strategies": [],   # [{strategy, outcome}]
                "session_notes": [],     # [{turn, note}]
            }
        return self.profiles[patient_id]

    # ------------------------------------------------------------------
    # Update KG from LLM-extracted JSON
    # ------------------------------------------------------------------

    def update(self, patient_id, extracted: dict, raw_message: str, turn: int = 0):
        """
        Update the KG for a patient from LLM-extracted structured data.

        extracted example:
        {
          "quit_goal": "quit smoking",
          "smoking_status": "10 cigarettes/day",
          "triggers": ["stress", "morning coffee"],
          "past_strategies": [{"strategy": "nicotine gum", "outcome": "helped"}]
        }
        """
        profile = self._get_or_create(patient_id)

        if extracted.get("quit_goal"):
            profile["quit_goal"] = extracted["quit_goal"]

        if extracted.get("smoking_status"):
            profile["smoking_status"] = extracted["smoking_status"]

        # Merge triggers (avoid duplicates)
        existing_triggers = {t["trigger"] for t in profile["triggers"]}
        for t in extracted.get("triggers", []):
            trigger_name = t if isinstance(t, str) else t.get("trigger", "")
            if trigger_name and trigger_name not in existing_triggers:
                profile["triggers"].append({
                    "trigger": trigger_name,
                    "intensity": t.get("intensity", "unknown") if isinstance(t, dict) else "unknown",
                    "context": t.get("context", "") if isinstance(t, dict) else ""
                })
                existing_triggers.add(trigger_name)

        # Merge strategies (avoid duplicates)
        existing_strategies = {s["strategy"] for s in profile["past_strategies"]}
        for s in extracted.get("past_strategies", []):
            strategy_name = s if isinstance(s, str) else s.get("strategy", "")
            if strategy_name and strategy_name not in existing_strategies:
                profile["past_strategies"].append({
                    "strategy": strategy_name,
                    "outcome": s.get("outcome", "unknown") if isinstance(s, dict) else "unknown"
                })
                existing_strategies.add(strategy_name)

        # Always append session note
        if raw_message.strip():
            profile["session_notes"].append({
                "turn": turn,
                "note": raw_message.strip()
            })

        self.save()

    # ------------------------------------------------------------------
    # Subgraph retrieval
    # ------------------------------------------------------------------

    def get_subgraph(self, patient_id, context_keywords: list = None) -> dict:
        """
        Return a relevant subgraph of the patient profile based on context keywords.

        Instead of dumping the whole KG into the prompt, we select:
        - Always: smoking_status, quit_goal
        - Contextually: triggers and strategies that match the current message context
        - Recent notes: last 3 session notes only
        """
        profile = self._get_or_create(patient_id)

        subgraph = {
            "smoking_status": profile.get("smoking_status"),
            "quit_goal": profile.get("quit_goal"),
            "relevant_triggers": [],
            "relevant_strategies": [],
            "recent_notes": profile["session_notes"][-3:] if profile["session_notes"] else []
        }

        if not context_keywords:
            # No context. return top triggers and strategies
            subgraph["relevant_triggers"] = profile["triggers"][:3]
            subgraph["relevant_strategies"] = profile["past_strategies"][:3]
            return subgraph

        keywords_lower = [k.lower() for k in context_keywords]

        # Filter triggers by keyword match
        for t in profile["triggers"]:
            if any(kw in t["trigger"].lower() for kw in keywords_lower):
                subgraph["relevant_triggers"].append(t)

        # If no match, fallback to all triggers (there may not be many)
        if not subgraph["relevant_triggers"]:
            subgraph["relevant_triggers"] = profile["triggers"]

        # All strategies are usually relevant (patient hasn't tried many)
        subgraph["relevant_strategies"] = profile["past_strategies"]

        return subgraph

    def subgraph_to_text(self, subgraph: dict) -> str:
        """Convert subgraph dict to a concise text block for prompt injection."""
        lines = ["[Patient Profile]"]

        if subgraph.get("smoking_status"):
            lines.append(f"- Smoking status: {subgraph['smoking_status']}")

        if subgraph.get("quit_goal"):
            lines.append(f"- Quit goal: {subgraph['quit_goal']}")

        if subgraph.get("relevant_triggers"):
            trigger_strs = []
            for t in subgraph["relevant_triggers"]:
                s = t["trigger"]
                if t.get("intensity") and t["intensity"] != "unknown":
                    s += f" (intensity: {t['intensity']})"
                trigger_strs.append(s)
            lines.append(f"- Known triggers: {', '.join(trigger_strs)}")

        if subgraph.get("relevant_strategies"):
            strat_strs = []
            for s in subgraph["relevant_strategies"]:
                text = s["strategy"]
                if s.get("outcome") and s["outcome"] != "unknown":
                    text += f" → {s['outcome']}"
                strat_strs.append(text)
            lines.append(f"- Tried strategies: {', '.join(strat_strs)}")

        if subgraph.get("recent_notes"):
            recent = [n["note"] for n in subgraph["recent_notes"]]
            lines.append(f"- Recent context: {' | '.join(recent)}")

        return "\n".join(lines)