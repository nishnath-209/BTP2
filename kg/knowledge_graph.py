"""
kg/knowledge_graph.py

JSON-backed Knowledge Graph for patient profiles.
Interface is storage-agnostic — swap JSON for Neo4j later without touching the pipeline.

KG update strategy:
- smoking_status: always overwrite with new value, keep history in smoking_status_history
- triggers: substring-based dedup to avoid near-duplicates like "stress" / "stressed",
            "after food" / "after eating food". LLM extraction prompt uses standard
            labels so most collisions are caught before reaching here.
- past_strategies: exact string dedup (strategy names are usually distinct)
- session_notes: only logged when meaningful information was extracted (not raw chat)
- motivation_reason: list, deduped by exact string
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
                "motivation_reason": [],         # why they want to quit
                "triggers": [],                  # [{trigger, turn}]
                "past_strategies": [],           # [{strategy, outcome}]
                "smoking_status": None,
                "smoking_status_history": [],   # previous values before overwrite
                "quit_goal": None,
                "session_notes": [],             # [{turn, note}] — only meaningful turns
            }
        return self.profiles[patient_id]

    # ------------------------------------------------------------------
    # Trigger deduplication — substring-based normalization
    # ------------------------------------------------------------------

    def _is_duplicate_trigger(self, new_trigger: str, existing_triggers_set: set) -> bool:
        """
        Returns True if new_trigger is semantically redundant with any existing trigger.
        Uses substring containment to catch near-duplicates:
          "stress" vs "stressed" → duplicate
          "after food" vs "after eating food" → duplicate
          "work" vs "work stress" → duplicate
        """
        new = new_trigger.lower().strip()
        for existing in existing_triggers_set:
            e = existing.lower().strip()
            if new == e or new in e or e in new:
                return True
        return False

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
          "triggers": ["stress", "after meals"],
          "past_strategies": [{"strategy": "nicotine gum", "outcome": "helped a little"}],
          "is_closing": false
        }
        """
        profile = self._get_or_create(patient_id)

        # quit_goal — overwrite if new value present
        if extracted.get("quit_goal"):
            profile["quit_goal"] = extracted["quit_goal"]

        # motivation_reason — accumulate, dedup by exact string
        if extracted.get("motivation_reason"):
            reason = extracted["motivation_reason"]
            if isinstance(reason, list):
                for r in reason:
                    if r and r not in profile["motivation_reason"]:
                        profile["motivation_reason"].append(r)
            elif reason and reason not in profile["motivation_reason"]:
                profile["motivation_reason"].append(reason)

        # smoking_status — always overwrite, keep history
        if extracted.get("smoking_status"):
            new_status = extracted["smoking_status"]
            if profile["smoking_status"] and profile["smoking_status"] != new_status:
                # Archive previous value before overwriting
                if profile["smoking_status"] not in profile.get("smoking_status_history", []):
                    profile.setdefault("smoking_status_history", []).append(
                        profile["smoking_status"]
                    )
            profile["smoking_status"] = new_status

        # triggers — substring-based dedup
        existing_trigger_names = {t["trigger"] for t in profile["triggers"]}
        for t in extracted.get("triggers", []):
            trigger_name = t if isinstance(t, str) else t.get("trigger", "")
            trigger_name = trigger_name.strip()
            if trigger_name and not self._is_duplicate_trigger(trigger_name, existing_trigger_names):
                profile["triggers"].append({
                    "trigger": trigger_name,
                    "turn": turn,
                })
                existing_trigger_names.add(trigger_name)

        # past_strategies — exact string dedup
        existing_strategies = {s["strategy"] for s in profile["past_strategies"]}
        for s in extracted.get("past_strategies", []):
            strategy_name = s if isinstance(s, str) else s.get("strategy", "")
            strategy_name = strategy_name.strip()
            if strategy_name and strategy_name not in existing_strategies:
                profile["past_strategies"].append({
                    "strategy": strategy_name,
                    "outcome": s.get("outcome", "unknown") if isinstance(s, dict) else "unknown"
                })
                existing_strategies.add(strategy_name)

        # session_notes — only log turns where meaningful info was extracted
        # Avoids storing "yes", "i dont know", "ok" as notes
        has_meaningful_extraction = any([
            extracted.get("quit_goal"),
            extracted.get("motivation_reason"),
            extracted.get("smoking_status"),
            extracted.get("triggers"),
            extracted.get("past_strategies"),
        ])
        if has_meaningful_extraction and raw_message.strip():
            profile["session_notes"].append({
                "turn": turn,
                "note": raw_message.strip(),
            })

        self.save()

    # ------------------------------------------------------------------
    # Subgraph retrieval
    # ------------------------------------------------------------------

    def get_subgraph(self, patient_id, context_keywords: list = None) -> dict:
        """
        Return a relevant subgraph of the patient profile based on context keywords.

        Instead of dumping the whole KG into the prompt, we select:
        - Always: smoking_status, quit_goal, motivation_reason
        - Contextually: triggers that match current context keywords
        - Always: all past strategies (usually few, all relevant)
        - No session_notes — history block in the prompt already covers recent context
        """
        profile = self._get_or_create(patient_id)

        subgraph = {
            "motivation_reason": profile.get("motivation_reason"),
            "relevant_triggers": [],
            "relevant_strategies": profile["past_strategies"],
            "smoking_status": profile.get("smoking_status"),
            "quit_goal": profile.get("quit_goal"),
        }

        if not context_keywords:
            subgraph["relevant_triggers"] = profile["triggers"][:3]
            return subgraph

        keywords_lower = [k.lower() for k in context_keywords]

        # Filter triggers by keyword match (substring)
        for t in profile["triggers"]:
            if any(kw in t["trigger"].lower() for kw in keywords_lower):
                subgraph["relevant_triggers"].append(t)

        # Fallback: return all triggers if none matched
        if not subgraph["relevant_triggers"]:
            subgraph["relevant_triggers"] = profile["triggers"]
            # print("[KG] Trigger keyword match failed — returning all triggers as fallback")

        return subgraph

    # ------------------------------------------------------------------
    # Subgraph → prompt text
    # ------------------------------------------------------------------

    def subgraph_to_text(self, subgraph: dict) -> str:
        """Convert subgraph dict to a concise text block for prompt injection."""

        lines = ["[Patient Profile]"]

        if subgraph.get("smoking_status"):
            lines.append(f"- Smoking status: {subgraph['smoking_status']}")

        if subgraph.get("quit_goal"):
            lines.append(f"- Quit goal: {subgraph['quit_goal']}")

        if subgraph.get("motivation_reason"):
            reasons = subgraph["motivation_reason"]
            if isinstance(reasons, list) and reasons:
                lines.append(f"- Motivation to quit: {', '.join(reasons)}")
            elif isinstance(reasons, str):
                lines.append(f"- Motivation to quit: {reasons}")

        if subgraph.get("relevant_triggers"):
            trigger_strs = [t["trigger"] for t in subgraph["relevant_triggers"]]
            lines.append(f"- Known triggers: {', '.join(trigger_strs)}")

        if subgraph.get("relevant_strategies"):
            strat_strs = []
            for s in subgraph["relevant_strategies"]:
                text = s["strategy"]
                if s.get("outcome") and s["outcome"] != "unknown":
                    text += f" → {s['outcome']}"
                strat_strs.append(text)
            lines.append(f"- Tried strategies: {', '.join(strat_strs)}")

        return "\n".join(lines)