"""
kg/triplet_store.py

Structured clinical triplets: Trigger -> Symptom -> Technique
Used during CoT reasoning to match patient's current trigger to evidence-based techniques.
These are loaded from a JSON file
"""

import json
import os


TRIPLETS_PATH = "kg/triplets.json"


def load_triplets():
    if os.path.exists(TRIPLETS_PATH):
        with open(TRIPLETS_PATH, "r") as f:
            return json.load(f)
    return []


def get_relevant_triplets(context_keywords: list, top_k: int = 3) -> list:
    """
    Return triplets where trigger or symptom matches any of the context keywords.
    Falls back to returning first top_k triplets if no match.
    """
    triplets = load_triplets()
    keywords_lower = [k.lower() for k in context_keywords]

    matched = []
    for t in triplets:
        trigger = t.get("trigger", "").lower()
        symptom = t.get("symptom", "").lower()
        if any(kw in trigger or kw in symptom for kw in keywords_lower):
            matched.append(t)

    if not matched:
        matched = triplets[:top_k]

    return matched[:top_k]


def triplets_to_text(triplets: list) -> str:
    """Format triplets as a structured text block for the prompt."""
    if not triplets:
        return ""

    lines = ["[Clinical Triplets — Trigger -> Symptom -> Technique]"]
    for t in triplets:
        lines.append(
            f"- Trigger: {t.get('trigger', '?')} | "
            f"Symptom: {t.get('symptom', '?')} | "
            f"Technique: {t.get('technique', '?')}"
        )
    return "\n".join(lines)