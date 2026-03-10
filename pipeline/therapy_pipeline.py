import json

from rag.retriever import retrieve
from llm.model import generate


# --- simple per-session user profile (a lightweight KG) ---
user_profile = {
    "smoking_status": None,
    "triggers": [],
    "quit_goal": None,
    "past_strategies": [],
    "notes": [],
}


def update_user_profile(user_message, profile):
    """Update a simple user profile (KG-like) from the latest user message.

    This version uses the LLM to extract structured fields (JSON) from the message.
    If extraction fails, it falls back to simple keyword heuristics.
    """

    # First try to get structured info via the LLM 
    # more robust than keywords
    extraction_prompt = f"""
Extract the following information from the user message as valid JSON. Only return JSON.

Fields:
- quit_goal: either "quit smoking", "reduce smoking", or null
- triggers: list of triggers (e.g. "stress", "work", "social situations")
- past_strategies: list of strategies mentioned (e.g. "nicotine patch", "deep breathing")
- smoking_status: short phrase if they mention their smoking status, otherwise null

User message:
{user_message}
"""

    raw = generate(extraction_prompt)

    # Attempt to recover JSON from model output
    json_text = raw.strip()
    if "{" in json_text and "}" in json_text:
        json_text = json_text[json_text.index("{"): json_text.rindex("}") + 1]

    try:
        extracted = json.loads(json_text)
    except Exception:
        extracted = None

    if extracted and isinstance(extracted, dict):
        if extracted.get("quit_goal"):
            profile["quit_goal"] = extracted["quit_goal"]

        if extracted.get("smoking_status"):
            profile["smoking_status"] = extracted["smoking_status"]

        for field in ("triggers", "past_strategies"):
            if isinstance(extracted.get(field), list):
                for item in extracted[field]:
                    if item and item not in profile[field]:
                        profile[field].append(item)

        # keep the last few messages as notes
        if user_message.strip() and user_message.strip() not in profile["notes"]:
            profile["notes"].append(user_message.strip())

        return  # extraction succeeded

    # Fallback: keyword heuristics (still useful if LLM extraction fails)
    msg = user_message.lower()

    if "quit" in msg or "stop smoking" in msg:
        profile["quit_goal"] = "quit smoking"

    if "reduce" in msg or "cut down" in msg:
        profile["quit_goal"] = "reduce smoking"

    trigger_keywords = {
        "stress": "stress",
        "work": "work",
        "social": "social situations",
        "coffee": "coffee",
        "alcohol": "alcohol",
        "bored": "boredom",
        "family": "family / home",
    }

    for k, v in trigger_keywords.items():
        if k in msg and v not in profile["triggers"]:
            profile["triggers"].append(v)

    strategy_keywords = {
        "patch": "nicotine patch",
        "gum": "nicotine gum",
        "lozenge": "nicotine lozenge",
        "vape": "vaping",
        "exercise": "exercise",
        "meditat": "meditation",
        "breath": "deep breathing",
        "counsel": "counseling / therapy",
    }

    for k, v in strategy_keywords.items():
        if k in msg and v not in profile["past_strategies"]:
            profile["past_strategies"].append(v)

    if len(msg.split()) > 3:
        profile["notes"].append(user_message.strip())


def profile_to_text(profile):
    """Convert the profile (KG) into a small text block for prompt context."""

    print(profile)

    lines = ["User profile (derived from prior messages):"]

    if profile.get("smoking_status"):
        lines.append(f"- Smoking status: {profile['smoking_status']}")

    if profile.get("quit_goal"):
        lines.append(f"- Quit goal: {profile['quit_goal']}")

    if profile.get("triggers"):
        lines.append(f"- Known triggers: {', '.join(profile['triggers'])}")

    if profile.get("past_strategies"):
        lines.append(f"- Tried strategies: {', '.join(profile['past_strategies'])}")

    if profile.get("notes"):
        lines.append(f"- Notes: {profile['notes'][-3:]}")  # last few notes

    return "\n".join(lines)


def build_prompt(user_message, context, profile):

    context_text = "\n".join(context)
    profile_text = profile_to_text(profile)

    prompt = f"""

You are a therapist helping patients overcome tobacco addiction.

{profile_text}

Relevant therapy advice (from clinical knowledge + evidence):
{context_text}

Perform step-by-step reasoning to select the best advice for this user.

1) Identify the user's main struggle
2) Identify any triggers and risk situations
3) Select 1-2 coping strategies that match the user's preferences
4) Provide a safe, empathetic, actionable suggestion

Patient message:
{user_message}

Therapist response:

"""

    return prompt


def therapy_chat(user_message):
    # Update the user profile (KG) each turn
    update_user_profile(user_message, user_profile)

    context = retrieve(user_message)

    print("\nSTEP 2: RETRIEVED KNOWLEDGE")
    for i, c in enumerate(context):
        print(f"[{i+1}] {c}")

    prompt = build_prompt(user_message, context, user_profile)
    
    print("\nSTEP 3: PROMPT SENT TO LLM")
    print(prompt)

    response = generate(prompt)

    return response