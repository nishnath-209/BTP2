"""
pipeline/therapy_pipeline.py

Full pipeline:
  1. Extract structured info from user message → update KG
  2. Retrieve relevant clinical knowledge via RAG
  3. Query KG for relevant subgraph
  4. Detect session phase from KG (Assessment / Exploration / Motivation / Planning)
  5. Lookup matched clinical triplets (only used in Planning phase)
  6. Build prompt: style rules + phase instruction + KG + clinical + history + CoT
  7. Generate response via LLM
  8. Log full session turn to JSON
"""

import json
import re
from datetime import datetime

from rag.retriever import retrieve
from llm.model import generate, MODEL_NAME
from kg.knowledge_graph import KnowledgeGraph
from kg.triplet_store import get_relevant_triplets, triplets_to_text
from logger.session_logger import SessionLogger
from logger.conversation_history import add_turn



# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

# Number of recent turns to keep in sliding window (each turn = 1 patient + 1 therapist message)
HISTORY_WINDOW = 6  # 3 exchanges — change here to adjust

# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

kg = KnowledgeGraph(storage_path="kg/patient_profiles.json")
logger = SessionLogger(log_dir="logs")
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

_turn = 0

# Conversation history — list of {"role": "patient"/"therapist", "content": "..."}
# This is the full history; we slice it per prompt using HISTORY_WINDOW
_conversation_history = []


# ------------------------------------------------------------------
# Step 1: Extract structured info from message → update KG
# ------------------------------------------------------------------

def extract_and_update_kg(patient_id, user_message):
    """Use LLM to extract structured fields, then update the KG."""

    extraction_prompt = f"""Extract ONLY information that is explicitly and directly stated in the patient message below.
Do NOT infer, assume, or add anything not clearly mentioned.
If a field is not explicitly mentioned, return null or empty list.
Return valid JSON only. No explanation, no extra text.

Fields:
- quit_goal: ONLY if patient explicitly says they want to quit or reduce. One of: "quit smoking", "reduce smoking", or null
- motivation_reason: WHY they want to quit — only if explicitly stated. Examples: "health issues", "family pressure", "kids asked me to stop", "chest pain", "doctor advised". null if not mentioned.
- smoking_status: Extract ONLY substance + duration OR quantity if explicitly mentioned. Must answer "what" and "how long/how many". Do NOT extract when/why they smoke — those are triggers.
  Examples: "bidis for 5 years", "10 cigarettes/day", "cigarettes for 10 years". If only timing/context is mentioned (e.g. "after food", "when stressed"), return null.
- triggers: Short label for any situation/emotion the patient says makes them smoke or increases urge.
  Extract the trigger concept, not the full sentence. Examples of trigger labels: "social pressure", "work stress", "after meals", "boredom", "alcohol", "evenings"
  Empty list [] if none explicitly mentioned.
- past_strategies: ONLY strategies patient explicitly says they have tried. Empty list [] if none mentioned.
  Format: [{{"strategy": "...", "outcome": "..."}}]

Patient message:
{user_message}

JSON:
{{"""

    raw = generate(extraction_prompt)

    print("Extracted fields for kg updation : " + raw)

    json_text = raw.strip()
    if "{" in json_text and "}" in json_text:
        json_text = json_text[json_text.index("{"):json_text.rindex("}") + 1]

    try:
        extracted = json.loads(json_text)
    except Exception:
        extracted = {}

    msg_lower = user_message.lower()

    # Rule-based fallback for smoking_status
    if not extracted.get("smoking_status"):
        substances = ["bidi", "bidis", "cigarette", "cigarettes", "tobacco"]
        duration_match = re.search(r"(\d+\s+(?:year|month|week|day)s?)", msg_lower)
        quantity_match = re.search(r"(\d+\s+(?:bidi|bidis|cigarette|cigarettes)(?:\s*(?:a|per)\s*(?:day|week))?)", msg_lower)
        for substance in substances:
            if substance in msg_lower:
                if quantity_match:
                    extracted["smoking_status"] = quantity_match.group(1).strip()
                elif duration_match:
                    extracted["smoking_status"] = f"{substance} for {duration_match.group(1).strip()}"
                break

    # Don't overwrite existing smoking_status with a weaker value
    existing_status = kg.profiles.get(patient_id, {}).get("smoking_status")
    new_status = extracted.get("smoking_status")
    if existing_status and new_status and len(existing_status) >= len(new_status):
        extracted["smoking_status"] = None

    # Validate triggers and strategies against actual message
    msg_lower = user_message.lower()

    validated_triggers = []
    for t in extracted.get("triggers", []):
        trigger_text = t if isinstance(t, str) else t.get("trigger", "")
        if any(word in msg_lower for word in trigger_text.lower().split()):
            validated_triggers.append(trigger_text)
    extracted["triggers"] = validated_triggers

    validated_strategies = []
    for s in extracted.get("past_strategies", []):
        strategy_text = s if isinstance(s, str) else s.get("strategy", "")
        if any(word in msg_lower for word in strategy_text.lower().split()):
            validated_strategies.append(s)
    extracted["past_strategies"] = validated_strategies

    global _turn
    _turn += 1
    kg.update(patient_id, extracted, user_message, turn=_turn)

    return extracted


# ------------------------------------------------------------------
# Step 2: Extract context keywords for subgraph + triplet lookup
# ------------------------------------------------------------------

def extract_context_keywords(user_message, extracted):
    keywords = []

    for t in extracted.get("triggers", []):
        if isinstance(t, str):
            keywords.append(t)
        elif isinstance(t, dict):
            keywords.append(t.get("trigger", ""))

    keyword_map = {
        "morning": "morning craving",
        "coffee": "coffee",
        "stress": "stress",
        "stressed": "stress",
        "work": "work stress",
        "bored": "boredom",
        "alcohol": "alcohol",
        "evening": "evening loneliness",
        "meal": "after meals",
        "food": "after meals",
        "social": "social situations",
        "friend": "social situations",
        "craving": "strong craving (general)",
        "urge": "strong craving (general)",
        "mood": "negative mood",
        "sad": "negative mood",
        "relapse": "relapse after quit attempt",
    }

    msg_lower = user_message.lower()
    for word, keyword in keyword_map.items():
        if word in msg_lower and keyword not in keywords:
            keywords.append(keyword)

    return keywords if keywords else ["strong craving (general)"]


# ------------------------------------------------------------------
# Step 3: Build conversation history block (sliding window)
# ------------------------------------------------------------------

def build_history_block(window=HISTORY_WINDOW):
    """
    Returns the last `window` messages from conversation history
    formatted as a readable dialogue block for the prompt.
    Full history is preserved in _conversation_history for logging.
    """
    recent = _conversation_history[-window:]
    if not recent:
        return ""

    lines = ["[Recent Conversation]"]
    for msg in recent:
        role = "Patient" if msg["role"] == "patient" else "Therapist"
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Step 4: Detect session phase from KG
# ------------------------------------------------------------------

def detect_session_phase(subgraph: dict) -> tuple:
    """
    Determine which phase the session is in based on what the KG has collected so far.

    Phase 1 — Assessment   : basic facts not yet known (smoking status, duration)
    Phase 2 — Exploration  : triggers not yet understood
    Phase 3 — Motivation   : quit goal / past attempts not yet explored
    Phase 4 — Planning     : enough known — can introduce strategies

    Returns (phase_number, phase_name, phase_instruction)
    """

    has_status     = bool(subgraph.get("smoking_status"))
    has_triggers   = bool(subgraph.get("relevant_triggers"))
    has_motivation = bool(subgraph.get("motivation_reason"))
    has_strategy   = bool(subgraph.get("relevant_strategies"))

    if not has_status:
        return (1, "Assessment", (
            "You are in the ASSESSMENT phase. "
            "Focus on gathering basic facts: how long they have been smoking, "
            "how much they smoke per day, and what substance. "
            "Do NOT ask about triggers or suggest strategies yet. "
            "Ask only one simple factual question."
        ))

    if not has_triggers:
        return (2, "Exploration", (
            "You are in the EXPLORATION phase. "
            "You know the basics. Now gently explore what situations or emotions "
            "make them smoke more — triggers like stress, meals, social settings, boredom. "
            "Do NOT suggest any strategies yet. Just understand their situation."
        ))

    if not has_motivation:
        return (3, "Motivation", (
            "You are in the MOTIVATION phase. "
            "You understand their triggers. Now explore their motivation: "
            "why do they want to quit — is it health, family, cost, doctor advice? "
            "Have they tried before and what happened? "
            "Be empathetic. Do NOT push strategies yet."
        ))

    return (4, "Planning", (
            "You are in the PLANNING phase. "
            "You have enough understanding of this patient. "
            "You can now gently introduce ONE relevant coping strategy or technique "
            "based on their specific triggers and past attempts. "
            "Keep it simple — one idea at a time, check if they are open to it."
    ))


def build_prompt(user_message, rag_context, kg_text, triplets_text, history_block,
                 phase_num, phase_name, phase_instruction):

    rag_text = "\n".join(f"- {c}" for c in rag_context)

    # Only inject clinical knowledge in planning phase 
    clinical_section = ""
    if phase_num >= 4:
        if rag_text.strip():
            clinical_section += f"[Clinical Knowledge]\n{rag_text}\n\n"
        if triplets_text.strip():
            clinical_section += f"{triplets_text}\n\n"

    prompt = f"""You are a warm, empathetic therapist helping patients overcome tobacco addiction.

STYLE RULES — follow these strictly:
- Keep your response to 1-3 sentences maximum
- Ask only ONE question per response
- Do NOT list techniques, bullet points, or dump information
- First acknowledge what the patient said, then ask ONE follow-up question
- Only suggest a technique when you are in the Planning phase
- Use simple, conversational language — not clinical or formal
- IMPORTANT: If the patient says they will try something, agrees to a plan, or signals they are ready to act — you MUST stop asking questions. Give ONE short warm closing sentence like "That's a good start, try it and see how it feels." Do NOT suggest anything new after this.
[Session Phase: {phase_num} — {phase_name}]
{phase_instruction}

{kg_text}

{clinical_section}{history_block}

---

Current patient message:
{user_message}

---

[REASONING]
1. Phase I am in: {phase_name}
2. What does the phase instruction tell me to focus on?
3. Is the patient signaling they are done or ready to act? If yes → closing only, no question.
4. What single question or response fits this phase?

[RESPONSE]
Therapist:"""

    return prompt


# ------------------------------------------------------------------
# Main chat function
# ------------------------------------------------------------------

CLOSURE_PHRASES = [
    "i will try", "i'll try", "i will do", "i'll do", "i will start",
    "okay i will", "ok i will", "sounds good", "thank you", "thanks",
    "i'll give it a try", "let me try", "i can try", "i am ready",
    "i will follow", "okay thanks", "ok thanks", "that makes sense",
    "i understand", "got it", "sure i will"
]

def is_closing_message(message: str) -> bool:
    msg = message.lower().strip()
    return any(phrase in msg for phrase in CLOSURE_PHRASES)

CLOSING_RESPONSES = [
    "That's a great first step. Try it and see how it feels — we can talk more in our next session.",
    "Good. Take it one day at a time, and don't be too hard on yourself if it's hard at first.",
    "That's the spirit. Try it out, and remember every small step counts.",
]

def therapy_chat(user_message, patient_id="default_patient"):
    global _conversation_history

    print("\n" + "=" * 60)

    # Add patient message to history
    _conversation_history.append({"role": "patient", "content": user_message})

    if is_closing_message(user_message):
        import random
        closing = random.choice(CLOSING_RESPONSES)
        _conversation_history.append({"role": "therapist", "content": closing})
        print(f"[Closure detected] Returning closing response.")
        return closing

    print("STEP 1: EXTRACTING INFO → UPDATING KG")
    extracted = extract_and_update_kg(patient_id, user_message)
    print("Extracted:", json.dumps(extracted, indent=2))

    print("\nSTEP 2: RETRIEVING CLINICAL KNOWLEDGE (RAG)")
    rag_context = retrieve(user_message)
    for i, c in enumerate(rag_context):
        print(f"  [{i+1}] {c[:100]}...")
    
    print("\nSTEP 3: QUERYING KG SUBGRAPH")
    context_keywords = extract_context_keywords(user_message, extracted)
    print("Context keywords:", context_keywords)
    subgraph = kg.get_subgraph(patient_id, context_keywords)
    kg_text = kg.subgraph_to_text(subgraph)
    print(kg_text)

    print("\nSTEP 4: DETECTING SESSION PHASE")
    phase_num, phase_name, phase_instruction = detect_session_phase(subgraph)
    print(f"  → Phase {phase_num}: {phase_name}")

    print("\nSTEP 5: MATCHING CLINICAL TRIPLETS")
    triplets = get_relevant_triplets(context_keywords)
    triplets_text = triplets_to_text(triplets)
    print(triplets_text)

    print("\nSTEP 6: BUILDING PROMPT")
    history_block = build_history_block()
    prompt = build_prompt(
        user_message, rag_context, kg_text, triplets_text, history_block,
        phase_num, phase_name, phase_instruction
    )
    print(prompt)

    print("\nSTEP 7: GENERATING RESPONSE")
    full_output = generate(prompt, max_new_tokens=200)
    print(full_output)

    # Add therapist response to history
    _conversation_history.append({"role": "therapist", "content": full_output.strip()})

    print(f"\n[History] Total turns in memory: {len(_conversation_history)}, "
          f"Window used in prompt: {min(len(_conversation_history), HISTORY_WINDOW)}")

    # ------------------------------------------------------------------
    # Step 8: Log the full turn
    # ------------------------------------------------------------------
    logger.log_turn(
        patient_id=patient_id,
        session_id=SESSION_ID,
        turn=_turn,
        model_name=MODEL_NAME,
        user_message=user_message,
        extracted=extracted,
        context_keywords=context_keywords,
        kg_subgraph=subgraph,
        rag_chunks=rag_context,
        triplets=triplets,
        session_phase={
            "phase_num": phase_num,
            "phase_name": phase_name,
            "phase_instruction": phase_instruction,
        },
        prompt=prompt,
        cot_reasoning=full_output,
        final_response=full_output,
    )

    add_turn(patient_id, SESSION_ID, user_message, full_output.strip())

    return full_output.strip()