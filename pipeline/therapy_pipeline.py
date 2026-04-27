"""
pipeline/therapy_pipeline.py

Full pipeline:
  1. Extract structured info from user message (incl. is_closing) → update KG
  2. Retrieve relevant clinical knowledge via RAG
  3. Query KG for relevant subgraph
  4. Detect session phase from KG + extracted intent
       Phase 1 — Assessment   : smoking status not yet known
       Phase 2 — Exploration  : triggers not yet understood
       Phase 3 — Motivation   : quit goal / motivation not yet explored
       Phase 4 — Planning     : enough known — can introduce strategies
       Phase 5 — Closing      : patient signals ready to act (only after Planning reached)
  5. Lookup matched clinical triplets (Planning phase only)
  6. Build prompt: phase-specific structure
  7. Generate response via LLM
  8. Log full session turn to JSON
"""

import json
import os
import re
import time
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

SESSION_ID = 1
# SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

_turn = 0
_reached_planning = False  # Gate: closing phase only reachable after Planning
_conversation_history = []

_current_patient_id = None
_current_session_id = None


def _load_session_state(patient_id, session_id):
    """Restore in-memory state from persisted logs for the given patient+session."""
    global _conversation_history, _turn, _reached_planning

    from logger.conversation_history import _load, _find_session
    data = _load()
    session = _find_session(data, patient_id, str(session_id))

    if session:
        _conversation_history = [
            {"role": m["role"], "content": m["message"]}
            for m in session["conversation"]
        ]
    else:
        _conversation_history = []

    log_path = f"logs/{patient_id}_{session_id}.json"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            session_data = json.load(f)
        turns = session_data.get("turns", [])
        _turn = max((t.get("turn", 0) for t in turns), default=0)
        _reached_planning = any(
            t.get("step4_session_phase", {}).get("phase_num", 0) >= 4
            for t in turns
        )
    else:
        _turn = 0
        _reached_planning = False


# Add this function to pipeline/therapy_pipeline.py
# Place it just before the therapy_chat() function
# Also add SESSION_ID to the global declaration inside it

def reset_for_new_patient(new_patient_id: str = "default_patient"):
    """
    Reset all session state so a new conversation can start cleanly.
    Called before each new conversation in evaluation.
    
    Resets:
      - _conversation_history  : clear dialogue history
      - _turn                  : reset turn counter
      - _reached_planning      : reset phase gate
      - SESSION_ID             : new session ID for this conversation
      - KG profile             : delete the patient profile from KG
                                 so it starts fresh (does NOT delete other patients)
    """
    global _conversation_history, _turn, _reached_planning, SESSION_ID

    _conversation_history = []
    _turn                 = 0
    _reached_planning     = False
    SESSION_ID            = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Remove this patient's profile from KG so it starts fresh
    # Other patients' profiles are untouched
    if new_patient_id in kg.profiles:
        del kg.profiles[new_patient_id]
        kg.save()

# ------------------------------------------------------------------
# Step 1: Extract structured info from message → update KG
# ------------------------------------------------------------------

def extract_and_update_kg(patient_id, user_message):
    """
    Use LLM to extract structured fields from the patient message, then update the KG.
    is_closing is extracted here so closure detection uses LLM intent, not phrase matching.
    """

    extraction_prompt = f"""Extract structured information from the patient message.

STRICT RULES:
- Extract ONLY what is explicitly stated in the message.
- Do NOT infer, assume, or interpret beyond the exact words.
- If a field is not explicitly mentioned, return null or empty list.
- Always return ALL fields. Output valid JSON only, no extra text.

SCHEMA (use exactly this format):
{{
  "quit_goal": string or null,
  "motivation_reason": list of strings,
  "smoking_status": string or null,
  "triggers": list of strings,
  "past_strategies": list of {{"strategy": string, "outcome": string}},
  "is_closing": boolean
}}

FIELD DEFINITIONS:

- quit_goal:
  ONLY if explicitly stated. Must be exactly one of:
  "quit <substance>", "reduce <substance>", or null.

- motivation_reason:
  WHY they want to quit — ONLY if explicitly stated.
  Always return a LIST (even if one item).
  Return [] if none mentioned.
  
- smoking_status:
  Substance + duration OR quantity only. Must answer "what" and "how long/how many".
  Substance can be cigarettes, bidis, tobacco, chewing tobacco, alcohol, or any other mentioned.
  Do NOT infer substance if not named. Do NOT extract timing or situations.
  Examples: "cigarettes for 15 years", "15 cigarettes/day", "bidis for 5 years"

- triggers:
  Any situation, emotion, time, or context that increases smoking urge.
  Use MOST SPECIFIC standard label only — do not return both general and specific:
  "stress", "work stress", "after meals", "morning craving",
  "boredom", "alcohol", "social situations", "evening loneliness", "negative mood"
  Otherwise use a short 2-3 word label. [] if none.

- past_strategies:
  ONLY strategies mentioned as already tried.
  Format: [{{"strategy": "...", "outcome": "..."}}]
  Outcome rules:
  - If the patient explicitly states the result → use their words (shortened)
  - If the patient implies failure (e.g., "went back", "didn't work", "couldn't continue") → use "did not help"
  - If outcome is unclear → use "unknown"

- is_closing:
  true ONLY if the patient explicitly indicates they are ending or committing to a plan.
  Examples of true: "Thanks, I'll try this", "Okay, I'll start tomorrow", "That helps, I'm done for now"
  Otherwise false.

IMPORTANT EDGE CASES:
- "I'm not ready to quit" → quit_goal = null
- "I tried before" → goes in past_strategies
- "I smoke when stressed" → triggers = ["stress"], smoking_status = null
- "I've been smoking 10 years" → smoking_status = "cigarettes for 10 years"
- "My kids don't like it and I'm gaining weight" → motivation_reason = ["family pressure", "weight concerns"]
- "I smoke in the morning and when stressed at work" → triggers = ["morning craving", "work stress"]

Patient message:
{user_message}

JSON:
"""

    raw = generate(extraction_prompt,max_new_tokens=300,temperature=0.1)

    # print("Extracted fields for kg updation: " + raw)

    json_text = raw.strip()
    if "{" in json_text and "}" in json_text:
        json_text = json_text[json_text.index("{"):json_text.rindex("}") + 1]

    try:
        extracted = json.loads(json_text)
    except Exception:
        extracted = {}

    msg_lower = user_message.lower()

    # Rule-based fallback for smoking_status if LLM missed it
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

    # Don't overwrite existing smoking_status with a new value — always update
    # Keep history of old value in smoking_status_history
    existing_status = kg.profiles.get(patient_id, {}).get("smoking_status")
    new_status = extracted.get("smoking_status")
    if existing_status and new_status and existing_status == new_status:
        extracted["smoking_status"] = None  # no change, skip update

    global _turn
    _turn += 1
    kg.update(patient_id, extracted, user_message, turn=_turn)

    return extracted


# ------------------------------------------------------------------
# Step 2: Extract context keywords for subgraph + triplet lookup
# ------------------------------------------------------------------

def extract_context_keywords(user_message, extracted):
    """
    Primary: use triggers already extracted by LLM in Step 1.
    Secondary: lightweight keyword map as fallback when LLM returned empty triggers.
    """
    keywords = []

    # Primary — use LLM-extracted triggers directly
    for t in extracted.get("triggers", []):
        trigger_text = t if isinstance(t, str) else t.get("trigger", "")
        if trigger_text:
            keywords.append(trigger_text)

    # Secondary — only run keyword map if LLM gave us nothing
    if not keywords:
        keyword_map = {
            "morning": "morning craving",
            "coffee": "morning craving",
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
# Step 4: Detect session phase
# ------------------------------------------------------------------

def detect_session_phase(subgraph: dict, extracted: dict) -> tuple:
    """
    Determine session phase based on KG state and LLM-extracted intent.

    Phase 5 (Closing) is checked first — only reachable after Planning was reached.
    is_closing comes from Step 1 extraction (LLM intent), not phrase matching.

    Returns (phase_number, phase_name, phase_instruction)
    """
    global _reached_planning

    has_status     = bool(subgraph.get("smoking_status"))
    has_triggers   = bool(subgraph.get("relevant_triggers"))
    has_motivation = bool(subgraph.get("motivation_reason"))

    # Phase 5 — Closing
    # Gated behind _reached_planning so "thank you" in early phases doesn't fire this
    if _reached_planning and extracted.get("is_closing") is True:
        return (5, "Closing", (
            "The patient has signaled they are ready to act or is wrapping up. "
            "Your ONLY job is to give ONE warm closing sentence. "
            "Do NOT ask any question. "
            "Do NOT suggest anything new. "
            "Do NOT summarize what was discussed. "
            "Just acknowledge their readiness warmly and wish them well. "
            "Example: 'That sounds like a solid plan — take it one step at a time.' "
            "One sentence only. Nothing more."
        ))

    # Phase 1 — Assessment
    if not has_status:
        return (1, "Assessment", (
            "Focus on gathering basic facts: how long they have been smoking, "
            "how much they smoke per day, and what substance. "
            "Do NOT ask about triggers or suggest strategies yet. "
            "Ask only one simple factual question."
        ))

    # Phase 2 — Exploration
    if not has_triggers:
        return (2, "Exploration", (
            "You know the basics. Now gently explore what situations or emotions "
            "make them smoke more — triggers like stress, meals, social settings, boredom. "
            "Do NOT suggest any strategies yet. Just understand their situation."
        ))

    # Phase 3 — Motivation
    if not has_motivation:
        return (3, "Motivation", (
            "You understand their triggers. Now explore their motivation: "
            "why do they want to quit — is it health, family, cost, doctor advice? "
            "Have they tried before and what happened? "
            "Be empathetic. Do NOT push strategies yet."
        ))

    # Phase 4 — Planning — only after enough turns to properly assess the patient
    if _turn < 3:
        return (3, "Motivation", (
            "You understand their triggers. Deepen your understanding of their motivation "
            "and readiness to quit before moving to strategies. "
            "Explore ambivalence, past attempts, or what a smoke-free life would mean to them. "
            "Do NOT suggest strategies yet."
        ))

    # Phase 4 — Planning
    # Set the flag so Closing becomes reachable from the next turn onward
    _reached_planning = True
    return (4, "Planning", (
        "You have enough understanding of this patient. "
        "Introduce ONE relevant coping strategy matched to their specific triggers and past attempts. "
        "Do NOT just name the strategy — briefly explain how to apply it in practice using the clinical knowledge (when to use it, what to do like a plan). "
        "Then check if the patient is open to trying it. "
        "Keep it simple, specific, and actionable."
    ))


# ------------------------------------------------------------------
# Step 6a: Build closing prompt (Phase 5 only)
# ------------------------------------------------------------------

def build_closing_prompt(user_message, history_block, phase_instruction):
    """
    Minimal prompt for closing phase.
    No KG, no RAG, no reasoning block — just tone context and one tight instruction.
    """
    history_lines = history_block.strip().split("\n")
    last_exchange = "\n".join(history_lines[-2:]) if len(history_lines) >= 2 else history_block

    return f"""You are a warm, empathetic therapist.

{phase_instruction}

{last_exchange}

Patient: {user_message}

Therapist:"""


# ------------------------------------------------------------------
# Step 6b: Build main prompt (Phases 1–4)
# ------------------------------------------------------------------

def build_prompt(user_message, rag_context, kg_text, triplets_text, history_block,
                 phase_num, phase_name, phase_instruction):
    """
    Full prompt for Phases 1-4.
    Clinical section (RAG + triplets) only injected in Planning phase (4).
    """
    rag_text = "\n".join(f"- {c}" for c in rag_context)

    clinical_section = ""
    if rag_text.strip():
        clinical_section += f"[Clinical Knowledge]\n{rag_text}\n\n"
    if triplets_text.strip():
        clinical_section += f"{triplets_text}\n\n"

    return f"""You are a warm, empathetic therapist helping patients overcome tobacco addiction.

STYLE RULES — follow these strictly:
- Keep your response to 1-3 sentences maximum
- Ask only ONE question per response
- Do NOT list techniques, bullet points, or dump information
- First acknowledge what the patient said, then ask ONE follow-up question
- Only suggest a technique when you are in the Planning phase
- Use simple, conversational language — not clinical or formal

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
3. What single question or response fits this phase?

[RESPONSE]
Therapist:"""

def build_prompt_no_phase(user_message, rag_context, kg_text, history_block):
    """
    Flat therapist prompt with no phase system.
    The LLM still acts as a full therapist with all context —
    just without structured phase-by-phase guidance.
    Used for the no_phase ablation variant.
    """
    rag_text = "\n".join(f"- {c}" for c in rag_context)

    clinical_section = ""
    if rag_text.strip():
        clinical_section = f"[Clinical Knowledge]\n{rag_text}\n\n"

    return f"""You are a warm, empathetic therapist helping patients overcome tobacco addiction.

STYLE RULES — follow these strictly:
- Keep your response to 1-3 sentences maximum
- Ask only ONE question per response
- Do NOT use bullet points or dump multiple pieces of information at once
- First acknowledge what the patient said, then ask ONE follow-up question
- Use simple, conversational language — not clinical or formal

APPROACH:
- Be empathetic and helpful throughout the conversation
- Use your clinical judgment to decide what the patient needs right now

{kg_text}

{clinical_section}{history_block}

---

Current patient message:
{user_message}

---

[REASONING]
1. What is the patient saying?
2. What is the most appropriate single response right now?

[RESPONSE]
Therapist:"""

# ------------------------------------------------------------------
# Main chat function
# ------------------------------------------------------------------

def therapy_chat(user_message, patient_id="default_patient", session_id=None):
    global _conversation_history, _current_patient_id, _current_session_id, SESSION_ID

    if session_id is None:
        session_id = SESSION_ID

    # Resume or switch session when patient_id+session_id changes
    # if patient_id != _current_patient_id or str(session_id) != str(_current_session_id):
    #     _current_patient_id = patient_id
    #     _current_session_id = session_id
    #     SESSION_ID = session_id
    #     _load_session_state(patient_id, session_id)
    #     if _conversation_history:
    #         print(f"[Session] Resumed session {session_id} for {patient_id} "
    #               f"({len(_conversation_history)//2} prior turns)")
    #     else:
    #         print(f"[Session] New session {session_id} for {patient_id}")

    # print("\n" + "=" * 60)

    # Add patient message to history
    _conversation_history.append({"role": "patient", "content": user_message})

    # print("STEP 1: EXTRACTING INFO → UPDATING KG")
    extracted = extract_and_update_kg(patient_id, user_message)
    # print("Extracted:", json.dumps(extracted, indent=2))

    time.sleep(3) 
    
    # print("\nSTEP 2: RETRIEVING CLINICAL KNOWLEDGE (RAG)")
    rag_context = retrieve(user_message)
    # for i, c in enumerate(rag_context):
        # print(f"  [{i+1}] {c[:100]}...")
    # rag_context = []

    # print("\nSTEP 3: QUERYING KG SUBGRAPH")
    context_keywords = extract_context_keywords(user_message, extracted)
    # print("Context keywords:", context_keywords)
    subgraph = kg.get_subgraph(patient_id, context_keywords)
    kg_text = kg.subgraph_to_text(subgraph)
    # print(kg_text)

    # print("\nSTEP 4: DETECTING SESSION PHASE")
    phase_num, phase_name, phase_instruction = detect_session_phase(subgraph, extracted)
    # print(f"  → Phase {phase_num}: {phase_name}")

    # print("\nSTEP 5: MATCHING CLINICAL TRIPLETS")
    # triplets = get_relevant_triplets(context_keywords)
    # triplets_text = triplets_to_text(triplets)
    # print(triplets_text)
    triplets = []
    triplets_text = ""

    # print("\nSTEP 6: BUILDING PROMPT")
    history_block = build_history_block()

    # if phase_num == 5:
    #     prompt = build_closing_prompt(user_message, history_block, phase_instruction)
    # else:
    #     prompt = build_prompt(
    #         user_message, rag_context, kg_text, triplets_text, history_block,
    #         phase_num, phase_name, phase_instruction
    #     )

    prompt = build_prompt_no_phase(
        user_message, rag_context, kg_text, history_block
    )


    # print(prompt)

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