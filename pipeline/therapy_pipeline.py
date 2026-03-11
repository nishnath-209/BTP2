"""
pipeline/therapy_pipeline.py

Full pipeline:
  1. Extract structured info from user message → update KG
  2. Retrieve relevant clinical knowledge via RAG
  3. Query KG for relevant subgraph (not full profile)
  4. Lookup matched clinical triplets
  5. Build prompt with CoT reasoning structure
  6. Generate response via LLM
"""

import json

from rag.retriever import retrieve
from llm.model import generate
from kg.knowledge_graph import KnowledgeGraph
from kg.triplet_store import get_relevant_triplets, triplets_to_text


# Single KG instance 
kg = KnowledgeGraph(storage_path="kg/patient_profiles.json")

# Turn counter per session
_turn = 0


# ------------------------------------------------------------------
# Step 1: Extract structured info from message → update KG
# ------------------------------------------------------------------

def extract_and_update_kg(patient_id, user_message):
    """Use LLM to extract structured fields, then update the KG."""

    extraction_prompt = f"""
Extract the following information from the user message as valid JSON. Only return JSON, no explanation.

Fields:
- quit_goal: "quit smoking", "reduce smoking", or null
- smoking_status: short phrase describing their smoking status, or null
- triggers: list of strings (e.g. ["stress", "morning coffee"])
- past_strategies: list of objects with "strategy" and "outcome" fields
  e.g. [{{"strategy": "nicotine gum", "outcome": "helped"}}]

User message:
{user_message}
"""

    raw = generate(extraction_prompt)

    # Safely extract JSON from model output
    json_text = raw.strip()
    if "{" in json_text and "}" in json_text:
        json_text = json_text[json_text.index("{"):json_text.rindex("}") + 1]

    try:
        extracted = json.loads(json_text)
    except Exception:
        extracted = {}

    global _turn
    _turn += 1
    kg.update(patient_id, extracted, user_message, turn=_turn)

    return extracted


# ------------------------------------------------------------------
# Step 2: Extract context keywords for subgraph + triplet lookup
# ------------------------------------------------------------------

def extract_context_keywords(user_message, extracted):
    """
    Combine triggers from extracted data + simple keywords from message.
    Used to query KG subgraph and triplet store contextually.
    """
    keywords = []

    # From LLM extraction
    for t in extracted.get("triggers", []):
        if isinstance(t, str):
            keywords.append(t)
        elif isinstance(t, dict):
            keywords.append(t.get("trigger", ""))

    # Simple keyword fallback from raw message
    keyword_map = {
        "morning": "morning craving",
        "coffee": "coffee",
        "stress": "stress",
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
# Step 3: Build prompt — RAG + KG subgraph + Triplets + CoT
# ------------------------------------------------------------------

def build_prompt(user_message, rag_context, kg_text, triplets_text):

    rag_text = "\n".join(f"- {c}" for c in rag_context)

    prompt = f"""You are an expert therapist helping patients overcome tobacco addiction.
Use the patient profile, clinical knowledge, and evidence-based techniques below to reason carefully before responding.

{kg_text}

[Clinical Knowledge — Retrieved from therapy sessions]
{rag_text}

{triplets_text}

---

Now reason step by step before giving your response.

[REASONING]
1. What is the patient's main struggle right now?
2. Which triggers are active based on their message and profile?
3. Which technique(s) from the clinical triplets or knowledge best match this patient's situation?
4. What tone is appropriate (empathetic, direct, motivational)?

[RESPONSE]
Write a warm, empathetic, and actionable therapist response based on your reasoning above.

---

Patient message:
{user_message}

Therapist:
"""

    return prompt


# ------------------------------------------------------------------
# Main chat function
# ------------------------------------------------------------------

def therapy_chat(user_message, patient_id="default_patient"):

    print("\n" + "=" * 60)
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

    print("\nSTEP 4: MATCHING CLINICAL TRIPLETS")
    triplets = get_relevant_triplets(context_keywords)
    triplets_text = triplets_to_text(triplets)
    print(triplets_text)

    print("\nSTEP 5: BUILDING PROMPT")
    prompt = build_prompt(user_message, rag_context, kg_text, triplets_text)
    print(prompt)

    print("\nSTEP 6: GENERATING RESPONSE")
    response = generate(prompt)

    return response