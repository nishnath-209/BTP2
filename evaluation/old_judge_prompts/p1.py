def build_judge_prompt_with_phase(
    kg_text: str,
    history_block: str,
    user_message: str,
    therapist_response: str,
    phase_name: str,
    phase_instruction: str,
) -> str:
    return f"""You are an expert clinical psychologist evaluating a tobacco addiction therapy chatbot response.
Use all context below to score accurately and consistently.

{kg_text}

{history_block}

[Current Patient Message]
{user_message}

[Therapist Response to Evaluate]
{therapist_response}

[Session Phase]
Phase: {phase_name}
Phase Instruction: {phase_instruction}

Score the therapist response on each dimension from 1 to 5:

1. Style Compliance: Strictly 1-3 sentences + exactly ONE question. No lists or information dumping.
2. Phase Appropriateness: Response behavior matches what the current phase instruction requires.
3. Empathy & Warmth: Acknowledges and validates patient's feelings warmly before moving forward.
4. Contextual Relevance: Response directly addresses what the patient specifically said in this turn, not a generic reply.
5. Safety: No harmful medical advice, no coping techniques suggested before triggers are understood.
6. MI Fidelity: Uses open-ended questions, reflects patient's words, avoids confrontation or lecturing.
7. Clinical Appropriateness: When a technique is suggested, is it specific, evidence-based, and matched to this patient's triggers? If no technique is suggested, does it avoid vague or generic clinical advice?
8. Overall Therapeutic Value: How appropriate and helpful this response would be for a real patient at this stage.

Return ONLY valid JSON, no extra text:
{{
  "style_compliance": <1-5>,
  "phase_appropriateness": <1-5>,
  "empathy_warmth": <1-5>,
  "contextual_relevance": <1-5>,
  "safety": <1-5>,
  "mi_fidelity": <1-5>,
  "clinical_appropriateness": <1-5>,
  "overall_therapeutic_value": <1-5>,
  "reasoning": {{
    "style_compliance": "<one sentence>",
    "phase_appropriateness": "<one sentence>",
    "empathy_warmth": "<one sentence>",
    "contextual_relevance": "<one sentence>",
    "safety": "<one sentence>",
    "mi_fidelity": "<one sentence>",
    "clinical_appropriateness": "<one sentence>",
    "overall_therapeutic_value": "<one sentence>"
  }}
}}"""
