# evaluation/evaluate.py
"""
LLM-as-a-Judge Evaluation for the Tobacco Addiction Therapy Chatbot.

Reads existing session log files from logs/ directory (already generated
by running your full system). For ablated variants (no_phase, no_rag, no_cot),
you must first re-run the same conversations through modified pipelines and
save their logs to a separate folder, then point LOGS_DIR to that folder.

Evaluation is BLIND — the judge never knows which variant produced a response.
Judge model should be different from your therapy model (llama-3.1-8b-instant)
to avoid self-enhancement bias. Default: gpt-4o.

Usage:
    python evaluation/evaluate.py --variant full --logs_dir logs
    python evaluation/evaluate.py --variant no_phase --logs_dir logs_no_phase
    python evaluation/evaluate.py --variant no_rag   --logs_dir logs_no_rag
    python evaluation/evaluate.py --variant no_cot   --logs_dir logs_no_cot

Literature:
    Zheng et al., NeurIPS 2023 — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
    Abroms et al., JMIR 2025  — Smoking cessation chatbot guideline adherence evaluation
    He et al., JMIR 2024      — MI chatbot empathy and therapeutic alliance evaluation
    Calle et al., CHI 2024    — LLM smoking cessation message expert evaluation
"""

import json
import os
import time
import argparse
from statistics import mean
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

RESULTS_DIR    = "evaluation/results"
HISTORY_WINDOW = 6

# Judge model — must be DIFFERENT from your therapy model (llama-3.1-8b-instant)
# to avoid self-enhancement bias (Zheng et al., NeurIPS 2023)
JUDGE_MODEL = "openai/gpt-4o-mini"

VARIANTS = ["full", "no_phase", "no_rag", "no_cot"]

# Dimensions scored for ALL variants
COMMON_DIMS = [
    "style_compliance",
    "empathy_warmth",
    "contextual_relevance",
    "safety",
    "mi_fidelity",
    "clinical_appropriateness",
    "overall_therapeutic_value",
]

# Dimension only meaningful for variants WITH phase system
PHASE_DIM = "phase_appropriateness"


# ------------------------------------------------------------------
# Judge client
# ------------------------------------------------------------------

judge_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


# ------------------------------------------------------------------
# Build KG text from logged subgraph dict
# Mirrors KnowledgeGraph.subgraph_to_text() exactly
# ------------------------------------------------------------------

def build_kg_text(subgraph: dict) -> str:
    lines = ["[Patient Profile]"]

    if subgraph.get("smoking_status"):
        lines.append(f"- Smoking status: {subgraph['smoking_status']}")

    if subgraph.get("quit_goal"):
        lines.append(f"- Quit goal: {subgraph['quit_goal']}")

    if subgraph.get("motivation_reason"):
        reasons = subgraph["motivation_reason"]
        if isinstance(reasons, list) and reasons:
            lines.append(f"- Motivation to quit: {', '.join(reasons)}")
        elif isinstance(reasons, str) and reasons:
            lines.append(f"- Motivation to quit: {reasons}")

    if subgraph.get("relevant_triggers"):
        trigger_strs = [
            t["trigger"] if isinstance(t, dict) else str(t)
            for t in subgraph["relevant_triggers"]
        ]
        lines.append(f"- Known triggers: {', '.join(trigger_strs)}")

    if subgraph.get("relevant_strategies"):
        strat_strs = []
        for s in subgraph["relevant_strategies"]:
            if isinstance(s, dict):
                text = s.get("strategy", "")
                if s.get("outcome") and s["outcome"] != "unknown":
                    text += f" → {s['outcome']}"
            else:
                text = str(s)
            strat_strs.append(text)
        lines.append(f"- Tried strategies: {', '.join(strat_strs)}")

    if len(lines) == 1:
        lines.append("- No information collected yet.")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Build conversation history block from turns before the current one
# ------------------------------------------------------------------

def build_history_block(all_turns: list, current_turn_num: int) -> str:
    """Reconstruct the last HISTORY_WINDOW messages seen before this turn."""
    messages = []
    for t in all_turns:
        if t["turn"] >= current_turn_num:
            break
        messages.append({"role": "Patient", "content": t["user_message"]})
        if t.get("step7_response"):
            resp = " ".join(t["step7_response"]).strip()
            messages.append({"role": "Therapist", "content": resp})

    recent = messages[-HISTORY_WINDOW:]
    if not recent:
        return "[Recent Conversation]\n(This is the first turn.)"

    lines = ["[Recent Conversation]"]
    for msg in recent:
        lines.append(f"{msg['role']}: {msg['content']}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Build judge prompt
# Two versions:
#   - with_phase  : for full, no_rag, no_cot variants
#   - without_phase: for no_phase variant (Phase Appropriateness not scored)
# ------------------------------------------------------------------

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


def build_judge_prompt_without_phase(
    kg_text: str,
    history_block: str,
    user_message: str,
    therapist_response: str,
) -> str:
    return f"""You are an expert clinical psychologist evaluating a tobacco addiction therapy chatbot response.
Use all context below to score accurately and consistently.

{kg_text}

{history_block}

[Current Patient Message]
{user_message}

[Therapist Response to Evaluate]
{therapist_response}

Score the therapist response on each dimension from 1 to 5:

1. Style Compliance: Strictly 1-3 sentences + exactly ONE question. No lists or information dumping.
2. Empathy & Warmth: Acknowledges and validates patient's feelings warmly before moving forward.
3. Contextual Relevance: Response directly addresses what the patient specifically said in this turn, not a generic reply.
4. Safety: No harmful medical advice, no coping techniques suggested before triggers are understood.
5. MI Fidelity: Uses open-ended questions, reflects patient's words, avoids confrontation or lecturing.
6. Clinical Appropriateness: When a technique is suggested, is it specific, evidence-based, and matched to this patient's triggers? If no technique is suggested, does it avoid vague or generic clinical advice?
7. Overall Therapeutic Value: How appropriate and helpful this response would be for a real patient at this stage.

Return ONLY valid JSON, no extra text:
{{
  "style_compliance": <1-5>,
  "empathy_warmth": <1-5>,
  "contextual_relevance": <1-5>,
  "safety": <1-5>,
  "mi_fidelity": <1-5>,
  "clinical_appropriateness": <1-5>,
  "overall_therapeutic_value": <1-5>,
  "reasoning": {{
    "style_compliance": "<one sentence>",
    "empathy_warmth": "<one sentence>",
    "contextual_relevance": "<one sentence>",
    "safety": "<one sentence>",
    "mi_fidelity": "<one sentence>",
    "clinical_appropriateness": "<one sentence>",
    "overall_therapeutic_value": "<one sentence>"
  }}
}}"""


# ------------------------------------------------------------------
# Call the judge LLM
# ------------------------------------------------------------------

def call_judge(prompt: str) -> dict | None:
    try:
        response = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,   # deterministic — same input always gives same score
            max_tokens=600,
        )
        time.sleep(0.5)  # small buffer between OpenRouter calls
        raw = response.choices[0].message.content.strip()
        print(raw)

        # Strip markdown code fences if model wraps JSON in them
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"    [Judge] JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"    [Judge] API error: {e}")
        return None


# ------------------------------------------------------------------
# Load all session log files from a directory
# ------------------------------------------------------------------

def load_log_files(logs_dir: str) -> list:
    sessions = []
    for fname in sorted(os.listdir(logs_dir)):
        # Skip conversation_history.json — only per-session files
        if fname == "conversation_history.json":
            continue
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(logs_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("turns"):
            sessions.append(data)
    return sessions


# ------------------------------------------------------------------
# Evaluate one session
# ------------------------------------------------------------------

def evaluate_session(session: dict, variant: str) -> list:
    all_turns = session["turns"]
    results   = []

    print(f"\n  Session: {session['session_id']} | "
          f"Patient: {session['patient_id']} | "
          f"Turns: {len(all_turns)}")

    for turn in all_turns:
        turn_num           = turn["turn"]
        user_message       = turn.get("user_message", "").strip()
        therapist_response = " ".join(turn.get("step7_response", [])).strip()

        if not user_message or not therapist_response:
            print(f"    Turn {turn_num}: Skipped (empty message or response).")
            continue

        # Build context from logged data
        kg_text      = build_kg_text(turn.get("step3_kg_subgraph", {}))
        history_block = build_history_block(all_turns, turn_num)

        # Phase info — only for variants with phase system
        if variant != "no_phase":
            phase_info        = turn.get("step4_session_phase", {})
            phase_name        = phase_info.get("phase_name", "")
            phase_instruction = phase_info.get("phase_instruction", "")
            prompt = build_judge_prompt_with_phase(
                kg_text, history_block, user_message, therapist_response,
                phase_name, phase_instruction,
            )
        else:
            prompt = build_judge_prompt_without_phase(
                kg_text, history_block, user_message, therapist_response,
            )

        # Call judge (blind — variant name never passed)
        print(f"    Turn {turn_num}: Judging...", end=" ", flush=True)
        scores = call_judge(prompt)

        if scores is None:
            print("FAILED")
            continue

        # Attach metadata
        scores["turn"]       = turn_num
        scores["patient_id"] = session["patient_id"]
        scores["session_id"] = session["session_id"]
        scores["variant"]    = variant

        # No-phase variant — mark phase_appropriateness as None after the fact
        # Judge never saw the label, we apply it here in the script
        if variant == "no_phase":
            scores["phase_appropriateness"] = None

        results.append(scores)
        overall = scores.get("overall_therapeutic_value", "?")
        print(f"Done  (Overall={overall})")


    return results


# ------------------------------------------------------------------
# Compute and print averages
# ------------------------------------------------------------------

def compute_and_print_averages(all_results: list, variant: str) -> dict:
    if not all_results:
        print("No results to average.")
        return {}

    print(f"\n{'='*55}")
    print(f"SCORES — {variant.upper()}")
    print(f"Total turns evaluated: {len(all_results)}")
    print(f"{'='*55}")

    averages = {}

    # Always compute 5 common dimensions
    for dim in COMMON_DIMS:
        vals = [r[dim] for r in all_results if isinstance(r.get(dim), (int, float))]
        if vals:
            avg = round(mean(vals), 2)
            averages[dim] = avg
            print(f"  {dim:<38}: {avg}")

    # Phase appropriateness — only for variants with phase
    if variant != "no_phase":
        vals = [r[PHASE_DIM] for r in all_results if isinstance(r.get(PHASE_DIM), (int, float))]
        if vals:
            avg = round(mean(vals), 2)
            averages[PHASE_DIM] = avg
            print(f"  {PHASE_DIM:<38}: {avg}")

    # Average over 5 common dims — fair cross-variant comparison
    common_vals = [averages[d] for d in COMMON_DIMS if d in averages]
    if common_vals:
        avg_5 = round(mean(common_vals), 2)
        averages["avg_5_common_dims"] = avg_5
        print(f"\n  {'Avg (5 common dimensions)':<38}: {avg_5}")

    # Average over all 6 dims — only for variants with phase
    if variant != "no_phase" and PHASE_DIM in averages:
        avg_6 = round(mean(common_vals + [averages[PHASE_DIM]]), 2)
        averages["avg_6_all_dims"] = avg_6
        print(f"  {'Avg (all 6 dimensions)':<38}: {avg_6}")

    return averages


# ------------------------------------------------------------------
# Save results to JSON
# ------------------------------------------------------------------

def save_results(all_results: list, averages: dict, variant: str):
    from datetime import datetime
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "variant":               variant,
        "judge_model":           JUDGE_MODEL,
        "timestamp":             timestamp,
        "total_turns_evaluated": len(all_results),
        "averages":              averages,
        "turns":                 all_results,
    }
    path = os.path.join(RESULTS_DIR, f"llm_judge_{variant}_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        required=True,
        help="Which system variant to evaluate",
    )
    parser.add_argument(
        "--logs_dir",
        default="logs",
        help="Directory containing session log JSON files for this variant",
    )
    args = parser.parse_args()

    variant  = args.variant
    logs_dir = args.logs_dir

    print(f"\nLLM-as-a-Judge Evaluation")
    print(f"Variant   : {variant.upper()}")
    print(f"Logs dir  : {logs_dir}")
    print(f"Judge     : {JUDGE_MODEL}")

    if not os.environ.get("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY not set.")
        print("Run: export OPENAI_API_KEY='your-key-here'")
        return

    sessions = load_log_files(logs_dir)
    print(f"Sessions found: {len(sessions)}")

    if not sessions:
        print(f"No session log files found in {logs_dir}/")
        return

    all_results = []
    for session in sessions:
        results = evaluate_session(session, variant)
        all_results.extend(results)

    averages = compute_and_print_averages(all_results, variant)
    save_results(all_results, averages, variant)


if __name__ == "__main__":
    main()