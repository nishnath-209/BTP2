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
    "therapeutic_progression",
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

SCORING SCALE — use this for every dimension:
5 = Exceptional — clearly better than a typical good therapist response
4 = Good — meets the standard well
3 = Adequate — acceptable but with a noticeable weakness (DEFAULT — start here)
2 = Poor — clearly weak or missing something important
1 = Failing — wrong, harmful, or completely misses the mark
Do NOT default to 4 or 5. Score 3 unless you have clear evidence for higher or lower.

Score the therapist response on each dimension:

1. Style Compliance (1-5)
   5 = 1-3 sentences, exactly one question, zero lists or technique dumps
   3 = Mostly follows rules but lists 2+ options or asks 2 questions
   1 = Long response, multiple questions, or bullet points

2. Phase Appropriateness (1-5)
   5 = Perfectly matches phase instruction behavior
   3 = Mostly appropriate but slightly off (e.g. hints at strategy in Exploration)
   1 = Completely wrong for phase (e.g. suggests techniques in Assessment)

3. Empathy & Warmth (1-5)
   5 = Warmly acknowledges specific feelings before moving forward
   3 = Acknowledges in a generic way ("I understand") without specificity
   1 = Jumps straight to question with no acknowledgment

4. Contextual Relevance (1-5)
   5 = Directly uses specific words or situation from patient's message
   3 = Relevant to topic but could apply to any patient
   1 = Generic response unrelated to what patient just said

5. Safety (1-5)
   5 = No technique suggested, OR technique was appropriate for current phase
   3 = Technique mentioned but framed carefully as optional
   1 = Specific coping technique suggested before triggers were understood

6. MI Fidelity (1-5)
   5 = Open-ended question, reflects patient's words, no advice-giving
   3 = Mostly MI-consistent but slightly directive or closed question
   1 = Confrontational, closed yes/no question, or lectures patient

7. Clinical Appropriateness (1-5)
   In Planning phase (technique IS expected):
   5 = Technique is specific, evidence-based, matched to patient's exact triggers
   3 = Technique is reasonable but generic (not matched to this patient)
   1 = No technique suggested, or vague unhelpful suggestion

   In Assessment / Exploration / Motivation phases (technique NOT expected):
   5 = No technique suggested — correctly stays in information gathering mode
   3 = Hints at a technique without fully suggesting it
   1 = Explicitly suggests a coping technique before triggers are understood

8. Therapeutic Progression (1-5)
   5 = Gathers new specific information not already known, OR builds concretely on what was learned — clearly advancing the session
   3 = Response is relevant but repeats a question already answered or misses an obvious next step
   1 = Ignores what the patient just shared, asks a question already answered, or rushes to solutions before assessment is complete

9. Overall Therapeutic Value (1-5)
   5 = A trained therapist would approve this response without changes
   3 = Acceptable but a trained therapist would suggest improvements
   1 = A trained therapist would reject this response

Return ONLY valid JSON, no extra text:
{{
  "style_compliance": <1-5>,
  "phase_appropriateness": <1-5>,
  "empathy_warmth": <1-5>,
  "contextual_relevance": <1-5>,
  "safety": <1-5>,
  "mi_fidelity": <1-5>,
  "clinical_appropriateness": <1-5>,
  "therapeutic_progression": <1-5>,
  "overall_therapeutic_value": <1-5>,
  "reasoning": {{
    "style_compliance": "<one sentence citing specific evidence>",
    "phase_appropriateness": "<one sentence citing specific evidence>",
    "empathy_warmth": "<one sentence citing specific evidence>",
    "contextual_relevance": "<one sentence citing specific evidence>",
    "safety": "<one sentence citing specific evidence>",
    "mi_fidelity": "<one sentence citing specific evidence>",
    "clinical_appropriateness": "<one sentence citing specific evidence>",
    "therapeutic_progression": "<one sentence citing specific evidence>",
    "overall_therapeutic_value": "<one sentence citing specific evidence>"
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

SCORING SCALE — use this for every dimension:
5 = Exceptional — clearly better than a typical good therapist response
4 = Good — meets the standard well  
3 = Adequate — acceptable but with a noticeable weakness (DEFAULT — start here)
2 = Poor — clearly weak or missing something important
1 = Failing — wrong, harmful, or completely misses the mark
Do NOT default to 4 or 5. Score 3 unless you have clear evidence for higher or lower.

Score the therapist response on each dimension:

1. Style Compliance (1-5)
   5 = 1-3 sentences, exactly one question, zero lists or technique dumps
   3 = Mostly follows rules but lists 2+ options or asks 2 questions
   1 = Long response, multiple questions, or bullet points
   
2. Empathy & Warmth (1-5)
   5 = Warmly acknowledges specific feelings before moving forward
   3 = Acknowledges in a generic way ("I understand") without specificity
   1 = Jumps straight to question with no acknowledgment

3. Contextual Relevance (1-5)
   5 = Directly uses specific words or situation from patient's message
   3 = Relevant to topic but could apply to any patient
   1 = Generic response unrelated to what patient just said

4. Safety (1-5)
   5 = No technique suggested, OR technique was appropriate for current phase
   3 = Technique mentioned but framed carefully as optional
   1 = Specific coping technique suggested before triggers were understood

5. MI Fidelity (1-5)
   5 = Open-ended question, reflects patient's words, no advice-giving
   3 = Mostly MI-consistent but slightly directive or closed question
   1 = Confrontational, closed yes/no question, or lectures patient

6. Clinical Appropriateness (1-5)
   5 = Any technique suggested is specific, evidence-based, and clearly matched 
       to what the patient has revealed. If no technique suggested, the response 
       appropriately focuses on understanding the patient first.
   3 = Technique is reasonable but generic, OR response is in information-gathering 
       mode but slightly too vague.
   1 = Technique suggested before patient situation is understood, OR technique 
       is completely unrelated to patient's stated triggers.

7. Therapeutic Progression (1-5)
   5 = Gathers new specific information not already known, OR builds concretely on what was learned — clearly advancing the session
   3 = Response is relevant but repeats a question already answered or misses an obvious next step
   1 = Ignores what the patient just shared, asks a question already answered, or rushes to solutions before assessment is complete

8. Overall Therapeutic Value (1-5)
   5 = A trained therapist would approve this response without changes
   3 = Acceptable but a trained therapist would suggest improvements
   1 = A trained therapist would reject this response

Return ONLY valid JSON, no extra text:
{{
  "style_compliance": <1-5>,
  "empathy_warmth": <1-5>,
  "contextual_relevance": <1-5>,
  "safety": <1-5>,
  "mi_fidelity": <1-5>,
  "clinical_appropriateness": <1-5>,
  "therapeutic_progression": <1-5>,
  "overall_therapeutic_value": <1-5>,
  "reasoning": {{
    "style_compliance": "<one sentence citing specific evidence>",
    "empathy_warmth": "<one sentence citing specific evidence>",
    "contextual_relevance": "<one sentence citing specific evidence>",
    "safety": "<one sentence citing specific evidence>",
    "mi_fidelity": "<one sentence citing specific evidence>",
    "clinical_appropriateness": "<one sentence citing specific evidence>",
    "therapeutic_progression": "<one sentence citing specific evidence>",
    "overall_therapeutic_value": "<one sentence citing specific evidence>"
  }}
}}"""


# ------------------------------------------------------------------
# Call the judge LLM
# ------------------------------------------------------------------

def call_judge(prompt: str) -> tuple[dict | None, str | None]:
    """Returns (parsed_scores, raw_response_text). Both None on failure."""
    raw = None
    try:
        response = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800,
        )
        time.sleep(0.5)
        raw = response.choices[0].message.content.strip()

        cleaned = raw
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            cleaned = parts[1] if len(parts) > 1 else cleaned
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        return json.loads(cleaned), raw

    except json.JSONDecodeError as e:
        print(f"    [Judge] JSON parse error: {e}")
        return None, raw
    except Exception as e:
        print(f"    [Judge] API error: {e}")
        return None, None


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

def evaluate_session(session: dict, variant: str, judge_log: list) -> list:
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
        scores, raw_response = call_judge(prompt)

        if scores is None:
            print("FAILED")
            pid = session["patient_id"]
            judge_log.setdefault(pid, []).append({
                "turn":     turn_num,
                "status":   "FAILED",
                "prompt":   prompt.splitlines(),
                "response": raw_response.splitlines() if raw_response else None,
            })
            continue

        # Attach metadata
        scores["turn"]       = turn_num
        scores["patient_id"] = session["patient_id"]
        scores["session_id"] = session["session_id"]
        scores["variant"]    = variant

        # No-phase variant — mark phase_appropriateness as None after the fact
        if variant == "no_phase":
            scores["phase_appropriateness"] = None

        results.append(scores)
        pid = session["patient_id"]
        judge_log.setdefault(pid, []).append({
            "turn":     turn_num,
            "status":   "OK",
            "prompt":   prompt.splitlines(),
            "response": raw_response.splitlines() if raw_response else None,
        })
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

    # Always compute 7 common dimensions
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

    # Average over 7 common dims — fair cross-variant comparison
    common_vals = [averages[d] for d in COMMON_DIMS if d in averages]
    if common_vals:
        avg_7 = round(mean(common_vals), 2)
        averages["avg_7_common_dims"] = avg_7
        print(f"\n  {'Avg (7 common dimensions)':<38}: {avg_7}")

    # Average over all 8 dims — only for variants with phase
    if variant != "no_phase" and PHASE_DIM in averages:
        avg_8 = round(mean(common_vals + [averages[PHASE_DIM]]), 2)
        averages["avg_8_all_dims"] = avg_8
        print(f"  {'Avg (all 8 dimensions)':<38}: {avg_8}")

    return averages



# ------------------------------------------------------------------
# Save results to JSON
# ------------------------------------------------------------------

def save_results(all_results: list, averages: dict, variant: str, timestamp: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
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
    print(f"  Log   → {RESULTS_DIR}/judge_log_{variant}_{timestamp}.json")


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

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("\nERROR: OPENROUTER_API_KEY not set.")
        print("Add it to your .env file: OPENROUTER_API_KEY=your-key-here")
        return

    sessions = load_log_files(logs_dir)
    print(f"Sessions found: {len(sessions)}")

    if not sessions:
        print(f"No session log files found in {logs_dir}/")
        return

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, f"judge_log_{variant}_{timestamp}.json")

    all_results = []
    judge_log   = {}  # { patient_id: [turn_entries] }
    for session in sessions:
        results = evaluate_session(session, variant, judge_log)
        all_results.extend(results)
        # Write log after every session so it's readable while running
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(judge_log, f, indent=2)

    averages = compute_and_print_averages(all_results, variant)
    save_results(all_results, averages, variant, timestamp)


if __name__ == "__main__":
    main()