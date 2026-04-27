# evaluation/pairwise_evaluate.py
"""
Pairwise Evaluation for the Tobacco Addiction Therapy Chatbot.

For the same patient message, shows the judge two responses —
one from the Full system, one from an ablated variant — and asks
which is better. This is more discriminative than absolute scoring.

Fixed patient conversations ensure identical patient input across all variants.

Usage:
    # Step 1 — generate responses (change variant, comment pipeline components)
    python evaluation/pairwise_evaluate.py --mode generate --variant full
    python evaluation/pairwise_evaluate.py --mode generate --variant no_phase
    python evaluation/pairwise_evaluate.py --mode generate --variant no_rag
    python evaluation/pairwise_evaluate.py --mode generate --variant no_cot

    # Step 2 — run judge
    python evaluation/pairwise_evaluate.py --mode judge --compare no_phase
    python evaluation/pairwise_evaluate.py --mode judge --compare no_rag
    python evaluation/pairwise_evaluate.py --mode judge --compare no_cot

    # Step 3 — summary table
    python evaluation/pairwise_evaluate.py --mode summary
"""

import json
import os
import sys
import re
import time
import argparse
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pipeline.therapy_pipeline as pipeline_module
import logger.conversation_history as conv_history_module
from logger.session_logger import SessionLogger
from pipeline.therapy_pipeline import therapy_chat, reset_for_new_patient

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

RESULTS_DIR = "evaluation/pairwise_results"
JUDGE_MODEL = "openai/gpt-4o-mini"
VARIANTS    = ["full", "no_phase", "no_rag", "no_cot"]

judge_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ------------------------------------------------------------------
# Fixed patient scenarios — identical input for all variants
# ------------------------------------------------------------------

PAIRWISE_SCENARIOS = [
    {
        "patient_id": "pw_p01",
        "description": "Long-term smoker, doctor warning, clear triggers and motivation",
        "turns": [
            "I have been smoking cigarettes for 15 years, about a pack a day.",
            "Mostly after meals and when I am stressed at work.",
            "My doctor told me my lungs are getting worse. My kids also keep asking me to stop.",
            "I tried nicotine patches once but I kept forgetting to put them on.",
            "I want to quit completely. I just do not know where to start.",
            "Maybe I can try keeping gum with me for after meals. That sounds doable.",
        ],
    },
    {
        "patient_id": "pw_p02",
        "description": "Bidi smoker, family pressure, ambivalent about quitting",
        "turns": [
            "I smoke bidis, about 20 a day for 30 years.",
            "I do not know if I really want to quit. My son keeps pushing me.",
            "I smoke most in the morning after tea and when I feel bored at home.",
            "I never tried anything before. I always thought I could stop on my own.",
            "My breathing is getting bad lately. Maybe that is a reason to try.",
            "Okay I will think about it. Maybe small steps.",
        ],
    },
    {
        "patient_id": "pw_p03",
        "description": "Young smoker, social and stress triggers, wants to reduce",
        "turns": [
            "I smoke about 5 cigarettes a day, been doing it for 2 years.",
            "Mainly when I am with friends who smoke or during exam stress.",
            "I want to reduce first. I am not ready to quit completely yet.",
            "I tried avoiding parties for a week but ended up smoking more during exams.",
            "I think gum or something to keep my hands busy might help.",
        ],
    },
    {
        "patient_id": "pw_p04",
        "description": "Heavy smoker, multiple failed strategies, health motivation",
        "turns": [
            "I have been smoking two packs a day for 20 years.",
            "I smoke after every meal, with my morning coffee, and when I feel anxious.",
            "I want to quit because I had chest pain last month and my doctor is worried.",
            "I tried nicotine gum, patches, and cold turkey. None of them worked for long.",
            "I always go back when I get very stressed. That is my biggest problem.",
            "I think I need something for the stress specifically, not just the craving.",
        ],
    },
    {
        "patient_id": "pw_p05",
        "description": "Resistant patient, short answers, unclear motivation",
        "turns": [
            "I smoke. Have been for about 10 years.",
            "I do not know. Sometimes after food I guess.",
            "My wife wants me to stop.",
            "I have not really tried anything.",
            "I am not sure I am ready.",
        ],
    },
    {
        "patient_id": "pw_p06",
        "description": "Patient dumps all info in turn 1 — tests phase handling",
        "turns": [
            "I have been smoking bidis for 25 years, about 15 a day. I want to quit because of my health and my grandchildren. I tried nicotine gum before but did not like the taste. I usually smoke after meals and when I am stressed.",
            "Yes I am serious about quitting this time.",
            "The stress is the hardest part. Work and money problems.",
            "I have not really tried anything for stress specifically.",
            "Maybe I can try something for after meals first. That seems easier.",
        ],
    },
    {
        "patient_id": "pw_p07",
        "description": "Patient asks direct question — tests MI fidelity",
        "turns": [
            "I smoke cigarettes, about 10 a day for 8 years.",
            "Stress and boredom mostly. Also after eating.",
            "I want to quit for my health mainly.",
            "What do you think I should do? Just tell me what works.",
            "Okay I will try that. What if it does not work though?",
        ],
    },
    {
        "patient_id": "pw_p08",
        "description": "Chewing tobacco user — tests substance-agnostic handling",
        "turns": [
            "I use chewing tobacco, about 5 pouches a day for 12 years.",
            "After meals and during work mainly. It helps me concentrate.",
            "I want to stop because my dentist said there is some damage to my gums.",
            "I tried stopping once but got very irritable and could not focus at work.",
            "Maybe I can reduce first and see how it goes.",
        ],
    },
    {
        "patient_id": "pw_p09",
        "description": "Resistant patient with pushback",
        "turns": [
            "I smoke about 20 cigarettes a day. Been smoking for 15 years.",
            "I smoke when I feel like it. There is no specific trigger.",
            "I am here because my doctor sent me. I do not particularly want to quit.",
            "I have tried before and it does not work. Quitting just makes me miserable.",
            "Fine. What is the point of all these questions anyway.",
        ],
    },
    {
        "patient_id": "pw_p10",
        "description": "Patient progresses through all phases naturally",
        "turns": [
            "I have been smoking for 7 years, about 12 cigarettes a day.",
            "I smoke when I am anxious and in social situations with other smokers.",
            "I want to quit because I am planning to have a baby soon.",
            "I tried cold turkey twice. The second time I lasted 3 months then relapsed at a party.",
            "Social situations are the hardest. I do not know how to say no.",
            "Okay I think I can try having a response ready when someone offers me a cigarette.",
            "Yes that sounds like a good plan. I will try it this weekend.",
        ],
    },
]


# ------------------------------------------------------------------
# Setup variant directories
# ------------------------------------------------------------------

def setup_variant(variant: str):
    variant_dir = f"{RESULTS_DIR}/logs/{variant}"
    kg_dir      = f"{RESULTS_DIR}/kg/{variant}"
    os.makedirs(variant_dir, exist_ok=True)
    os.makedirs(kg_dir,      exist_ok=True)

    pipeline_module.logger           = SessionLogger(log_dir=variant_dir)
    pipeline_module.kg.storage_path  = f"{kg_dir}/patient_profiles.json"
    pipeline_module.kg.profiles      = {}
    conv_history_module.HISTORY_FILE = f"{variant_dir}/conversation_history.json"


# ------------------------------------------------------------------
# Generate responses for one variant
# ------------------------------------------------------------------

def generate_responses(variant: str):
    setup_variant(variant)
    all_responses = {}

    print(f"\n{'='*60}")
    print(f"Generating responses — {variant.upper()}")
    print(f"Ensure pipeline is configured for this variant.")
    print(f"{'='*60}")

    for scenario in PAIRWISE_SCENARIOS:
        patient_id  = scenario["patient_id"]
        turns       = scenario["turns"]
        pid_variant = f"pw_{variant}_{patient_id}"

        reset_for_new_patient(pid_variant)
        print(f"\n  [{patient_id}] {scenario['description']}")

        scenario_responses = []
        for turn_num, patient_msg in enumerate(turns, 1):
            print(f"    Turn {turn_num}: {patient_msg[:70]}...")
            try:
                response = therapy_chat(patient_msg, pid_variant)
                print(f"             → {response[:80]}...")
            except Exception as e:
                print(f"    [ERROR] {e}")
                response = ""

            scenario_responses.append({
                "turn":        turn_num,
                "patient_msg": patient_msg,
                "response":    response,
            })
            time.sleep(3)

        all_responses[patient_id] = scenario_responses

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"responses_{variant}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "variant":   variant,
            "timestamp": datetime.now().isoformat(),
            "responses": all_responses,
        }, f, indent=2)
    print(f"\n  Saved → {path}")


# ------------------------------------------------------------------
# Simple KG accumulation from patient message
# Approximates pipeline KG so judge has patient context
# ------------------------------------------------------------------

def _update_kg(kg: dict, patient_msg: str):
    msg = patient_msg.lower()

    if not kg["smoking_status"]:
        for substance in ["cigarette", "bidi", "tobacco", "chewing"]:
            if substance in msg:
                qty   = re.search(r"(\d+)\s*(?:a|per)?\s*day", msg)
                years = re.search(r"(\d+)\s*year", msg)
                if qty:
                    kg["smoking_status"] = f"{substance}s, {qty.group(1)}/day"
                elif years:
                    kg["smoking_status"] = f"{substance}s for {years.group(1)} years"
                break

    if not kg["quit_goal"] and any(w in msg for w in ["want to quit", "want to stop", "trying to quit"]):
        kg["quit_goal"] = "quit smoking"

    trigger_map = {
        "after meal": "after meals", "after eating": "after meals",
        "stress": "stress", "anxious": "anxiety", "bored": "boredom",
        "social": "social situations", "morning": "morning craving",
        "work": "work stress", "alcohol": "alcohol",
    }
    for kw, label in trigger_map.items():
        if kw in msg and label not in kg["relevant_triggers"]:
            kg["relevant_triggers"].append(label)

    motivation_map = {
        "health": "health concerns", "doctor": "doctor advised",
        "child": "family pressure", "family": "family pressure",
        "breathing": "breathing problems",
    }
    for kw, label in motivation_map.items():
        if kw in msg and label not in kg["motivation_reason"]:
            kg["motivation_reason"].append(label)


def _kg_to_text(kg: dict) -> str:
    lines = ["[Patient Profile]"]
    if kg.get("smoking_status"):
        lines.append(f"- Smoking status: {kg['smoking_status']}")
    if kg.get("quit_goal"):
        lines.append(f"- Quit goal: {kg['quit_goal']}")
    if kg.get("motivation_reason"):
        lines.append(f"- Motivation: {', '.join(kg['motivation_reason'])}")
    if kg.get("relevant_triggers"):
        lines.append(f"- Known triggers: {', '.join(kg['relevant_triggers'])}")
    if len(lines) == 1:
        lines.append("- No information collected yet.")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Pairwise judge prompt
# ------------------------------------------------------------------

def build_pairwise_prompt(
    kg_text: str,
    history_block: str,
    user_message: str,
    response_a: str,
    response_b: str,
) -> str:
    return f"""You are an expert clinical psychologist comparing two therapy chatbot responses to the same patient message.

{kg_text}

{history_block}

[Current Patient Message]
{user_message}

[Response A]
{response_a}

[Response B]
{response_b}

Which response is more therapeutically appropriate at this stage?

Consider:
- Does it correctly gather information before suggesting strategies?
- Does it acknowledge the patient's specific situation warmly?
- Does it ask exactly one focused open-ended question?
- Does it avoid suggesting coping techniques before understanding triggers?
- Does it advance the conversation by gathering new information?

Return ONLY valid JSON:
{{
  "winner": "A" or "B" or "tie",
  "confidence": "high" or "medium" or "low",
  "reasoning": "<two sentences explaining the key difference>"
}}"""


# ------------------------------------------------------------------
# Run pairwise judge — Full vs ablated variant
# Uses position alternation to control for position bias
# ------------------------------------------------------------------

def run_pairwise_judge(compare_variant: str):
    full_path    = os.path.join(RESULTS_DIR, "responses_full.json")
    compare_path = os.path.join(RESULTS_DIR, f"responses_{compare_variant}.json")

    for path, name in [(full_path, "full"), (compare_path, compare_variant)]:
        if not os.path.exists(path):
            print(f"ERROR: {name} responses not found at {path}")
            print(f"Run: python evaluation/pairwise_evaluate.py --mode generate --variant {name}")
            return

    with open(full_path)    as f: full_data    = json.load(f)
    with open(compare_path) as f: compare_data = json.load(f)

    full_responses    = full_data["responses"]
    compare_responses = compare_data["responses"]

    results      = []
    full_wins    = 0
    compare_wins = 0
    ties         = 0
    total        = 0

    print(f"\n{'='*60}")
    print(f"Pairwise Judge: Full vs {compare_variant.upper()}")
    print(f"{'='*60}")

    for scenario in PAIRWISE_SCENARIOS:
        patient_id    = scenario["patient_id"]
        full_turns    = full_responses.get(patient_id, [])
        compare_turns = compare_responses.get(patient_id, [])

        if not full_turns or not compare_turns:
            print(f"\n  [{patient_id}] Skipping — missing responses")
            continue

        print(f"\n  [{patient_id}] {scenario['description']}")

        # Accumulated KG and history for context
        accumulated_kg = {
            "smoking_status": None, "quit_goal": None,
            "motivation_reason": [], "relevant_triggers": [],
        }
        history_msgs = []

        for turn_num in range(1, min(len(full_turns), len(compare_turns)) + 1):
            full_turn    = next((t for t in full_turns    if t["turn"] == turn_num), None)
            compare_turn = next((t for t in compare_turns if t["turn"] == turn_num), None)

            if not full_turn or not compare_turn:
                continue

            patient_msg  = full_turn["patient_msg"]
            response_a   = full_turn["response"]    # Full
            response_b   = compare_turn["response"] # Compare

            if not response_a or not response_b:
                continue

            # Build context
            kg_text = _kg_to_text(accumulated_kg)
            history_lines = ["[Recent Conversation]"]
            for prev in history_msgs[-6:]:
                history_lines.append(f"{prev['role']}: {prev['content']}")
            history_block = "\n".join(history_lines) if len(history_lines) > 1 else "[Recent Conversation]\n(This is the first turn.)"

            # Alternate position to control bias
            # Odd turns: Full=A, Compare=B
            # Even turns: Compare=A, Full=B
            if turn_num % 2 == 1:
                prompt_a, prompt_b, full_is_a = response_a, response_b, True
            else:
                prompt_a, prompt_b, full_is_a = response_b, response_a, False

            prompt = build_pairwise_prompt(kg_text, history_block, patient_msg, prompt_a, prompt_b)

            print(f"    Turn {turn_num}: Judging...", end=" ", flush=True)

            try:
                api_response = judge_client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=300,
                )
                time.sleep(1)
                raw = api_response.choices[0].message.content.strip()

                if raw.startswith("```"):
                    parts = raw.split("```")
                    raw = parts[1] if len(parts) > 1 else raw
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()

                judgment   = json.loads(raw)
                raw_winner = judgment.get("winner", "tie")

                # Correct for position swap
                if raw_winner == "tie":
                    actual_winner = "tie"
                elif full_is_a:
                    actual_winner = "full" if raw_winner == "A" else compare_variant
                else:
                    actual_winner = "full" if raw_winner == "B" else compare_variant

                if actual_winner == "full":       full_wins    += 1
                elif actual_winner == compare_variant: compare_wins += 1
                else:                             ties         += 1
                total += 1

                print(f"Winner={actual_winner} ({judgment.get('confidence','?')}) — {judgment.get('reasoning','')[:80]}")

                results.append({
                    "patient_id":       patient_id,
                    "turn":             turn_num,
                    "patient_msg":      patient_msg[:100],
                    "full_response":    response_a[:100],
                    "compare_response": response_b[:100],
                    "raw_winner":       raw_winner,
                    "actual_winner":    actual_winner,
                    "full_was_A":       full_is_a,
                    "confidence":       judgment.get("confidence"),
                    "reasoning":        judgment.get("reasoning"),
                })

            except Exception as e:
                print(f"FAILED — {e}")

            # Update context for next turn (use Full system response as gold history)
            history_msgs.append({"role": "Patient",   "content": patient_msg})
            history_msgs.append({"role": "Therapist", "content": response_a})
            _update_kg(accumulated_kg, patient_msg)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: Full vs {compare_variant.upper()}")
    print(f"{'='*60}")
    if total > 0:
        print(f"  Full wins         : {full_wins}/{total} ({round(full_wins/total*100)}%)")
        print(f"  {compare_variant} wins  : {compare_wins}/{total} ({round(compare_wins/total*100)}%)")
        print(f"  Ties              : {ties}/{total} ({round(ties/total*100)}%)")

    summary = {
        "comparison":       f"full_vs_{compare_variant}",
        "timestamp":        datetime.now().isoformat(),
        "total_turns":      total,
        "full_wins":        full_wins,
        "compare_wins":     compare_wins,
        "ties":             ties,
        "full_win_rate":    round(full_wins / total, 2) if total else 0,
        "compare_win_rate": round(compare_wins / total, 2) if total else 0,
        "tie_rate":         round(ties / total, 2) if total else 0,
        "turn_results":     results,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"pairwise_full_vs_{compare_variant}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved → {path}")


# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------

def print_summary():
    print(f"\n{'='*70}")
    print("PAIRWISE SUMMARY — Full System vs Ablations")
    print(f"{'='*70}")
    print(f"{'Comparison':<25} {'Full Wins':>12} {'Ablation Wins':>15} {'Ties':>8} {'Full%':>8}")
    print("-" * 70)

    for variant in ["no_phase", "no_rag", "no_cot"]:
        path = os.path.join(RESULTS_DIR, f"pairwise_full_vs_{variant}.json")
        if not os.path.exists(path):
            print(f"  full_vs_{variant:<16} not yet run")
            continue
        with open(path) as f:
            data = json.load(f)
        t   = data["total_turns"]
        fw  = data["full_wins"]
        aw  = data["compare_wins"]
        tie = data["ties"]
        pct = round(fw / t * 100) if t else 0
        print(f"  full_vs_{variant:<16} {fw:>5}/{t:<6} {aw:>6}/{t:<8} {tie:>4}/{t:<3} {pct:>6}%")

    print(f"{'='*70}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["generate", "judge", "summary"], required=True)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--compare", choices=["no_phase", "no_rag", "no_cot"])
    args = parser.parse_args()

    if args.mode == "generate":
        if not args.variant:
            print("ERROR: --variant required for generate mode")
            return
        generate_responses(args.variant)

    elif args.mode == "judge":
        if not args.compare:
            print("ERROR: --compare required for judge mode")
            return
        if not os.getenv("OPENROUTER_API_KEY"):
            print("ERROR: OPENROUTER_API_KEY not set")
            return
        run_pairwise_judge(args.compare)

    elif args.mode == "summary":
        print_summary()


if __name__ == "__main__":
    main()