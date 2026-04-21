import json
from datetime import datetime
from llm.model import generate  # your existing function

# -------------------------------
# CONFIG
# -------------------------------
TEMPERATURE = 0.1
LOG_FILE = "extraction_logs.json"  # JSONL format (one record per line)

# -------------------------------
# PROMPT TEMPLATE
# -------------------------------
EXTRACTION_PROMPT_TEMPLATE = """Extract structured information from the patient message.

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
  Extract ANY reason stated, not limited to examples.
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

# -------------------------------
# HELPER: Extract JSON safely
# -------------------------------
def parse_json(output):
    try:
        start = output.index("{")
        end = output.rindex("}") + 1
        return json.loads(output[start:end])
    except Exception:
        return None

# -------------------------------
# MAIN LOOP
# -------------------------------
def run():
    print("=== Extraction Tester ===")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("Patient: ").strip()

        if user_input.lower() == "exit":
            print("Exiting...")
            break

        # Build prompt
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            user_message=user_input
        )

        # Call LLM
        raw_output = generate(prompt, temperature=TEMPERATURE)

        # Parse JSON
        parsed = parse_json(raw_output)

        # Print result
        print("\n--- Extracted ---")
        if parsed:
            print(json.dumps(parsed, indent=2))
        else:
            print("❌ Invalid JSON:", raw_output)

        # Save to log file
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "raw_output": raw_output,
            "parsed_output": parsed
        }

        # Load existing logs
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
        except:
            logs = []

        # Append new entry
        logs.append(log_entry)

        # Save pretty JSON
        with open(LOG_FILE, "w") as f:
            json.dump(logs, f, indent=2)

        print(f"\n💾 Saved to {LOG_FILE}\n")


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    run()