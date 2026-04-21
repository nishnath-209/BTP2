# evaluation/automatic_metrics.py
"""
Automatic Metrics Evaluation — BLEU, ROUGE-1, ROUGE-2, ROUGE-L, METEOR
for the Tobacco Addiction Therapy Chatbot.

Uses the EDosthi dataset as the reference (gold standard).

How it works per turn:
    1. Feed EDosthi patient message to therapy_chat()
    2. therapy_chat() runs the full pipeline:
           KG update → RAG → subgraph → phase → prompt → generate
    3. Compare our generated response against the EDosthi provider response
       using BLEU, ROUGE, METEOR
    4. Replace our generated response in history with EDosthi provider response
       so the next turn sees the gold context (not our diverged response)
    5. KG is updated from patient messages only — untouched, correct as-is

Why EDosthi provider response goes into history (not ours):
    The reference provider responded to a specific conversation context.
    To compare fairly at each turn, our system should see the same context
    the reference provider saw. This is standard practice in turn-level
    dialogue evaluation.

Install dependencies:
    pip install rouge-score nltk --break-system-packages
    python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab')"

Usage:
    python evaluation/automatic_metrics.py --variant full
    python evaluation/automatic_metrics.py --variant no_phase
    python evaluation/automatic_metrics.py --variant no_rag
    python evaluation/automatic_metrics.py --variant no_cot
    python evaluation/automatic_metrics.py --variant all
"""

import json
import os
import re
import sys
import argparse
from statistics import mean
import time
from datetime import datetime


from rouge_score import rouge_scorer as rouge_scorer_lib
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize


# Add project root to path so pipeline imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the actual pipeline — no reimplementation needed
import pipeline.therapy_pipeline as pipeline_module
from pipeline.therapy_pipeline import therapy_chat, reset_for_new_patient
import logger.conversation_history as conv_history_module
from logger.session_logger import SessionLogger
from kg.knowledge_graph import KnowledgeGraph

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

DATASET_PATH = "D:/BTP/p/data/edosthi_dataset.json"
RESULTS_DIR  = "evaluation/results"

# Set to an integer to limit (e.g. 20 for quick test), None for all 100
MAX_CONVERSATIONS = None
START_FROM = 0  # change to resume from a specific conversation index


VARIANTS = ["full", "no_phase", "no_rag", "no_cot"]


# ------------------------------------------------------------------
# Clean EDosthi messages
# Dataset has stage directions like (sighs), (shrugs), (pauses)
# ------------------------------------------------------------------

def clean_message(text: str) -> str:
    text = re.sub(r"\(.*?\)", "", text)
    return " ".join(text.split()).strip()


# Metric computation

def compute_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = word_tokenize(reference.lower())
    hyp_tokens = word_tokenize(hypothesis.lower())
    smoother   = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)


def compute_rouge(reference: str, hypothesis: str) -> dict:
    scorer = rouge_scorer_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_meteor(reference: str, hypothesis: str) -> float:
    ref_tokens = word_tokenize(reference.lower())
    hyp_tokens = word_tokenize(hypothesis.lower())
    return meteor_score([ref_tokens], hyp_tokens)


def compute_all_metrics(reference: str, hypothesis: str) -> dict:
    bleu   = compute_bleu(reference, hypothesis)
    rouge  = compute_rouge(reference, hypothesis)
    meteor = compute_meteor(reference, hypothesis)
    return {
        "bleu":   round(bleu, 4),
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "meteor": round(meteor, 4),
    }


# ------------------------------------------------------------------
# Evaluate one variant across the dataset
# ------------------------------------------------------------------


def evaluate_variant(variant: str, dataset: list) -> dict:

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Variant-specific folders
    variant_log_dir = f"evaluation/logs/{variant}"
    variant_kg_dir  = f"evaluation/kg/{variant}"
    os.makedirs(variant_log_dir, exist_ok=True)
    os.makedirs(variant_kg_dir,  exist_ok=True)

    # Override logger
    pipeline_module.logger = SessionLogger(log_dir=variant_log_dir)

    # Override conversation history
    conv_history_module.HISTORY_FILE = (
        f"evaluation/logs/{variant}/conversation_history.json"
    )

    # Override KG storage path
    pipeline_module.kg = KnowledgeGraph(
        storage_path=f"evaluation/kg/{variant}/patient_profiles.json"
    )
    

    print(f"\n{'='*60}")
    print(f"Variant : {variant.upper()}")
    print(f"{'='*60}")

    all_scores         = []
    conversations_done = 0
    limit = (START_FROM + MAX_CONVERSATIONS) if MAX_CONVERSATIONS else len(dataset)

    for conv_idx, conversation in enumerate(dataset):
        if conv_idx < START_FROM:
            continue
        if conv_idx >= limit:
            break

        messages   = conversation["messages"]
        conv_id    = conversation["metadata"].get("conversation_id", conv_idx)
        patient_id = f"eval_{variant}_{conv_id}"

        # Reset all pipeline global state for this new conversation:
        #   _conversation_history = []
        #   _turn = 0
        #   _reached_planning = False
        #   SESSION_ID = new timestamp
        #   KG profile for this patient_id deleted
        reset_for_new_patient(patient_id)

        print(f"\n  Conv {conv_idx + 1}/{limit}  "
              f"(id={conv_id}, msgs={len(messages)})")

        # EDosthi always starts with a provider opening message.
        # Add it directly to pipeline's _conversation_history as a therapist
        # greeting so our system has the same starting context.
        if messages and messages[0]["role"] == "provider":
            opening = clean_message(messages[0]["content"])
            pipeline_module._conversation_history.append({
                "role": "therapist",
                "content": opening
            })

        turn_num = 0

        for i, msg in enumerate(messages):

            if msg["role"] != "patient":
                continue

            patient_message = clean_message(msg["content"])
            if not patient_message:
                continue

            # Find the next provider turn — this is our gold reference
            reference = None
            for j in range(i + 1, len(messages)):
                if messages[j]["role"] == "provider":
                    reference = clean_message(messages[j]["content"])
                    break

            if not reference:
                continue

            turn_num += 1

            # Call the actual pipeline
            # therapy_chat() will:
            #   - read pipeline._conversation_history (gold context so far)
            #   - extract from patient_message and update KG
            #   - run RAG, phase detection, prompt building, generation
            #   - append patient_message + OUR response to _conversation_history
            try:
                our_response = therapy_chat(patient_message, patient_id)
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    print(f"    Rate limit hit. Saving progress and stopping.")
                    save_results(
                        {
                            "variant":                 variant,
                            "conversations_evaluated": conversations_done,
                            "total_turns_evaluated":   len(all_scores),
                            "turn_scores":             all_scores,
                        },
                        {},
                        variant,
                        run_timestamp,
                    )
                    return {
                        "variant":                 variant,
                        "conversations_evaluated": conversations_done,
                        "total_turns_evaluated":   len(all_scores),
                        "turn_scores":             all_scores,
                        "run_timestamp":           run_timestamp,
                    }
                print(f"    Turn {turn_num}: Error — {e}. Skipping.")
                time.sleep(4)
                continue
            
            

            # Compute metrics: our response vs EDosthi gold reference
            metrics = compute_all_metrics(reference, our_response)
            metrics.update({
                "conv_id":     conv_id,
                "turn":        turn_num,
                "patient_msg": patient_message[:100],
                "reference":   reference[:100],
                "generated":   our_response[:100],
            })
            all_scores.append(metrics)

            print(
                f"    Turn {turn_num}: "
                f"BLEU={metrics['bleu']:.3f}  "
                f"R1={metrics['rouge1']:.3f}  "
                f"R2={metrics['rouge2']:.3f}  "
                f"RL={metrics['rougeL']:.3f}  "
                f"METEOR={metrics['meteor']:.3f}"
            )

            # IMPORTANT: therapy_chat() added OUR response to _conversation_history.
            # Replace it with the EDosthi gold reference so the next turn
            # sees the same context the real provider saw — not our diverged response.
            # KG is already correctly updated from patient_message — do not touch it.
            if (pipeline_module._conversation_history and
                    pipeline_module._conversation_history[-1]["role"] == "therapist"):
                pipeline_module._conversation_history[-1]["content"] = reference

            time.sleep(2)

        conversations_done += 1
        save_results(
            {
                "variant":                 variant,
                "conversations_evaluated": conversations_done,
                "total_turns_evaluated":   len(all_scores),
                "turn_scores":             all_scores,
            },
            {},   # averages computed at the very end, not here
            variant,
            run_timestamp,
        )

    return {
        "variant":                 variant,
        "conversations_evaluated": conversations_done,
        "total_turns_evaluated":   len(all_scores),
        "turn_scores":             all_scores,
        "run_timestamp":           run_timestamp,
    }


# ------------------------------------------------------------------
# Compute corpus-level averages
# ------------------------------------------------------------------

def compute_averages(results: dict) -> dict:
    scores = results["turn_scores"]
    dims   = ["bleu", "rouge1", "rouge2", "rougeL", "meteor"]

    if not scores:
        print("No scores to average.")
        return {}

    print(f"\n{'='*55}")
    print(f"AVERAGES — {results['variant'].upper()}")
    print(f"Conversations : {results['conversations_evaluated']}")
    print(f"Turns         : {results['total_turns_evaluated']}")
    print(f"{'='*55}")

    averages = {}
    for dim in dims:
        vals = [s[dim] for s in scores if dim in s]
        avg  = round(mean(vals), 4) if vals else 0.0
        averages[dim] = avg
        print(f"  {dim.upper():<10}: {avg:.4f}")

    return averages


# ------------------------------------------------------------------
# Save results
# ------------------------------------------------------------------

def save_results(results: dict, averages: dict, variant: str, run_timestamp: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "variant":                 variant,
        "run_timestamp":           datetime.strptime(run_timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S"),
        "conversations_evaluated": results["conversations_evaluated"],
        "total_turns_evaluated":   results["total_turns_evaluated"],
        "averages":                averages,
        "turn_scores":             results["turn_scores"],
    }
    path = os.path.join(RESULTS_DIR, f"auto_metrics_{variant}_{run_timestamp}.json")  # compact in filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved -> {path}")


# ------------------------------------------------------------------
# Print comparison table when running all variants
# ------------------------------------------------------------------

def print_comparison_table(all_averages: dict):
    dims     = ["bleu", "rouge1", "rouge2", "rougeL", "meteor"]
    variants = list(all_averages.keys())

    print(f"\n{'='*70}")
    print("COMPARISON TABLE — All Variants")
    print(f"{'='*70}")
    print(f"{'Metric':<12}", end="")
    for v in variants:
        print(f"{v.upper():>14}", end="")
    print()
    print("-" * 70)
    for dim in dims:
        print(f"{dim.upper():<12}", end="")
        for v in variants:
            val = all_averages[v].get(dim, 0.0)
            print(f"{val:>14.4f}", end="")
        print()
    print(f"{'='*70}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=VARIANTS + ["all"],
        required=True,
        help="Variant to evaluate. Use 'all' to run all 4.",
    )
    args = parser.parse_args()

    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        print("Place edosthi_dataset_bak.json in the project root.")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    limit = MAX_CONVERSATIONS or len(dataset)
    print(f"Dataset   : {len(dataset)} conversations")
    print(f"Evaluating: {limit} conversations")

    variants_to_run = VARIANTS if args.variant == "all" else [args.variant]
    all_averages    = {}

    for variant in variants_to_run:
        results  = evaluate_variant(variant, dataset)
        averages = compute_averages(results)
        run_ts    = results["run_timestamp"]
        # run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(results, averages, variant, run_ts)
        all_averages[variant] = averages
        

    if len(variants_to_run) > 1:
        print_comparison_table(all_averages)


if __name__ == "__main__":
    main()