# evaluation/session_metrics.py
"""
Session-Level Clinical Progression Metrics.

Computed directly from session log files — no LLM calls needed.
These metrics directly demonstrate the value of the phase system
by measuring clinical progression quality over entire sessions.

Key metrics:
  - information_completeness_rate   : % sessions where all 3 key KG fields collected
  - avg_turns_to_triggers           : how many turns before triggers first known
  - sessions_reaching_planning_rate : % sessions that reached Phase 4
  - premature_planning_rate         : % sessions where Planning started before triggers known
  - phase_distribution              : % of turns spent in each phase

Usage:
    python evaluation/session_metrics.py --logs_dir z_good_convs/full    --variant full
    python evaluation/session_metrics.py --logs_dir z_good_convs/no_phase --variant no_phase
    python evaluation/session_metrics.py --logs_dir z_good_convs/no_rag   --variant no_rag
    python evaluation/session_metrics.py --logs_dir z_good_convs/no_cot   --variant no_cot
    python evaluation/session_metrics.py --summary  # print comparison table after all 4 done
"""

import json
import os
import argparse
from statistics import mean


RESULTS_DIR = "evaluation/results"


# ------------------------------------------------------------------
# Load log files
# ------------------------------------------------------------------

def load_log_files(logs_dir: str) -> list:
    sessions = []
    for fname in sorted(os.listdir(logs_dir)):
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
# Phase distribution helper
# ------------------------------------------------------------------

def _phase_distribution(sessions: list) -> dict:
    counts = {}
    total  = 0
    for session in sessions:
        for turn in session["turns"]:
            p   = turn.get("step4_session_phase", {})
            num = p.get("phase_num")
            name = p.get("phase_name", f"phase_{num}")
            if num:
                key = f"{num}_{name}"
                counts[key] = counts.get(key, 0) + 1
                total += 1
    return {k: round(v / total, 2) for k, v in sorted(counts.items())} if total else {}


# ------------------------------------------------------------------
# Core computation
# ------------------------------------------------------------------

def compute_session_metrics(sessions: list, variant: str) -> dict:
    has_phase  = (variant != "no_phase")
    per_session = []

    for session in sessions:
        turns = session["turns"]

        smoking_turn    = None
        triggers_turn   = None
        motivation_turn = None
        planning_turn   = None
        phases_seen     = set()

        for turn in turns:
            n  = turn["turn"]
            kg = turn.get("step3_kg_subgraph", {})

            if smoking_turn is None and kg.get("smoking_status"):
                smoking_turn = n

            if triggers_turn is None and kg.get("relevant_triggers"):
                triggers_turn = n

            if motivation_turn is None:
                mr = kg.get("motivation_reason")
                if (isinstance(mr, list) and mr) or (isinstance(mr, str) and mr):
                    motivation_turn = n

            if has_phase:
                phase_num = turn.get("step4_session_phase", {}).get("phase_num")
                if phase_num:
                    phases_seen.add(phase_num)
                    if phase_num >= 4 and planning_turn is None:
                        planning_turn = n

        all_collected = all([smoking_turn, triggers_turn, motivation_turn])

        # Premature planning: reached Phase 4 before triggers were known
        premature = False
        if has_phase and planning_turn is not None:
            if triggers_turn is None or planning_turn < triggers_turn:
                premature = True

        per_session.append({
            "patient_id":          session["patient_id"],
            "total_turns":         len(turns),
            "smoking_turn":        smoking_turn,
            "triggers_turn":       triggers_turn,
            "motivation_turn":     motivation_turn,
            "all_fields_collected": all_collected,
            "planning_turn":       planning_turn   if has_phase else None,
            "reached_planning":    planning_turn is not None if has_phase else None,
            "premature_planning":  premature       if has_phase else None,
            "phases_seen":         sorted(phases_seen) if has_phase else None,
        })

    n = len(per_session)
    if n == 0:
        return {}

    def _safe_mean(vals):
        vals = [v for v in vals if v is not None]
        return round(mean(vals), 2) if vals else None

    metrics = {
        "variant":                       variant,
        "total_sessions":                n,
        "information_completeness_rate": round(
            sum(1 for s in per_session if s["all_fields_collected"]) / n, 2
        ),
        "avg_turns_to_smoking_status":   _safe_mean(
            [s["smoking_turn"] for s in per_session]
        ),
        "avg_turns_to_triggers":         _safe_mean(
            [s["triggers_turn"] for s in per_session]
        ),
        "avg_turns_to_motivation":       _safe_mean(
            [s["motivation_turn"] for s in per_session]
        ),
    }

    if has_phase:
        reaching = [s for s in per_session if s["reached_planning"]]
        metrics.update({
            "sessions_reaching_planning_rate": round(len(reaching) / n, 2),
            "sessions_reaching_planning_count": len(reaching),
            "avg_turn_planning_reached":        _safe_mean(
                [s["planning_turn"] for s in reaching]
            ),
            "premature_planning_rate":          round(
                sum(1 for s in per_session if s["premature_planning"]) / n, 2
            ),
            "premature_planning_count":         sum(
                1 for s in per_session if s["premature_planning"]
            ),
            "phase_distribution":               _phase_distribution(sessions),
        })
    else:
        # No-Phase — no phase metrics but track premature strategy mentions
        metrics.update({
            "sessions_reaching_planning_rate": "N/A",
            "premature_planning_rate":         "N/A",
            "phase_distribution":              "N/A",
        })

    metrics["per_session"] = per_session
    return metrics


# ------------------------------------------------------------------
# Print metrics
# ------------------------------------------------------------------

def print_metrics(metrics: dict):
    variant = metrics.get("variant", "unknown").upper()
    print(f"\n{'='*55}")
    print(f"SESSION METRICS — {variant}")
    print(f"{'='*55}")
    skip = {"per_session", "variant", "phase_distribution"}
    for k, v in metrics.items():
        if k in skip:
            continue
        print(f"  {k:<42}: {v}")

    if metrics.get("phase_distribution") and metrics["phase_distribution"] != "N/A":
        print(f"\n  Phase Distribution:")
        for phase, pct in metrics["phase_distribution"].items():
            bar = "█" * int(pct * 30)
            print(f"    {phase:<20}: {pct:.0%}  {bar}")


# ------------------------------------------------------------------
# Save metrics
# ------------------------------------------------------------------

def save_metrics(metrics: dict, variant: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"session_metrics_{variant}.json")
    # Remove per_session from saved file to keep it readable
    save_data = {k: v for k, v in metrics.items() if k != "per_session"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved → {path}")


# ------------------------------------------------------------------
# Comparison table
# ------------------------------------------------------------------

def print_comparison_table():
    variants = ["full", "no_phase", "no_rag", "no_cot"]
    all_metrics = {}

    for variant in variants:
        path = os.path.join(RESULTS_DIR, f"session_metrics_{variant}.json")
        if not os.path.exists(path):
            print(f"  session_metrics_{variant}.json not found — run that variant first")
            continue
        with open(path) as f:
            all_metrics[variant] = json.load(f)

    if not all_metrics:
        print("No metrics found. Run all variants first.")
        return

    print(f"\n{'='*85}")
    print("SESSION METRICS COMPARISON TABLE")
    print(f"{'='*85}")

    metrics_to_show = [
        ("total_sessions",                   "Total Sessions"),
        ("information_completeness_rate",     "Info Completeness Rate"),
        ("avg_turns_to_smoking_status",       "Avg Turns → Smoking Status"),
        ("avg_turns_to_triggers",             "Avg Turns → Triggers"),
        ("avg_turns_to_motivation",           "Avg Turns → Motivation"),
        ("sessions_reaching_planning_rate",   "Sessions Reaching Planning"),
        ("avg_turn_planning_reached",         "Avg Turn Planning Reached"),
        ("premature_planning_rate",           "Premature Planning Rate"),
    ]

    # Header
    print(f"\n{'Metric':<35}", end="")
    for v in variants:
        if v in all_metrics:
            print(f"{v.upper():>12}", end="")
    print()
    print("-" * 85)

    for key, label in metrics_to_show:
        print(f"{label:<35}", end="")
        for v in variants:
            if v not in all_metrics:
                continue
            val = all_metrics[v].get(key, "—")
            if isinstance(val, float):
                print(f"{val:>12.2f}", end="")
            else:
                print(f"{str(val):>12}", end="")
        print()

    print(f"{'='*85}")

    # Phase distribution separately
    print(f"\nPhase Distribution (% of turns in each phase):")
    print(f"{'Phase':<25}", end="")
    for v in variants:
        if v in all_metrics:
            print(f"{v.upper():>12}", end="")
    print()
    print("-" * 70)

    # Collect all phase keys
    all_phases = set()
    for v, m in all_metrics.items():
        pd = m.get("phase_distribution", {})
        if isinstance(pd, dict):
            all_phases.update(pd.keys())

    for phase in sorted(all_phases):
        print(f"{phase:<25}", end="")
        for v in variants:
            if v not in all_metrics:
                continue
            pd  = all_metrics[v].get("phase_distribution", {})
            val = pd.get(phase, 0.0) if isinstance(pd, dict) else "N/A"
            if isinstance(val, float):
                print(f"{val:>12.0%}", end="")
            else:
                print(f"{str(val):>12}", end="")
        print()

    print(f"{'='*85}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", help="Directory with session log JSON files")
    parser.add_argument("--variant",  choices=["full", "no_phase", "no_rag", "no_cot"])
    parser.add_argument("--summary",  action="store_true", help="Print comparison table")
    args = parser.parse_args()

    if args.summary:
        print_comparison_table()
        return

    if not args.logs_dir or not args.variant:
        print("ERROR: --logs_dir and --variant required (or use --summary)")
        return

    if not os.path.exists(args.logs_dir):
        print(f"ERROR: {args.logs_dir} not found")
        return

    sessions = load_log_files(args.logs_dir)
    print(f"Loaded {len(sessions)} sessions from {args.logs_dir}")

    if not sessions:
        print("No session log files found.")
        return

    metrics = compute_session_metrics(sessions, args.variant)
    print_metrics(metrics)
    save_metrics(metrics, args.variant)


if __name__ == "__main__":
    main()