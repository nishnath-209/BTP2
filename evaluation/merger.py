import json
from statistics import mean

file1 = "evaluation/results/auto_metrics_full_20260412_160000.json"  # your actual filenames
file2 = "evaluation/results/auto_metrics_full_20260412_180000.json"

with open(file1) as f: run1 = json.load(f)
with open(file2) as f: run2 = json.load(f)

# Merge turn scores
combined_turns = run1["turn_scores"] + run2["turn_scores"]

# Recompute averages over all turns
dims = ["bleu", "rouge1", "rouge2", "rougeL", "meteor"]
averages = {}
for dim in dims:
    vals = [s[dim] for s in combined_turns if dim in s]
    averages[dim] = round(mean(vals), 4)

merged = {
    "variant": "full",
    "conversations_evaluated": run1["conversations_evaluated"] + run2["conversations_evaluated"],
    "total_turns_evaluated": len(combined_turns),
    "averages": averages,
    "turn_scores": combined_turns,
}

with open("evaluation/results/auto_metrics_full_merged.json", "w") as f:
    json.dump(merged, f, indent=2)

print("Merged. Total turns:", len(combined_turns))
print("Averages:", averages)