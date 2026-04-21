# evaluation/

This folder contains both evaluation scripts for the thesis.

---

## Files

```
evaluation/
├── evaluate.py            ← LLM-as-a-Judge (6 dimensions, 4 variants)
├── automatic_metrics.py   ← BLEU / ROUGE / METEOR (EDosthi dataset)
├── results/               ← All output JSON files saved here (auto-created)
│   ├── llm_judge_full.json
│   ├── llm_judge_no_phase.json
│   ├── llm_judge_no_rag.json
│   ├── llm_judge_no_cot.json
│   ├── auto_metrics_full.json
│   ├── auto_metrics_no_phase.json
│   ├── auto_metrics_no_rag.json
│   └── auto_metrics_no_cot.json
```

---

## 1. LLM-as-a-Judge  (`evaluate.py`)

Reads your existing session log files. Sends each therapist response to GPT-4o
for scoring on 6 clinical dimensions. Evaluation is **blind** — judge never
knows which variant produced the response.

### Setup

```bash
export OPENAI_API_KEY="your-key-here"
```

### Run

```bash
# Full system — reads from logs/
python evaluation/evaluate.py --variant full --logs_dir logs

# Ablated variants — point to their respective log folders
python evaluation/evaluate.py --variant no_phase --logs_dir logs_no_phase
python evaluation/evaluate.py --variant no_rag   --logs_dir logs_no_rag
python evaluation/evaluate.py --variant no_cot   --logs_dir logs_no_cot
```

### Dimensions scored (1–5)

| # | Dimension             | All variants? |
|---|-----------------------|---------------|
| 1 | Style Compliance      | ✅            |
| 2 | Phase Appropriateness | ❌ No-Phase   |
| 3 | Empathy & Warmth      | ✅            |
| 4 | Safety                | ✅            |
| 5 | MI Fidelity           | ✅            |
| 6 | Overall Therapeutic Value | ✅        |

---

## 2. Automatic Metrics  (`automatic_metrics.py`)

Uses `edosthi_dataset_bak.json` as reference. For each patient message in the
dataset, generates a response from your system and compares it against the
reference provider response using BLEU, ROUGE-1, ROUGE-2, ROUGE-L, METEOR.

### Setup

```bash
pip install rouge-score nltk --break-system-packages
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

Make sure `edosthi_dataset_bak.json` is in the project root.

### Run

```bash
# Single variant
python evaluation/automatic_metrics.py --variant full
python evaluation/automatic_metrics.py --variant no_phase
python evaluation/automatic_metrics.py --variant no_rag
python evaluation/automatic_metrics.py --variant no_cot

# All 4 variants + comparison table
python evaluation/automatic_metrics.py --variant all
```

### Controlling how many conversations to evaluate

Open `automatic_metrics.py` and change:
```python
MAX_CONVERSATIONS = 20   # quick test
MAX_CONVERSATIONS = None  # all 100
```

---

## Note on Low BLEU/ROUGE Scores

BLEU and ROUGE measure word overlap. In open-ended therapy dialogue, two
responses can be clinically correct but use different words, so scores
will naturally be low. This is a known limitation discussed in:

> Reiter (2018) — "A Structured Review of the Validity of BLEU"

The cross-variant comparison (Full vs No-Phase vs No-RAG vs No-CoT) is
what matters — not the absolute values. Acknowledge this in your thesis.