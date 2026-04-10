import json


DATA_FOLDER = "p/data"
RAG_FOLDER = "p/rag"
SCRIPTS_FOLDER = "p/scripts"
LLM_FOLDER = "p/llm"
PIPELINE_FOLDER = "p/pipeline"

INPUT_FILE =  DATA_FOLDER + "/therapy_sessions.json"
OUTPUT_FILE = RAG_FOLDER + "/rag_data.txt"

knowledge = []

with open(INPUT_FILE) as f:
    sessions = json.load(f)

for session in sessions:
    for msg in session:

        if msg["role"] == "Therapist":

            text = msg["content"].strip()

            if len(text.split()) > 12:
                knowledge.append(text)

with open(OUTPUT_FILE, "w") as f:
    for k in knowledge:
        f.write(k + "\n\n")

print("Extracted", len(knowledge), "therapy knowledge chunks")