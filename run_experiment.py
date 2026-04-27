"""
run_experiment.py

Entry point for the therapy chatbot.
Run: python run_experiment.py

"""

import os
import pipeline.therapy_pipeline as pipeline_module
import logger.conversation_history as conv_history_module
from logger.session_logger import SessionLogger
from pipeline.therapy_pipeline import therapy_chat

# ------------------------------------------------------------------
# Config — change these two lines before each variant run
# ------------------------------------------------------------------

VARIANT    = "no_phase"           # "full" / "no_phase" / "no_rag" / "no_cot"
PATIENT_ID = "manual_p_02"      # change for each new conversation
SESSION_ID = 1

# ------------------------------------------------------------------
# Override paths so logs go to the right variant folder
# ------------------------------------------------------------------

LOG_DIR = f"manual_sessions/{VARIANT}"
KG_DIR  = f"manual_sessions/kg/{VARIANT}"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(KG_DIR,  exist_ok=True)

pipeline_module.logger           = SessionLogger(log_dir=LOG_DIR)
pipeline_module.kg.storage_path  = f"{KG_DIR}/patient_profiles.json"
pipeline_module.kg.load()
# if PATIENT_ID in pipeline_module.kg.profiles:
#     del pipeline_module.kg.profiles[PATIENT_ID]
conv_history_module.HISTORY_FILE = f"{LOG_DIR}/conversation_history.json"

# ------------------------------------------------------------------
# Chat loop
# ------------------------------------------------------------------

print(f"\nVariant : {VARIANT.upper()}")
print(f"Patient : {PATIENT_ID}")
print(f"Logs    : {LOG_DIR}/")
print("Type 'exit' to end session\n")

while True:
    msg = input("Patient: ").strip()

    if not msg:
        continue

    if msg.lower() == "exit":
        print("Session ended.")
        break

    response = therapy_chat(msg, patient_id=PATIENT_ID, session_id=SESSION_ID)
    print(f"\nTherapist: {response}\n")