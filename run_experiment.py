"""
run_experiment.py

Entry point for the therapy chatbot.
Run: python run_experiment.py
"""

from pipeline.therapy_pipeline import therapy_chat

PATIENT_ID = "patient_013"
SESSION_ID = 1  # Keep the same value to resume; change to start a new session

print("Type 'exit' to quit\n")

while True:
    msg = input("Patient: ").strip()

    if not msg:
        continue

    if msg.lower() == "exit":
        print("Session ended.")
        break

    response = therapy_chat(msg, patient_id=PATIENT_ID, session_id=SESSION_ID)

    print(f"\nTherapist: {response}\n")