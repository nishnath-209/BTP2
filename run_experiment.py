"""
run_experiment.py

Entry point for the therapy chatbot.
Run: python run_experiment.py
"""

from pipeline.therapy_pipeline import therapy_chat

PATIENT_ID = "patient_011"

print("Type 'exit' to quit\n")

while True:
    msg = input("Patient: ").strip()

    if not msg:
        continue

    if msg.lower() == "exit":
        print("Session ended.")
        break

    response = therapy_chat(msg, patient_id=PATIENT_ID)

    print(f"\nTherapist: {response}\n")