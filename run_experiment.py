from pipeline.therapy_pipeline import therapy_chat

print("\n Here\n")

while True:

    msg = input("Patient: ")

    if msg.lower() == "exit":
        break

    response = therapy_chat(msg)

    print("\nTherapist:", response)
    print()