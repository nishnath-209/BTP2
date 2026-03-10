from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# MODEL_NAME = "google/flan-t5-base"

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("Running on:", device)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# model = AutoModelForSeq2SeqLM.from_pretrained(
#     MODEL_NAME
# ).to(device)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)



def generate(prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response