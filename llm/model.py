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



# def generate(prompt):

#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     output = model.generate(
#         **inputs,
#         max_new_tokens=200,
#         temperature=0.7
#     )

#     response = tokenizer.decode(output[0], skip_special_tokens=True)

#     return response

def generate(prompt: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
    """
    Generate a response from the LLM.
    pipeline calls this
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )

    # Decode only the newly generated tokens (not the prompt)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip()