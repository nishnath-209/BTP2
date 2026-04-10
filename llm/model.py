# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# import torch

# # MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
# # MODEL_NAME = "google/flan-t5-base"

# device = "cuda" if torch.cuda.is_available() else "cpu"
# # device = "cpu"
# print("Running on:", device)

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# # model = AutoModelForSeq2SeqLM.from_pretrained(
# #     MODEL_NAME
# # ).to(device)

# bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",          
#         bnb_4bit_use_double_quant=True,      
#         bnb_4bit_compute_dtype=torch.float16 
#     )


# # model = AutoModelForCausalLM.from_pretrained(
# #         MODEL_NAME,
# #         quantization_config=bnb_config,
# #         device_map="auto",   
# #     )


# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True,
#     attn_implementation="eager"
# )

# # def generate(prompt):

# #     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# #     output = model.generate(
# #         **inputs,
# #         max_new_tokens=200,
# #         temperature=0.7
# #     )

# #     response = tokenizer.decode(output[0], skip_special_tokens=True)

# #     return response

# def generate(prompt: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
#     """
#     Generate a response from the LLM.
#     pipeline calls this
#     """
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     output = model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         temperature=temperature,
#         do_sample=True,
#         use_cache=True,
#         repetition_penalty=1.1,
#     )

#     # Decode only the newly generated tokens (not the prompt)
#     new_tokens = output[0][inputs["input_ids"].shape[1]:]
#     response = tokenizer.decode(new_tokens, skip_special_tokens=True)

#     return response.strip()


from openai import OpenAI
import os

MODEL_NAME = "llama-3.1-8b-instant"

client = OpenAI(
    api_key="",
    base_url="https://api.groq.com/openai/v1"
)

print("Using API model:", MODEL_NAME)


def generate(prompt: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
    """
    Generate a response from the LLM
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            # {
            #     "role": "system",
            #     "content": "You are a warm, empathetic therapist helping patients overcome tobacco addiction."
            # },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_new_tokens
    )

    return response.choices[0].message.content.strip()