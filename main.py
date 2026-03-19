
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded!")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    prompt = f"User: {user_input}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=100
    )

    input_length=inputs["input_ids"].shape[-1]
    response = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)

    print("Bot:", response)
