
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

history = []

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    prompt = """You are a helpful assistant.
    Use previous conversation if relevant.
    Do not make up facts about the user.

    """
    MAX_TURNS = 3
    recent_history = history[-MAX_TURNS:]

    for u, a in recent_history:
        prompt += f"User: {u}\nAssistant: {a}\n"

    prompt += f"User: {user_input}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.45,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )

    input_length = inputs["input_ids"].shape[-1]
    response = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)

    if "User:" in response:
        response = response.split("User:")[0].strip()

    print("Bot:", response)

    history.append((user_input, response))

