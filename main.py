import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("loading model..")

tokenizer=AutoTokenizer.from_pretrained(model_name)

model=AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded!")

while True:
    user_input=input("You: ")
    system_prompt="You are a helpful assistant."
    prompt=f"{system_prompt}\nUser:{user_input}\nAssistant:"

    if user_input.lower()=="exit":
        break

    inputs=tokenizer(prompt,return_tensors="pt").to(model.device)

    output=model.generate(
        **inputs,
        max_new_tokens=100
    )

    response=tokenizer.decode(output[0],skip_special_tokens=True)
    print("response:",response)
    # print("Bot: ", response)
