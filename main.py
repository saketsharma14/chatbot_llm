from model_loader import load_model
from memory import Memory
from prompt_builder import build_prompt
from chatbot import Chatbot

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading model...")
tokenizer, model = load_model(model_name)
print("Model loaded!")

memory = Memory(max_turns=3)

bot = Chatbot(model, tokenizer, memory, build_prompt)

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = bot.generate_response(user_input)

    print("Bot:", response)
