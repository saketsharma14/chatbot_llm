def build_prompt(history, user_input):
    prompt = """You are a helpful assistant.

Answer the user's question directly.
Use previous conversation if relevant.
Do not make up facts about the user.

"""

    for u, a in history:
        prompt += f"User: {u}\nAssistant: {a}\n"

    prompt += f"User: {user_input}\nAssistant:"

    return prompt
