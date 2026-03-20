import torch

class Chatbot:
    def __init__(self, model, tokenizer, memory, prompt_builder):
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        self.prompt_builder = prompt_builder

    def generate_response(self, user_input):
        history = self.memory.get_recent()

        prompt = self.prompt_builder(history, user_input)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id
        )

        input_length = inputs["input_ids"].shape[-1]
        response = self.tokenizer.decode(
            output[0][input_length:], skip_special_tokens=True
        )

        if "User:" in response:
            response = response.split("User:")[0].strip()

        self.memory.add(user_input, response)

        return response
