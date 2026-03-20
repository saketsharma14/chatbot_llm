from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model
