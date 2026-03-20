# Local LLM Chatbot (Offline)

A fully offline chatbot built using an open-source Large Language Model (LLM) with multi-turn conversation support. This project demonstrates how to load, run, and interact with a local LLM using a modular and scalable architecture.

---

## Features

* Runs completely offline (no external APIs)
* Multi-turn conversation support (short-term memory)
* Modular architecture (clean separation of concerns)
* Prompt-based control of model behavior
* Built using Hugging Face Transformers and PyTorch

---

## Project Structure

```
.
├── main.py              # Entry point (CLI interface)
├── chatbot.py          # Core chatbot logic
├── model_loader.py     # Loads model and tokenizer
├── memory.py           # Handles conversation history
├── prompt_builder.py   # Builds structured prompts
└── README.md
```

---

## How It Works

The chatbot follows a simple pipeline:

```
User Input
   ↓
Prompt Builder (adds history + instructions)
   ↓
Tokenizer (text → tokens)
   ↓
LLM Inference (generation)
   ↓
Decoder (tokens → text)
   ↓
Response Output
```

### Key Components

* Model Loading
  Loads a local LLM using Hugging Face Transformers.

* Memory Module
  Stores recent conversation history (sliding window).

* Prompt Builder
  Structures input into a chat format:

  ```
  User: ...
  Assistant: ...
  ```

* Chatbot Engine
  Orchestrates tokenization, generation, and response handling.

---

## Example Interaction

```
You: My name is Paul
Bot: Nice to meet you, Paul.

You: What is my name?
Bot: Your name is Paul.
```

---

## Run the chatbot

```
python main.py
```

Type `exit` to quit.

---

## Model Used

* TinyLlama-1.1B-Chat (lightweight, runs locally)

---

## Limitations

* Small models may:

  * hallucinate occasionally
  * forget older context
  * generate inconsistent responses

* Memory is limited to recent turns (sliding window)

---

## Key Learnings

* How LLMs perform autoregressive text generation
* Importance of prompt structure in controlling behavior
* Handling context and memory in chat systems
* Limitations of small local models

---
