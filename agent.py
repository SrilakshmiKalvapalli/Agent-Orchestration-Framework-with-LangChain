from llm import ask_llm
from typing import List

# --- Tool: multiply two numbers ---
def multiply_numbers(text: str):
    try:
        a, b = map(int, text.split(","))
        return a * b
    except Exception:
        return "Invalid input. Use format: a,b"

TOOLS = {
    "multiply": multiply_numbers
}

# --- Run Zero-Shot style agent ---
def run_agent(question: str):
    """
    Zero-Shot React agent simulation for LangChain 1.1.0.
    - Looks for a tool keyword in question
    - If found, uses the tool
    - Else, uses ask_llm
    """
    # Simple keyword detection
    if "multiply" in question.lower():
        # extract numbers
        numbers = "".join(c if c.isdigit() or c=="," else "" for c in question)
        result = TOOLS["multiply"](numbers)
        return f"Answer (using tool): {result}"
    else:
        # fallback to LLM
        return ask_llm(question)





