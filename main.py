from llm import ask_llm
from memory import store_user_message, store_agent_message, retrieve_context
from langchain_core.prompts import PromptTemplate


def multiply_numbers(text: str):
    try:
        a, b = map(int, text.split(","))
        return a * b
    except Exception:
        return "Invalid input. Use format: a,b"

TOOLS = {
    "multiply": multiply_numbers
}

class CustomLLM:
    def invoke(self, prompt):
        response = ask_llm(prompt)
        return {"content": response}

llm = CustomLLM()


prompt = PromptTemplate(
    input_variables=["memory", "message"],
    template="Relevant past memory:\n{memory}\n\nCurrent message: {message}"
)

print("Type 'exit' to quit the chat.\n")


def run_agent(user_input):
    """
    Simulate zero-shot ReAct:
    - Check for tool keywords
    - Use tool if matched
    - Else fallback to LLM
    """
    if "multiply" in user_input.lower():
        
        numbers = "".join(c if c.isdigit() or c=="," else "" for c in user_input)
        result = TOOLS["multiply"](numbers)
        return f"Answer (using tool): {result}"
    else:
  
        context = retrieve_context(user_input)
        final_prompt = prompt.invoke({
            "memory": context,
            "message": user_input
        }).text
        result = llm.invoke(final_prompt)
        return result["content"]

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break


    store_user_message(user_input)

  
    agent_result = run_agent(user_input)

    
    store_agent_message(agent_result)

    print("Assistant:", agent_result)







