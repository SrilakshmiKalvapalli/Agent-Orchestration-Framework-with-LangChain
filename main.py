# main.py
from llm import ask_llm
from memory import store_user_message, store_agent_message, retrieve_context
from langchain_core.prompts import PromptTemplate
from tools import tools_list  # Milestone 2 tools: calculator, weather

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
    - Check for Milestone 1 multiply tool
    - Check for Milestone 2 tools (calculator, weather)
    - Else fallback to LLM
    """
    # --------------------
    # Milestone 1: multiply
    # --------------------
    if "multiply" in user_input.lower():
        numbers = "".join(c if c.isdigit() or c=="," else "" for c in user_input)
        result = TOOLS["multiply"](numbers)
        return f"Answer (using tool multiply): {result}"

    # --------------------
    # Milestone 2: custom tools
    # --------------------
    for tool in tools_list:
        if tool.name.lower() in user_input.lower():
            # extract argument after tool name
            arg = user_input.lower().replace(tool.name.lower(), "").strip()
            result = tool.func(arg)
            return f"Answer (using tool {tool.name}): {result}"

    # --------------------
    # Fallback: LLM
    # --------------------
    context = retrieve_context(user_input)
    final_prompt = prompt.invoke({
        "memory": context,
        "message": user_input
    }).text
    result = llm.invoke(final_prompt)
    return result["content"]

# --------------------
# Console loop
# --------------------
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    store_user_message(user_input)
    agent_result = run_agent(user_input)
    store_agent_message(agent_result)

    print("Assistant:", agent_result)








