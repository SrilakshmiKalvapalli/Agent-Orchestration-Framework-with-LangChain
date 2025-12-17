# main.py
from llm import ask_llm
from memory import store_user_message, store_agent_message, retrieve_context
from langchain_core.prompts import PromptTemplate
from tools import tools_list, create_tool_agent
from agent import run_agent
from orchestrator import run_multi_agent_pipeline


def multiply_numbers(text: str):
    try:
        a, b = map(int, text.split(","))
        return a * b
    except Exception:
        return "Invalid input. Use format: a,b"


TOOLS = {
    "multiply": multiply_numbers,
}


class CustomLLM:
    def invoke(self, prompt):
        response = ask_llm(prompt)
        return {"content": response}


llm = CustomLLM()

prompt = PromptTemplate(
    input_variables=["memory", "message"],
    template="Relevant past memory:\n{memory}\n\nCurrent message: {message}",
)

print("Type 'exit' to quit the chat.")
print("Type 'research: <your topic>' to use the multi-agent pipeline.")
print("Type 'tools: <your question>' to use the LangChain tool agent.\n")

tool_agent = create_tool_agent()

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # -------- Milestone 3: multi-agent mode --------
    if user_input.lower().startswith("research:"):
        query = user_input[len("research:"):].strip()
        store_user_message(query)
        agent_result = run_multi_agent_pipeline(query)
        store_agent_message(agent_result)
        print("Assistant (multi-agent):", agent_result)
        continue

    # -------- Milestone 2: LangChain tool agent mode --------
    if user_input.lower().startswith("tools:"):
        query = user_input[len("tools:"):].strip()
        store_user_message(query)
        try:
            agent_result = tool_agent.run(query)
        except Exception as e:
            agent_result = f"Agent Error: {str(e)}"
        store_agent_message(agent_result)
        print("Assistant (tool-agent):", agent_result)
        continue

    # -------- Milestone 1: multiply tool --------
    if "multiply" in user_input.lower():
        numbers = "".join(c if c.isdigit() or c == "," else "" for c in user_input)
        result = TOOLS["multiply"](numbers)
        agent_result = f"Answer (using tool multiply): {result}"
        store_user_message(user_input)
        store_agent_message(agent_result)
        print("Assistant:", agent_result)
        continue

    # -------- Milestone 2: custom tools (string matching) --------
    for tool in tools_list:
        if tool.name.lower() in user_input.lower():
            arg = user_input.lower().replace(tool.name.lower(), "").strip()
            result = tool.func(arg)
            agent_result = f"Answer (using tool {tool.name}): {result}"
            store_user_message(user_input)
            store_agent_message(agent_result)
            print("Assistant:", agent_result)
            break
    else:
        # -------- Fallback: LLM with long-term memory --------
        store_user_message(user_input)
        context = retrieve_context(user_input)
        final_prompt = prompt.invoke({
            "memory": context,
            "message": user_input,
        }).text
        result = llm.invoke(final_prompt)
        agent_result = result["content"]
        store_agent_message(agent_result)
        print("Assistant:", agent_result)









