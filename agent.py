# agent.py
from llm import ask_llm
from tools import tools_list
from memory import retrieve_context
from langchain_core.prompts import PromptTemplate


# Custom LLM wrapper
class CustomLLM:
    def invoke(self, prompt):
        response = ask_llm(prompt)
        return {"content": response}


llm = CustomLLM()

prompt_template = PromptTemplate(
    input_variables=["memory", "message"],
    template="Relevant past memory:\n{memory}\n\nCurrent message: {message}"
)


def run_agent(user_input):
    """
    Zero-shot ReAct simulation:
    - Checks for tool keywords in your custom tools
    - Uses tool if matched
    - Otherwise falls back to LLM
    """
    # TOOL check
    for tool in tools_list:
        if tool.name.lower() in user_input.lower():
            # extract argument (after tool name)
            arg = user_input.lower().replace(tool.name.lower(), "").strip()
            result = tool.func(arg)
            return f"Answer (using tool {tool.name}): {result}"

    # fallback to LLM
    context = retrieve_context(user_input)
    final_prompt = prompt_template.invoke({
        "memory": context,
        "message": user_input
    }).text
    result = llm.invoke(final_prompt)
    return result["content"]











