
from llm import ask_llm
from memory import store_user_message, store_agent_message, retrieve_context

# Correct imports for LangChain 1.1.0
from langchain_core.prompts import PromptTemplate

class CustomLLM:
    def invoke(self, prompt):
        response = ask_llm(prompt)
        return {"content": response}  


prompt = PromptTemplate(
    input_variables=["memory", "message"],
    template="Relevant past memory:\n{memory}\n\nCurrent message: {message}"
)


llm = CustomLLM()

print("Type 'exit' to quit the chat.")

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

   
    store_user_message(user_input)

 
    context = retrieve_context(user_input)

    
    final_prompt = prompt.invoke({
        "memory": context,
        "message": user_input
    }).text

    
    result = llm.invoke(final_prompt)
    answer = result["content"]

    store_agent_message(answer)

    print("\nAssistant:", answer)






