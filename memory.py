
from vectordb import add_memory, search_memory

def store_user_message(text):
    add_memory("User said: " + text)

def store_agent_message(text):
    add_memory("Assistant replied: " + text)

def retrieve_context(query):
    memories = search_memory(query)
    if memories:
        return "\n".join(memories)
    return "No relevant memory."










