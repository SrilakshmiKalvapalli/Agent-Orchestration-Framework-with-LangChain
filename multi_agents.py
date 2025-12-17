# multi_agents.py
from llm import ask_llm
from memory import retrieve_context, store_user_message, store_agent_message
from langchain.memory import ConversationBufferMemory

# Short‑term memories for each agent
research_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

summary_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)


def research_agent(user_query: str) -> str:
    """Agent 1: collects key facts or ideas about the query."""
    long_term = retrieve_context(user_query)

    prompt = f"""
You are a research agent.
User question: {user_query}

Relevant past memory:
{long_term}

Provide 5–7 bullet points of key facts or ideas that will help answer this question.
"""
    reply = ask_llm(prompt)

    # store in this agent's short‑term memory
    research_memory.save_context({"input": user_query}, {"output": reply})

    # also store in long‑term vector memory
    store_user_message(user_query)
    store_agent_message("RESEARCH: " + reply)

    return reply


def summarizer_agent(user_query: str, research_notes: str) -> str:
    """Agent 2: summarizes research into a final response."""
    long_term = retrieve_context(user_query)

    prompt = f"""
You are a summarization agent helping a trainer.
User question: {user_query}

Research notes:
{research_notes}

Relevant long-term memory:
{long_term}

Write a clear, well-structured answer in 2–3 short paragraphs.
"""
    reply = ask_llm(prompt)

    summary_memory.save_context(
        {"input": user_query + "\n" + research_notes},
        {"output": reply},
    )
    store_agent_message("SUMMARY: " + reply)

    return reply

