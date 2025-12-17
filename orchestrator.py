# orchestrator.py
from multi_agents import research_agent, summarizer_agent
from vectordb import add_memory, get_shared_memory  # <- change this name


# LangChain-style shared memory over FAISS
shared_memory = get_shared_memory()  # <- stays the same logically


def run_multi_agent_pipeline(user_query: str) -> str:
    """
    Orchestrates multi-agent collaboration:
    1) Research agent gathers key points.
    2) Result is stored in shared memory.
    3) Summarizer agent produces final answer.
    4) Final answer stored again in shared memory.
    """
    # Step 1: research
    research_out = research_agent(user_query)
    add_memory("RESEARCH RESULT: " + research_out)

    # also store in VectorStoreRetrieverMemory
    shared_memory.save_context(
        {"input": user_query},
        {"output": research_out},
    )

    # Step 2: summarize
    summary_out = summarizer_agent(user_query, research_out)
    add_memory("FINAL SUMMARY: " + summary_out)

    shared_memory.save_context(
        {"input": "SUMMARY FOR: " + user_query},
        {"output": summary_out},
    )

    return summary_out


