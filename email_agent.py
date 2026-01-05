from llm import ask_llm
from memory import retrieve_context, store_agent_message
from langchain.memory import ConversationBufferMemory

# Short-term memory for email agent
email_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

def email_agent(user_query: str, research_notes: str, summary: str) -> str:
    """
    Agent 3: Composes professional email from research + summary.
    Week 7 Task 1: Email composition agent.
    """
    long_term = retrieve_context(user_query)
    
    prompt = f"""
You are an Email Composer Agent. Your job is to create professional emails.

User original question: {user_query}

Research findings:
{research_notes}

Summary:
{summary}

Relevant long-term memory:
{long_term}

TASK: Write a complete professional email including:
1. Subject line (clear and specific)
2. Greeting (Dear [Name/Team],)
3. 3-4 paragraph body explaining key points and recommendations
4. Professional closing (Best regards, Your Name)
5. Suggested recipient email (if not provided)

Format your response as JSON:
{{
    "subject": "Your subject here",
    "recipient": "suggested@email.com", 
    "body": "Full email body here..."
}}
"""
    
    reply = ask_llm(prompt)
    
    # Store in short-term memory
    email_memory.save_context(
        {"input": f"{user_query}\nResearch: {research_notes}\nSummary: {summary}"},
        {"output": reply}
    )
    
    # Store in long-term memory
    store_agent_message("EMAIL: " + reply)
    
    return reply
