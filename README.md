I built a conversational agent using Python 3.13.4 and LangChain 1.1.0, connecting it to the OpenAI language model. The agent can respond to user queries through a console interface, storing inputs and retrieving relevant context from memory implemented with a FAISS vector database. Prompt templates are used to provide context-aware responses, and a simple tool-based action, such as multiplication, is implemented to simulate reasoning and action within the conversation.
 
In this part of the project, I improved my agent by adding support for custom tools and making it capable of performing actions based on the user’s message. I created separate tool functions like a calculator and a simulated weather checker, and integrated them so the agent can detect tool keywords, run the correct tool, and return the result automatically. The agent now handles tool responses, fallback LLM responses, and also manages errors gracefully. This update makes the agent more interactive, flexible, and capable of combining normal conversation with tool-based operations

In this milestone, I extended the project into a multi‑agent system with memory-aware reasoning. I created separate research and summarizer agents that work together in a pipeline: the research agent gathers key points for a query, and the summarizer agent turns those notes into a clear final answer. Both agents use short‑term conversational memory to keep track of their own interactions, and they share a long‑term memory built on a FAISS vector store so past queries and answers can influence future responses. This orchestration makes the assistant better at handling multi-step tasks, reusing previous knowledge, and producing more consistent, context-aware outputs.

This project implements a multi-step AI workflow that automates a “research → summarize → compose email” task using multiple specialized agents coordinated through a FastAPI backend. The backend exposes REST endpoints to trigger both single-agent queries and the full workflow, while a simple HTML/JavaScript interface allows users to submit prompts, view results, and test individual tools from the browser. The system includes shared memory, custom tools, and basic evaluation of workflow accuracy and interaction quality, along with cleaned-up code and documentation so the entire pipeline—from API to UI—can be easily set up, understood, and extended.


                         PROJECT ARCHITECTURE

                         ┌──────────────────────┐
                         │      Web Frontend    │
                         │       (HTML/JS)      │
                         └──────────┬───────────┘
                                    │ HTTP (JSON)
                                    │
                                    ▼
                        ┌──────────────────────────┐
                        │      FastAPI Backend     │
                        │  /agent/run, /workflow   │
                        └──────────┬──────────────┘
                                   │
                                   │ Orchestration
                                   ▼
                      ┌──────────────────────────────┐
                      │     Orchestration Engine     │
                      │  Research → Summarize → Mail │
                      └──────────┬──────────┬───────┘
                                 │          │
                                 │          │
                                 │          │
                 ┌───────────────┘          └───────────────┐
                 │                                          │
                 ▼                                          ▼
        ┌───────────────────┐                      ┌───────────────────┐
        │   Research Agent  │                      │  Email Agent      │
        └─────────┬─────────┘                      └─────────┬─────────┘
                  │                                           
                  │                                           
                  ▼                                           
        ┌───────────────────┐                      
        │ Summarizer Agent  │
        └─────────┬─────────┘
                  │
                  │ uses tools & memory
                  ▼
        ┌───────────────────┐      ┌───────────────────────┐
        │    Tools Layer    │      │  Individual Memories  │
        │ (Calculator, APIs)│      │ (per-agent histories) │
        └─────────┬─────────┘      └──────────┬────────────┘
                  │                           │
                  ▼                           ▼
                         ┌───────────────────────┐
                         │  Shared Memory Store  │
                         │ (Vector / DB context) │
                         └───────────────────────┘
## Deployment

Local development URL (run the server with `uvicorn api:app --reload`):

- API Docs: http://127.0.0.1:8000/docs
- Web UI: http://127.0.0.1:8000/ui
