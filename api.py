from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from typing import Optional

from orchestrator import run_multi_agent_pipeline
from agent import run_agent
from tools import tools_list, create_tool_agent
from memory import retrieve_context, store_user_message, store_agent_message
from vectordb import search_memory, stored_texts


app = FastAPI(title="Multi-Agent Orchestration API")

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Serve index.html from SAME folder, under /ui ----
# api.py and index.html are in the SAME directory
app.mount("/ui", StaticFiles(directory=".", html=True), name="static")

tool_agent = create_tool_agent()


class AgentRunRequest(BaseModel):
    message: str


class AgentRunResponse(BaseModel):
    agent: str = "single_agent"
    response: str


class WorkflowRunRequest(BaseModel):
    query: str


class WorkflowRunResponse(BaseModel):
    workflow_id: str
    full_text: str


class WorkflowStepRequest(BaseModel):
    workflow_id: Optional[str] = None
    step: str
    query: str


class MemoryResetRequest(BaseModel):
    agent_name: Optional[str] = None


class ToolTestRequest(BaseModel):
    tool_name: str
    input: str


class FeedbackRequest(BaseModel):
    workflow_id: str
    rating: int
    comments: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/agent/run", response_model=AgentRunResponse)
async def agent_run(req: AgentRunRequest):
    store_user_message(req.message)
    result = run_agent(req.message)
    store_agent_message(result)
    return AgentRunResponse(response=result)


@app.post("/workflow/run", response_model=WorkflowRunResponse)
async def workflow_run(req: WorkflowRunRequest):
    full = run_multi_agent_pipeline(req.query)
    wid = f"wf-{abs(hash(req.query))}"
    return WorkflowRunResponse(workflow_id=wid, full_text=full)


@app.post("/workflow/step")
async def workflow_step(req: WorkflowStepRequest):
    if req.step == "full":
        full = run_multi_agent_pipeline(req.query)
        return {"workflow_id": req.workflow_id or "temp", "result": full}
    return {
        "workflow_id": req.workflow_id or "temp",
        "message": "Step handling not yet implemented",
    }


@app.get("/memory/{agent_name}")
async def get_memory(agent_name: str):
    results = search_memory(agent_name)
    return {"agent_name": agent_name, "memories": results}


@app.post("/memory/reset")
async def reset_memory(req: MemoryResetRequest):
    stored_texts.clear()
    return {"message": "Memory reset", "agent_name": req.agent_name}


@app.post("/tools/test")
async def tools_test(req: ToolTestRequest):
    for tool in tools_list:
        if tool.name.lower() == req.tool_name.lower():
            result = tool.func(req.input)
            return {"tool": tool.name, "input": req.input, "result": result}
    return {"error": f"Tool {req.tool_name} not found"}


@app.get("/agents")
async def list_agents():
    return {
        "agents": [
            "single_agent",
            "tool_agent",
            "research_agent",
            "summarizer_agent",
            "email_agent",
            "multi_agent_pipeline",
        ]
    }


@app.get("/tools")
async def list_tools():
    return {"tools": [{"name": t.name, "description": t.description} for t in tools_list]}


@app.post("/evaluation/feedback")
async def feedback(req: FeedbackRequest):
    print("Feedback:", req.dict())
    return {"message": "Feedback received"}


@app.get("/logs/{workflow_id}")
async def get_logs(workflow_id: str):
    return {"workflow_id": workflow_id, "logs": ["Logging not implemented yet."]}

