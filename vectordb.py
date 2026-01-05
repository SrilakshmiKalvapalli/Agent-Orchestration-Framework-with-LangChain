# vectordb.py
from openai import OpenAI
import faiss
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

# ---------- Low-level custom FAISS memory (Milestones 1–3) ----------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

dimension = 1536
index = faiss.IndexFlatL2(dimension)
stored_texts: list[str] = []


def embed_text(text: str) -> np.ndarray | None:
    """Convert text to embedding vector using OpenAI embeddings.
    Returns None if OpenAI request fails (timeout / no internet / bad key).
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return np.array(response.data[0].embedding).astype("float32")
    except Exception as e:
        # Do not crash the app if OpenAI is unavailable
        print("embed_text error, skipping embedding:", e)
        return None


def add_memory(text: str) -> None:
    """Store text in vector DB (custom FAISS long-term memory).
    If embedding is None (OpenAI failed), just keep raw text.
    """
    vector = embed_text(text)
    if vector is None:
        # No vector; keep text so earlier milestones still see history.
        stored_texts.append(text)
        return

    index.add(np.array([vector]))
    stored_texts.append(text)


def search_memory(query: str, k: int = 3) -> list[str]:
    """Retrieve similar memories from the custom FAISS index.
    If embeddings are unavailable, return last k stored_texts.
    """
    if index.ntotal == 0:
        # No vectors at all → fallback to recent texts
        return stored_texts[-k:]

    query_vector = embed_text(query)
    if query_vector is None:
        # Cannot embed query → fallback
        return stored_texts[-k:]

    distances, ids = index.search(np.array([query_vector]), k)

    results: list[str] = []
    for idx in ids[0]:
        if idx < len(stored_texts):
            results.append(stored_texts[idx])
    return results


# ==== Shared memory for Milestone 3 / 4 (used by orchestrator) ====
# NOTE: avoids OpenAI call at startup so FastAPI can boot.

_shared_memory = None


def get_shared_memory():
    """
    Simple shared-memory implementation for cross-agent context.

    It exposes save_context/load_memory_variables so orchestrator.py
    and other modules continue to work, but doesn't call OpenAI.
    """
    global _shared_memory
    if _shared_memory is None:

        class DummySharedMemory:
            def __init__(self):
                # store a list of {"input": ..., "output": ...}
                self.history: list[dict] = []

            def save_context(self, inputs: dict, outputs: dict) -> None:
                self.history.append({"input": inputs, "output": outputs})

            def load_memory_variables(self, inputs: dict) -> dict:
                # you can customize filtering using `inputs` later
                return {"history": self.history}

        _shared_memory = DummySharedMemory()

    return _shared_memory


# ---- Small self-test (optional) ----
if __name__ == "__main__":
    # Test low-level FAISS memory
    add_memory("User said: Hello multi-agent world")
    add_memory("Assistant replied: This is a test message.")
    print("Search results for 'multi-agent':", search_memory("multi-agent"))

    # Test shared memory stub
    mem = get_shared_memory()
    mem.save_context({"input": "Who are you?"}, {"output": "I am an AI assistant."})
    print(mem.load_memory_variables({"input": "Tell me about yourself"}))


















