# vectordb.py
from openai import OpenAI
import faiss
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

dimension = 1536
index = faiss.IndexFlatL2(dimension)
stored_texts = []


def embed_text(text: str):
    """Convert text to embedding vector."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding).astype("float32")


def add_memory(text: str):
    """Store text in vector DB (custom FAISS long-term memory)."""
    vector = embed_text(text)
    index.add(np.array([vector]))
    stored_texts.append(text)


def search_memory(query: str, k=3):
    """Retrieve similar memories from the custom FAISS index."""
    if index.ntotal == 0:
        return []

    query_vector = embed_text(query)
    distances, ids = index.search(np.array([query_vector]), k)

    results = []
    for idx in ids[0]:
        if idx < len(stored_texts):
            results.append(stored_texts[idx])
    return results


# ==== LangChain VectorStoreRetrieverMemory (shared memory for Milestone 3) ====

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain.memory import VectorStoreRetrieverMemory

_shared_memory = None


def get_shared_memory():
    """
    Create (once) and return a LangChain VectorStoreRetrieverMemory.
    Uses a dummy initial text so FAISS knows the embedding dimension.
    """
    global _shared_memory
    if _shared_memory is None:
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        # Use one dummy text instead of an empty list to avoid IndexFlatL2 error
        vectorstore = LC_FAISS.from_texts(
            ["Initial shared memory placeholder"],
            embedding=embeddings,
        )
        _shared_memory = VectorStoreRetrieverMemory(
            retriever=vectorstore.as_retriever(),
        )
    return _shared_memory


# ---- Small self-test (optional) ----
if __name__ == "__main__":
    # Test VectorStoreRetrieverMemory part
    mem = get_shared_memory()
    mem.save_context({"input": "Who are you?"}, {"output": "I am an AI assistant."})
    print(mem.load_memory_variables({"input": "Tell me about yourself"}))
















