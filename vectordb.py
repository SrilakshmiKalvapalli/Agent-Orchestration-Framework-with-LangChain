
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
    """Store text in vector DB."""
    vector = embed_text(text)
    index.add(np.array([vector]))
    stored_texts.append(text)

def search_memory(query: str, k=3):
    """Retrieve similar memories."""
    if index.ntotal == 0:
        return []

    query_vector = embed_text(query)
    distances, ids = index.search(np.array([query_vector]), k)

    results = []
    for idx in ids[0]:
        if idx < len(stored_texts):
            results.append(stored_texts[idx])

    return results











