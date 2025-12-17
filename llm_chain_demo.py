# llm_chain_demo.py
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

load_dotenv()

def build_llm_chain():
    """Minimal LLMChain example to satisfy milestone requirement."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = PromptTemplate(
        input_variables=["topic"],
        template=(
            "You are a helpful AI tutor. "
            "Explain the topic below in simple terms in 3–4 sentences.\n\n"
            "Topic: {topic}"
        ),
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

if __name__ == "__main__":
    chain = build_llm_chain()
    out = chain.run({"topic": "How machine learning works"})
    print(out)
