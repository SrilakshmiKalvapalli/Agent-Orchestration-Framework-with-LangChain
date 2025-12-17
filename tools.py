# tools.py
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool  # LangChain Tool

load_dotenv()


# ------------------------
# Tool functions
# ------------------------

def calculator(expression: str) -> str:
    """
    Evaluates a simple math expression.
    Example: '2 + 5 * 3'
    """
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid math expression. Please use numbers and operators like +, -, *, /."


def fake_weather(location: str) -> str:
    """
    Returns simulated weather information for a location.
    """
    try:
        return f"The current weather in {location} is 25°C and sunny (simulated)."
    except Exception:
        return "Weather API failed. Please try again."


# ------------------------
# LangChain Tool objects
# ------------------------

lc_calculator_tool = Tool(
    name="CALCULATOR",
    func=calculator,
    description="Use this tool to calculate mathematical expressions like '23+45'.",
)

lc_weather_tool = Tool(
    name="WEATHER_API",
    func=fake_weather,
    description="Use this tool to get simulated weather for a city name.",
)

# Export list so other files can reuse
tools_list = [lc_calculator_tool, lc_weather_tool]

# ------------------------
# Agent setup
# ------------------------

SYSTEM_PROMPT = """
You are a tool-using agent. Follow these instructions:

1. Use CALCULATOR for any math expression.
2. Use WEATHER_API for any weather or temperature question.
3. If the user asks about both weather and math, call both tools.
4. If the question does not need tools, answer directly.
5. If a tool returns an error message, apologize and suggest another way.
6. Always explain your final answer clearly.
"""


def create_tool_agent():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = initialize_agent(
        tools=tools_list,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        system_message=SYSTEM_PROMPT,
    )
    return agent







