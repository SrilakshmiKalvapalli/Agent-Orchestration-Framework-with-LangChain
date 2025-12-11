# tools.py

# ------------------------
# Define tools as functions
# ------------------------

def calculator(expression: str):
    """
    Evaluates a simple math expression.
    Example: '2 + 5 * 3'
    """
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid math expression. Please use numbers and operators like +, -, *, /."

def fake_weather(location: str):
    """
    Returns simulated weather information for a location.
    """
    try:
        return f"The current weather in {location} is 25°C and sunny (simulated)."
    except Exception:
        return "Weather API failed. Please try again."

# ------------------------
# SimpleTool class to mimic a tool object
# ------------------------
class SimpleTool:
    def __init__(self, name, func):
        self.name = name
        self.func = func

# Create tool instances
calculator_tool = SimpleTool("calculator", calculator)
weather_tool = SimpleTool("weather", fake_weather)

# Export tools as a list
tools_list = [calculator_tool, weather_tool]


