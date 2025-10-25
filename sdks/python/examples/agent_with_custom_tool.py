import json
from alith import Agent

# 1. Define your custom tool as a plain Python function
# We'll create a simple function that pretends to get weather data.
def get_current_weather(location: str, unit: str = "celsius"):
    """
    Get the current weather for a specific location.
    
    Args:
        location: The city and state, e.g., "San Francisco, CA"
        unit: The unit for temperature, either "celsius" or "fahrenheit"
    """
    print(f"\n[Tool Call: get_current_weather(location={location}, unit={unit})]")
    
    # In a real app, you'd call a weather API here.
    # We'll just return some fake data.
    weather_data = {
        "location": location,
        "temperature": "15",
        "unit": unit,
        "forecast": "partly cloudy"
    }
    
    # The tool must return a string, so we'll dump the JSON
    return json.dumps(weather_data)

# 2. Create the agent and pass the tool in the 'tools' list
agent = Agent(
    model="gpt-3.5-turbo",
    tools=[get_current_weather]
)

# 3. Ask a question that requires the tool
# The agent will see the function's docstring and understand
# when to use it.
prompt = "What's the weather like in Berlin right now?"

print(f"User: {prompt}\n")
response = agent.prompt(prompt)

print(f"\nAgent: {response}")