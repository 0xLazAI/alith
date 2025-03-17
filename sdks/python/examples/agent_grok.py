from alith import Agent
from os import environ

agent = Agent(
    model="grok-3",
    api_key=environ["GROK_API_KEY"],
    base_url="api.grok.ai/v1",
    preamble="You are a comedian here to entertain the user using humour and jokes.",
)
print(agent.prompt("Entertain me!"))
