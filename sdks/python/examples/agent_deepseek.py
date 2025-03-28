from alith import Agent

agent = Agent(
    model="deepseek-chat",  # or `deepseek-reasoner` for DeepSeek R1 Model
    api_key="<Your API Key>",  # Replace with your api key or read it from env.
    base_url="api.deepseek.com",
    preamble="You are a comedian here to entertain the user using humour and jokes.",
)
print(agent.prompt("Entertain me!"))
