from alith import Agent

agent = Agent(
    model="llama3-70b-8192",
    api_key="Your Api key",
    base_url="https://api.groq.com/openai/v1",
)
print(agent.prompt("Calculate 2+2"))
