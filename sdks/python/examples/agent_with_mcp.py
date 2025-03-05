from alith import Agent

agent = Agent(
    name="Calculator Agent",
    model="gpt-4o-mini",
    preamble="You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.",
    mcp_config_path="servers_config.json",
)
print(agent.prompt("Calculate 10 - 3"))
