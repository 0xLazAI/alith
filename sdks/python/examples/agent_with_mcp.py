from alith import Agent

agent = Agent(
    model="gpt-4-mini",
    mcp_config_path="servers_config.json",
)
print(agent.prompt("Read the file: agent_with_mcp.py"))
print(agent.prompt("Read the content of file: agent_with_memory.py"))
