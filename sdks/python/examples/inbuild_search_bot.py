from alith import Agent, DuckDuckGoTool


def sum(x: int, y: int) -> int:
    """Add x and y together"""
    return x + y + 100

ddg_tool = DuckDuckGoTool().to_tool()

search_agent = Agent(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key="your key here",
    tools=[ddg_tool]
)
reply_agent = Agent(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key="your key here",
)
# print(agent.prompt("sum 10 and 20"))
response = search_agent.prompt("give me the youtube link to learn python programming")
prompt_summary = "make the following text more concise and to the point: " + response
print(prompt_summary)
summary=reply_agent.prompt(prompt_summary)
print(summary)

# summary=agent.prompt("summarize the following text: " + response)
# print(summary)