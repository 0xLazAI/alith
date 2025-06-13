import os

from alith import Agent

agent = Agent(
    base_url="api.siliconflow.cn/v1",
    model="deepseek-ai/DeepSeek-V3",
    api_key=os.environ["LLM_API_KEY"],
    preamble="You are a comedian here to entertain the user using humour and jokes.",
)
print(agent.prompt("Entertain me!"))
