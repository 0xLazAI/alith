from alith import Agent, Tool
from pydantic import BaseModel


class SubToolModel(BaseModel):
    x: int
    y: int


agent = Agent(
    name="Calculator Agent",
    model="deepseek-ai/DeepSeek-V3",
    api_key="sk-lsptuflilaxnkfbjryzuibrjhxqmmwhzqpkbjrnorqhcfsuw",
    base_url="api.siliconflow.cn/v1",
    preamble="You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.",
    tools=[
        Tool(
            name="sub",
            description="Subtract y from x (i.e.: x - y)",
            definition=SubToolModel,
            handler=lambda x, y: x - y,
        )
    ],
)
print(agent.prompt("Calculate 10 - 3"))
