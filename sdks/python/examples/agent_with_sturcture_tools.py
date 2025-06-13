from pydantic import BaseModel

from alith import Agent, Tool


class SubToolModel(BaseModel):
    x: int
    y: int


agent = Agent(
    name="Calculator Agent",
    model="gpt-4",
    preamble="You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.",
    tools=[
        Tool(
            name="sub",
            description="Subtract y from x (i.e.: x - y)",
            parameters=SubToolModel,
            handler=lambda x, y: x - y,
        )
    ],
)
print(agent.prompt("Calculate 10 - 3"))
