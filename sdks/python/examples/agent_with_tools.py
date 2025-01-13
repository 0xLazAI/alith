from alith import Agent


def sum(x: int, y: int) -> int:
    """Add x and y together"""
    return x + y


def sub(x: int, y: int) -> int:
    """Subtract y from x (i.e.: x - y)"""
    return x - y


agent = Agent(
    name="Calculator Agent",
    model="gpt-4o-mini",
    preamble="You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.",
    tools=[sum, sub],
)
print(agent.prompt("Calculate 10 - 3"))
