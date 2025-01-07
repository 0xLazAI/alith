from alith import Agent


def sum(x: int, y: int) -> int:
    """The sum of two num"""
    x + y


agent = Agent(name="dummy", model="gpt4o-mini", tools=[sum])
print(agent.prompt("A dummy promt"))
