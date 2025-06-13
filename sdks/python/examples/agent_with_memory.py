from alith import Agent, WindowBufferMemory

agent = Agent(
    model="gpt-4",
    preamble="You are a comedian here to entertain the user using humour and jokes.",
    memory=WindowBufferMemory(),
)
print(agent.prompt("Entertain me!"))
print(agent.prompt("Entertain me again!"))
