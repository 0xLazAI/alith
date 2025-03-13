"""Alith integration tools for AgentKit."""

from alith import Tool
from coinbase_agentkit import Action, AgentKit


def get_alith_tools(agent_kit: AgentKit) -> list[Tool]:
    """Get Alith tools from an AgentKit instance.

    Args:
        agent_kit: The AgentKit instance

    Returns:
        A list of Alith tools

    """
    actions: list[Action] = agent_kit.get_actions()

    tools = []
    for action in actions:
        tool = Tool(
            name=action.name,
            description=action.description,
            parameters=action.args_schema,
            handler=lambda **args: action.invoke(args),
        )
        tools.append(tool)

    return tools
