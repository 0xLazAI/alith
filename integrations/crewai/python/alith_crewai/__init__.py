"""Alith integration for CrewAI.

This package enables using Alith agents as the backend for CrewAI's
multi-agent orchestration framework, combining CrewAI's powerful
orchestration with Alith's high-performance inference and Web3 capabilities.

Example:
    >>> from alith_crewai import create_alith_crew_agent
    >>> from crewai import Crew, Task
    >>>
    >>> agent = create_alith_crew_agent(
    ...     role="Research Analyst",
    ...     goal="Find information about AI trends",
    ...     backstory="Expert AI researcher",
    ...     model="llama-3.3-70b-versatile"
    ... )
    >>>
    >>> task = Task(description="Research AI trends", agent=agent)
    >>> crew = Crew(agents=[agent], tasks=[task])
    >>> result = crew.kickoff()
"""

from .alith_llm import AlithLLM
from .agent_factory import create_alith_crew_agent
from .tool_converter import (
    convert_alith_tool_to_crewai,
    convert_crewai_tool_to_alith,
)

__all__ = [
    "AlithLLM",
    "create_alith_crew_agent",
    "convert_alith_tool_to_crewai",
    "convert_crewai_tool_to_alith",
]

__version__ = "0.1.0"
__author__ = "LazAI Labs"
