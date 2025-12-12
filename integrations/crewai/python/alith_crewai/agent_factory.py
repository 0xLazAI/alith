"""Factory functions for creating CrewAI agents powered by Alith."""

import os
from typing import Any, Dict, List, Optional, Union

from alith import Agent as AlithAgent
from alith import Tool as AlithTool

from .alith_llm import AlithLLM
from .tool_converter import convert_alith_tool_to_crewai, convert_crewai_tool_to_alith


def build_preamble(role: str, goal: str, backstory: str) -> str:
    """Build an Alith preamble from CrewAI agent attributes.
    
    Args:
        role: Agent's role (e.g., "Senior Data Scientist")
        goal: Agent's goal
        backstory: Agent's backstory/background
    
    Returns:
        Formatted preamble string for Alith agent
    """
    preamble = f"""You are a {role}.

Your goal is: {goal}

Background: {backstory}

Use your expertise and available tools to help accomplish your assigned tasks."""
    
    return preamble


def create_alith_crew_agent(
    role: str,
    goal: str,
    backstory: str,
    model: str = "llama-3.3-70b-versatile",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    tools: Optional[List[Union[AlithTool, Any]]] = None,
    verbose: bool = False,
    allow_delegation: bool = False,
    **kwargs
) -> Any:
    """Create a CrewAI agent powered by Alith.
    
    This factory function creates a CrewAI-compatible agent that uses
    Alith as the underlying inference engine, providing high-performance
    inference and Web3 capabilities.
    
    Args:
        role: Agent's role (e.g., "Senior Data Scientist")
        goal: What the agent aims to achieve
        backstory: Agent's background and expertise
        model: LLM model name (default: "llama-3.3-70b-versatile")
        api_key: API key for the model (defaults to GROQ_API_KEY env var)
        base_url: Base URL for the model API (defaults to Groq API)
        tools: List of tools (Alith or CrewAI format)
        verbose: Whether to enable verbose logging
        allow_delegation: Whether the agent can delegate tasks
        **kwargs: Additional CrewAI agent parameters
    
    Returns:
        CrewAI Agent instance powered by Alith
    
    Example:
        >>> from alith_crewai import create_alith_crew_agent
        >>> from crewai import Crew, Task
        >>>
        >>> agent = create_alith_crew_agent(
        ...     role="Research Analyst",
        ...     goal="Find and analyze information",
        ...     backstory="Expert researcher with 10 years experience",
        ...     model="llama-3.3-70b-versatile"
        ... )
        >>>
        >>> task = Task(description="Research AI trends", agent=agent)
        >>> crew = Crew(agents=[agent], tasks=[task])
        >>> result = crew.kickoff()
    """
    # Import CrewAI here to avoid hard dependency
    try:
        from crewai import Agent as CrewAgent
    except ImportError:
        raise ImportError(
            "CrewAI is required to use this integration. "
            "Install it with: pip install crewai"
        )
    
    # Set default API key and base URL for Groq
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "API key is required. Either pass api_key parameter or "
                "set GROQ_API_KEY environment variable."
            )
    
    if base_url is None:
        base_url = "https://api.groq.com/openai/v1"
    
    # Build preamble from role, goal, backstory
    preamble = build_preamble(role, goal, backstory)
    
    # Convert tools to Alith format
    alith_tools = []
    crewai_tools = []
    
    if tools:
        for tool in tools:
            # Check if it's already an Alith tool
            if isinstance(tool, AlithTool):
                alith_tools.append(tool)
                crewai_tools.append(convert_alith_tool_to_crewai(tool))
            else:
                # Assume it's a CrewAI tool
                alith_tool = convert_crewai_tool_to_alith(tool)
                alith_tools.append(alith_tool)
                crewai_tools.append(tool)
    
    # Create Alith agent
    alith_agent = AlithAgent(
        name=role,
        model=model,
        preamble=preamble,
        api_key=api_key,
        base_url=base_url,
        tools=alith_tools
    )
    
    # Create custom LLM
    alith_llm = AlithLLM(alith_agent)
    
    # CrewAI validates llm parameter during init and calls create_llm()
    # We need to set a dummy OPENAI_API_KEY to pass validation
    # even though we'll replace the LLM with our custom one
    import os as _os
    _original_openai_key = _os.environ.get("OPENAI_API_KEY")
    try:
        # Set dummy key temporarily
        _os.environ["OPENAI_API_KEY"] = "dummy-key-for-validation"
        
        # Create CrewAI agent 
        crew_agent = CrewAgent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm="gpt-4",  # Dummy model for validation
            tools=crewai_tools,
            verbose=verbose,
            allow_delegation=allow_delegation,
            **kwargs
        )
    finally:
        # Restore original key
        if _original_openai_key is None:
            _os.environ.pop("OPENAI_API_KEY", None)
        else:
            _os.environ["OPENAI_API_KEY"] = _original_openai_key
    
    # Now replace the llm with our Alith LLM after instantiation
    crew_agent.llm = alith_llm
    
    return crew_agent


def create_alith_agent_with_custom_llm(
    alith_agent: AlithAgent,
    role: str,
    goal: str,
    backstory: str,
    tools: Optional[List[Any]] = None,
    **kwargs
) -> Any:
    """Create a CrewAI agent from an existing Alith agent.
    
    This function allows you to use a pre-configured Alith agent
    within CrewAI, useful when you need fine-grained control over
    the Alith agent configuration.
    
    Args:
        alith_agent: Pre-configured Alith agent instance
        role: Agent's role for CrewAI
        goal: Agent's goal for CrewAI
        backstory: Agent's backstory for CrewAI
        tools: List of CrewAI tools (optional)
        **kwargs: Additional CrewAI agent parameters
    
    Returns:
        CrewAI Agent instance using the provided Alith agent
    
    Example:
        >>> from alith import Agent
        >>> from alith_crewai import create_alith_agent_with_custom_llm
        >>>
        >>> alith_agent = Agent(
        ...     model="llama-3.3-70b-versatile",
        ...     preamble="You are an expert researcher",
        ...     api_key="your-api-key"
        ... )
        >>>
        >>> crew_agent = create_alith_agent_with_custom_llm(
        ...     alith_agent=alith_agent,
        ...     role="Researcher",
        ...     goal="Research topics",
        ...     backstory="Expert researcher"
        ... )
    """
    try:
        from crewai import Agent as CrewAgent
    except ImportError:
        raise ImportError(
            "CrewAI is required. Install with: pip install crewai"
        )
    
    # Create AlithLLM from the existing agent
    alith_llm = AlithLLM(alith_agent)
    
    # Create CrewAI agent
    return CrewAgent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=alith_llm,
        tools=tools or [],
        **kwargs
    )
