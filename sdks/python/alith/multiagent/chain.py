"""
Sequential agent chain for multi-step workflows.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .agent import MultiAgent
from .memory import SharedMemory
from .types import AgentResult, AgentStatus


@dataclass
class ChainStep:
    """A step in an agent chain."""
    
    agent: MultiAgent
    prompt_template: str  # Can use {input}, {previous_output}, {context}
    condition: Optional[Callable[[AgentResult], bool]] = None  # Skip if condition returns False
    transform: Optional[Callable[[AgentResult], Any]] = None  # Transform output before next step
    name: Optional[str] = None
    
    def __post_init__(self):
        """Set default name if not provided."""
        if not self.name:
            self.name = self.agent.agent_id


class AgentChain:
    """
    Sequential chain of agents for multi-step workflows.
    
    Each agent in the chain receives:
    - The original input
    - The output from the previous agent
    - Shared context from memory
    """
    
    def __init__(
        self,
        steps: List[ChainStep],
        memory: Optional[SharedMemory] = None,
        stop_on_error: bool = True,
    ):
        """
        Initialize an agent chain.
        
        Args:
            steps: List of chain steps to execute sequentially
            memory: Optional shared memory for agent communication
            stop_on_error: Whether to stop execution on error
        """
        self.steps = steps
        self.memory = memory or SharedMemory()
        self.stop_on_error = stop_on_error
        
        # Register all agents in memory
        for step in steps:
            self.memory.register_agent(step.agent.agent_id)
    
    def execute(
        self,
        input_data: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[AgentResult]:
        """
        Execute the chain with input data.
        
        Args:
            input_data: Initial input for the chain
            context: Additional context dictionary
        
        Returns:
            List of results from each step
        """
        results = []
        previous_output = input_data
        context = context or {}
        
        # Update shared memory context
        if context:
            self.memory.update_context(context)
        
        for i, step in enumerate(self.steps):
            # Build prompt from template
            prompt = step.prompt_template.format(
                input=input_data,
                previous_output=previous_output,
                context=self.memory.context,
            )
            
            # Get messages for this agent
            messages = self.memory.get_messages(recipient=step.agent.agent_id)
            
            # Execute agent
            result = step.agent.execute(
                prompt=prompt,
                context=self.memory.context,
                messages=messages,
            )
            
            results.append(result)
            
            # Store result in memory
            self.memory.store_result(result)
            
            # Check for errors
            if result.status == AgentStatus.FAILED:
                if self.stop_on_error:
                    return results
                continue
            
            # Check condition
            if step.condition and not step.condition(result):
                result.status = AgentStatus.SKIPPED
                continue
            
            # Transform output if needed
            if step.transform:
                previous_output = step.transform(result)
            else:
                previous_output = result.output if result.output else ""
        
        return results
    
    def execute_async(
        self,
        input_data: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Execute the chain asynchronously.
        
        Note: This is a placeholder for future async implementation.
        Currently executes synchronously.
        """
        return self.execute(input_data, context)
    
    def add_step(
        self,
        agent: MultiAgent,
        prompt_template: str,
        condition: Optional[Callable[[AgentResult], bool]] = None,
        transform: Optional[Callable[[AgentResult], Any]] = None,
        name: Optional[str] = None,
        position: Optional[int] = None,
    ) -> None:
        """Add a step to the chain."""
        step = ChainStep(
            agent=agent,
            prompt_template=prompt_template,
            condition=condition,
            transform=transform,
            name=name,
        )
        
        if position is None:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)
        
        self.memory.register_agent(agent.agent_id)
    
    def get_final_result(self) -> Optional[AgentResult]:
        """Get the result from the last step."""
        if not self.steps:
            return None
        
        last_agent_id = self.steps[-1].agent.agent_id
        return self.memory.get_result(last_agent_id)








