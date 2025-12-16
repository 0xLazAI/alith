"""
Advanced agent orchestrator for complex multi-agent workflows.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from .agent import MultiAgent
from .chain import AgentChain, ChainStep
from .memory import SharedMemory
from .parallel import ParallelAgents, ParallelTask
from .types import AgentResult


@dataclass
class WorkflowCondition:
    """Condition for workflow branching."""
    
    check: Callable[[Dict[str, AgentResult]], bool]
    description: Optional[str] = None


@dataclass
class WorkflowStep:
    """A step in a workflow."""
    
    name: str
    agents: Union[MultiAgent, List[MultiAgent]]
    prompt: Union[str, Dict[str, str]]  # Single prompt or dict mapping agent_id to prompt
    condition: Optional[WorkflowCondition] = None
    parallel: bool = False  # Execute agents in parallel if multiple
    transform: Optional[Callable[[Dict[str, AgentResult]], Any]] = None
    next_step: Optional[str] = None  # Name of next step (for branching)
    
    def __post_init__(self):
        """Normalize agents to list."""
        if not isinstance(self.agents, list):
            self.agents = [self.agents]
        
        if not isinstance(self.prompt, dict):
            # Use same prompt for all agents
            self.prompt = {agent.agent_id: self.prompt for agent in self.agents}


class AgentOrchestrator:
    """
    Advanced orchestrator for complex multi-agent workflows.
    
    Supports:
    - Sequential and parallel execution
    - Conditional branching
    - Dynamic workflow construction
    - Result aggregation and transformation
    """
    
    def __init__(
        self,
        memory: Optional[SharedMemory] = None,
        max_parallel_workers: Optional[int] = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            memory: Optional shared memory for agent communication
            max_parallel_workers: Maximum workers for parallel execution
        """
        self.memory = memory or SharedMemory()
        self.parallel_executor = ParallelAgents(
            memory=self.memory,
            max_workers=max_parallel_workers,
        )
        self.steps: Dict[str, WorkflowStep] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def add_step(
        self,
        step: WorkflowStep,
    ) -> None:
        """Add a workflow step."""
        self.steps[step.name] = step
        
        # Register agents in memory
        for agent in step.agents:
            self.memory.register_agent(agent.agent_id)
    
    def execute(
        self,
        initial_input: str,
        context: Optional[Dict[str, Any]] = None,
        start_step: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the workflow.
        
        Args:
            initial_input: Initial input for the workflow
            context: Additional context
            start_step: Name of step to start from (default: first step)
        
        Returns:
            Dictionary with execution results and metadata
        """
        if not self.steps:
            return {
                "success": False,
                "error": "No steps defined",
                "results": {},
            }
        
        # Initialize context
        if context:
            self.memory.update_context(context)
        
        # Determine starting step
        if start_step and start_step in self.steps:
            current_step_name = start_step
        else:
            current_step_name = list(self.steps.keys())[0]
        
        all_results: Dict[str, AgentResult] = {}
        current_input = initial_input
        
        # Execute workflow
        while current_step_name:
            step = self.steps[current_step_name]
            
            # Check condition if present
            if step.condition and not step.condition.check(all_results):
                # Condition not met, skip or branch
                if step.next_step:
                    current_step_name = step.next_step
                else:
                    break
                continue
            
            # Execute step
            if step.parallel and len(step.agents) > 1:
                # Parallel execution
                tasks = [
                    ParallelTask(
                        agent=agent,
                        prompt=step.prompt.get(agent.agent_id, step.prompt.get(list(step.prompt.keys())[0])),
                        context=self.memory.context,
                        name=agent.agent_id,
                    )
                    for agent in step.agents
                ]
                step_results = self.parallel_executor.execute(tasks)
            else:
                # Sequential execution
                step_results = {}
                for agent in step.agents:
                    prompt = step.prompt.get(agent.agent_id, step.prompt.get(list(step.prompt.keys())[0]))
                    messages = self.memory.get_messages(recipient=agent.agent_id)
                    result = agent.execute(
                        prompt=prompt,
                        context=self.memory.context,
                        messages=messages,
                    )
                    step_results[agent.agent_id] = result
                    self.memory.store_result(result)
            
            # Store results
            all_results.update(step_results)
            
            # Transform output if needed
            if step.transform:
                current_input = step.transform(step_results)
            else:
                # Use output from first agent as input for next step
                if step_results:
                    first_result = list(step_results.values())[0]
                    current_input = first_result.output if first_result.output else current_input
            
            # Record execution
            self.execution_history.append({
                "step": current_step_name,
                "results": {k: v.to_dict() for k, v in step_results.items()},
                "input": current_input,
            })
            
            # Determine next step
            if step.next_step and step.next_step in self.steps:
                current_step_name = step.next_step
            else:
                break
        
        return {
            "success": True,
            "results": {k: v.to_dict() for k, v in all_results.items()},
            "final_output": current_input,
            "execution_history": self.execution_history,
            "memory_summary": self.memory.get_summary(),
        }
    
    def create_chain(
        self,
        step_names: List[str],
    ) -> AgentChain:
        """
        Create an agent chain from workflow steps.
        
        Args:
            step_names: List of step names in order
        
        Returns:
            AgentChain instance
        """
        chain_steps = []
        
        for step_name in step_names:
            if step_name not in self.steps:
                continue
            
            step = self.steps[step_name]
            if len(step.agents) != 1:
                raise ValueError(f"Step '{step_name}' must have exactly one agent for chain")
            
            agent = step.agents[0]
            prompt = step.prompt.get(agent.agent_id, list(step.prompt.values())[0])
            
            chain_steps.append(
                ChainStep(
                    agent=agent,
                    prompt_template=prompt,
                    condition=step.condition.check if step.condition else None,
                    transform=step.transform,
                    name=step_name,
                )
            )
        
        return AgentChain(steps=chain_steps, memory=self.memory)
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a summary of the workflow."""
        return {
            "steps": list(self.steps.keys()),
            "total_agents": sum(len(step.agents) for step in self.steps.values()),
            "execution_count": len(self.execution_history),
        }








