"""
Parallel agent execution for concurrent workflows.
"""

import concurrent.futures
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .agent import MultiAgent
from .memory import SharedMemory
from .types import AgentResult


@dataclass
class ParallelTask:
    """A task to execute in parallel."""
    
    agent: MultiAgent
    prompt: str
    context: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    
    def __post_init__(self):
        """Set default name if not provided."""
        if not self.name:
            self.name = self.agent.agent_id


class ParallelAgents:
    """
    Execute multiple agents in parallel.
    
    Useful for:
    - Independent tasks that can run concurrently
    - Gathering multiple perspectives
    - Parallel data processing
    """
    
    def __init__(
        self,
        memory: Optional[SharedMemory] = None,
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize parallel agent executor.
        
        Args:
            memory: Optional shared memory for agent communication
            max_workers: Maximum number of parallel workers (default: number of CPUs)
            timeout: Maximum time to wait for all tasks (seconds)
        """
        self.memory = memory or SharedMemory()
        self.max_workers = max_workers
        self.timeout = timeout
        self._lock = threading.Lock()
    
    def execute(
        self,
        tasks: List[ParallelTask],
        wait_for_all: bool = True,
    ) -> Dict[str, AgentResult]:
        """
        Execute multiple agents in parallel.
        
        Args:
            tasks: List of tasks to execute
            wait_for_all: Whether to wait for all tasks to complete
        
        Returns:
            Dictionary mapping agent IDs to results
        """
        # Register all agents
        for task in tasks:
            self.memory.register_agent(task.agent.agent_id)
        
        results = {}
        
        def execute_task(task: ParallelTask) -> AgentResult:
            """Execute a single task."""
            # Get messages for this agent
            messages = self.memory.get_messages(recipient=task.agent.agent_id)
            
            # Merge context
            context = task.context or {}
            if self.memory.context:
                context = {**self.memory.context, **context}
            
            # Execute agent
            result = task.agent.execute(
                prompt=task.prompt,
                context=context,
                messages=messages,
            )
            
            # Store result in memory
            with self._lock:
                self.memory.store_result(result)
                results[task.agent.agent_id] = result
            
            return result
        
        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(execute_task, task): task for task in tasks}
            
            if wait_for_all:
                # Wait for all to complete
                concurrent.futures.wait(futures, timeout=self.timeout)
            else:
                # Return as they complete
                for future in concurrent.futures.as_completed(futures, timeout=self.timeout):
                    pass
        
        return results
    
    def execute_with_callback(
        self,
        tasks: List[ParallelTask],
        callback: Callable[[str, AgentResult], None],
    ) -> Dict[str, AgentResult]:
        """
        Execute tasks in parallel and call callback as each completes.
        
        Args:
            tasks: List of tasks to execute
            callback: Function called with (agent_id, result) as each task completes
        
        Returns:
            Dictionary mapping agent IDs to results
        """
        results = {}
        
        def execute_task(task: ParallelTask) -> AgentResult:
            """Execute a single task."""
            messages = self.memory.get_messages(recipient=task.agent.agent_id)
            context = {**(task.context or {}), **self.memory.context}
            
            result = task.agent.execute(
                prompt=task.prompt,
                context=context,
                messages=messages,
            )
            
            with self._lock:
                self.memory.store_result(result)
                results[task.agent.agent_id] = result
                callback(task.agent.agent_id, result)
            
            return result
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(execute_task, task): task for task in tasks}
            concurrent.futures.wait(futures, timeout=self.timeout)
        
        return results
    
    def execute_async(
        self,
        tasks: List[ParallelTask],
    ):
        """
        Execute tasks asynchronously.
        
        Note: This is a placeholder for future async implementation.
        Currently executes synchronously using threads.
        """
        return self.execute(tasks)








