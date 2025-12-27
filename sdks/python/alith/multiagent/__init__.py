"""
Alith Multi-Agent Framework

A comprehensive framework for building multi-agent systems with:
- Sequential agent chains
- Parallel agent execution
- Shared memory and communication
- Agent orchestration
- Role-based agent specialization
"""

from .agent import MultiAgent, AgentRole
from .chain import AgentChain, ChainStep
from .parallel import ParallelAgents, ParallelTask
from .memory import SharedMemory
from .orchestrator import AgentOrchestrator, WorkflowStep, WorkflowCondition
from .types import AgentMessage, AgentResult, AgentStatus

__all__ = [
    # Core classes
    "MultiAgent",
    "AgentRole",
    "AgentChain",
    "ChainStep",
    "ParallelAgents",
    "ParallelTask",
    "SharedMemory",
    "AgentOrchestrator",
    "WorkflowStep",
    "WorkflowCondition",
    # Types
    "AgentMessage",
    "AgentResult",
    "AgentStatus",
]








