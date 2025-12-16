"""
Multi-Agent base classes and role definitions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agent import Agent as BaseAgent
from .types import AgentMessage, AgentResult, AgentStatus


class AgentRole:
    """Predefined agent roles for common use cases."""
    
    RESEARCHER = "researcher"
    WRITER = "writer"
    ANALYZER = "analyzer"
    CODER = "coder"
    REVIEWER = "reviewer"
    TRANSLATOR = "translator"
    SUMMARIZER = "summarizer"
    PLANNER = "planner"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    CUSTOM = "custom"

    @staticmethod
    def get_preamble(role: str) -> str:
        """Get default preamble for a role."""
        preambles = {
            AgentRole.RESEARCHER: "You are a research specialist. Your role is to research topics thoroughly and provide comprehensive, well-sourced information.",
            AgentRole.WRITER: "You are a professional writer. Your role is to create clear, engaging, and well-structured content based on provided information.",
            AgentRole.ANALYZER: "You are a data analyst. Your role is to analyze information, identify patterns, and provide insights.",
            AgentRole.CODER: "You are a software engineer. Your role is to write clean, efficient, and well-documented code.",
            AgentRole.REVIEWER: "You are a quality reviewer. Your role is to review content, code, or analysis and provide constructive feedback.",
            AgentRole.TRANSLATOR: "You are a professional translator. Your role is to translate text accurately while preserving meaning and context.",
            AgentRole.SUMMARIZER: "You are a summarization specialist. Your role is to create concise, accurate summaries of provided content.",
            AgentRole.PLANNER: "You are a strategic planner. Your role is to create detailed plans and break down complex tasks into steps.",
            AgentRole.EXECUTOR: "You are an executor. Your role is to execute tasks based on provided plans and instructions.",
            AgentRole.VALIDATOR: "You are a validator. Your role is to validate results, check for errors, and ensure quality standards.",
        }
        return preambles.get(role, "You are a helpful AI assistant.")


@dataclass
class MultiAgent:
    """
    Wrapper around Alith Agent for multi-agent systems.
    
    Provides additional features like:
    - Agent identification
    - Role-based behavior
    - Message handling
    - Result tracking
    """
    
    agent_id: str
    agent: BaseAgent
    role: str = AgentRole.CUSTOM
    description: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Other agent IDs this agent depends on
    
    def __post_init__(self):
        """Initialize agent with role-based preamble if not already set."""
        if not self.agent.preamble and self.role != AgentRole.CUSTOM:
            self.agent.preamble = AgentRole.get_preamble(self.role)
    
    def execute(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        messages: Optional[List[AgentMessage]] = None,
    ) -> AgentResult:
        """
        Execute the agent with a prompt.
        
        Args:
            prompt: The prompt to send to the agent
            context: Additional context dictionary
            messages: Messages from other agents to include in context
            
        Returns:
            AgentResult with the execution outcome
        """
        import time
        start_time = time.time()
        
        try:
            # Build enhanced prompt with context
            enhanced_prompt = self._build_prompt(prompt, context, messages)
            
            # Execute agent
            output = self.agent.prompt(enhanced_prompt)
            
            execution_time = time.time() - start_time
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                output=output,
                execution_time=execution_time,
                metadata={
                    "role": self.role,
                    "prompt_length": len(prompt),
                    "output_length": len(output) if isinstance(output, str) else 0,
                },
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
            )
    
    def _build_prompt(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        messages: Optional[List[AgentMessage]] = None,
    ) -> str:
        """Build enhanced prompt with context and messages."""
        parts = [prompt]
        
        # Add context
        if context:
            context_str = "\n\nContext:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"
            parts.append(context_str)
        
        # Add messages from other agents
        if messages:
            messages_str = "\n\nMessages from other agents:\n"
            for msg in messages:
                messages_str += f"[{msg.sender}]: {msg.content}\n"
            parts.append(messages_str)
        
        return "\n".join(parts)
    
    def send_message(
        self,
        content: str,
        recipient: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """Create a message to send to other agents."""
        return AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            content=content,
            metadata=metadata or {},
        )
    
    def can_handle(self, task: str) -> bool:
        """Check if this agent can handle a given task based on capabilities."""
        if not self.capabilities:
            return True  # No restrictions
        
        task_lower = task.lower()
        return any(cap.lower() in task_lower for cap in self.capabilities)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "description": self.description,
            "capabilities": self.capabilities,
            "dependencies": self.dependencies,
        }








