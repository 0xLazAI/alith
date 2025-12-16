"""
Shared memory system for multi-agent communication.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import AgentMessage, AgentResult


@dataclass
class SharedMemory:
    """
    Shared memory for multi-agent systems.
    
    Provides:
    - Message broadcasting and routing
    - Shared context storage
    - Result aggregation
    - Agent state management
    """
    
    agent_ids: List[str] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)
    results: Dict[str, AgentResult] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    max_messages: int = 1000
    max_results: int = 100
    
    def register_agent(self, agent_id: str) -> None:
        """Register an agent in the shared memory."""
        if agent_id not in self.agent_ids:
            self.agent_ids.append(agent_id)
    
    def send_message(
        self,
        message: AgentMessage,
        store: bool = True,
    ) -> None:
        """
        Send a message to the shared memory.
        
        Args:
            message: The message to send
            store: Whether to store the message in history
        """
        if store:
            self.messages.append(message)
            # Trim old messages if limit exceeded
            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages:]
    
    def get_messages(
        self,
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[AgentMessage]:
        """
        Get messages from the shared memory.
        
        Args:
            sender: Filter by sender agent ID
            recipient: Filter by recipient agent ID (None for broadcast)
            limit: Maximum number of messages to return
        
        Returns:
            List of matching messages
        """
        filtered = self.messages
        
        if sender:
            filtered = [m for m in filtered if m.sender == sender]
        
        if recipient is not None:
            filtered = [m for m in filtered if m.recipient == recipient]
        
        if limit:
            filtered = filtered[-limit:]
        
        return filtered
    
    def store_result(self, result: AgentResult) -> None:
        """Store an agent result."""
        self.results[result.agent_id] = result
        # Trim old results if limit exceeded
        if len(self.results) > self.max_results:
            # Keep only the most recent results
            sorted_results = sorted(
                self.results.items(),
                key=lambda x: x[1].timestamp,
            )
            self.results = dict(sorted_results[-self.max_results:])
    
    def get_result(self, agent_id: str) -> Optional[AgentResult]:
        """Get the latest result from an agent."""
        return self.results.get(agent_id)
    
    def get_all_results(self) -> Dict[str, AgentResult]:
        """Get all stored results."""
        return self.results.copy()
    
    def set_context(self, key: str, value: Any) -> None:
        """Set a context value."""
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self.context.get(key, default)
    
    def update_context(self, updates: Dict[str, Any]) -> None:
        """Update multiple context values."""
        self.context.update(updates)
    
    def clear_context(self) -> None:
        """Clear all context."""
        self.context.clear()
    
    def clear_messages(self) -> None:
        """Clear all messages."""
        self.messages.clear()
    
    def clear_results(self) -> None:
        """Clear all results."""
        self.results.clear()
    
    def clear_all(self) -> None:
        """Clear all data."""
        self.clear_messages()
        self.clear_results()
        self.clear_context()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the shared memory state."""
        return {
            "registered_agents": len(self.agent_ids),
            "total_messages": len(self.messages),
            "total_results": len(self.results),
            "context_keys": list(self.context.keys()),
        }








