"""Custom LLM provider that uses Alith agents as the backend for CrewAI."""

from typing import Any, Dict, List, Optional, Union

from alith import Agent


class AlithLLM:
    """Custom LLM that uses Alith Agent as the inference backend.
    
    This class wraps an Alith agent to make it compatible with CrewAI's
    LLM interface, enabling high-performance inference and Web3 capabilities
    within CrewAI workflows.
    
    Args:
        agent: The Alith agent instance to use as the backend
        **kwargs: Additional parameters (for compatibility)
    
    Example:
        >>> from alith import Agent
        >>> from alith_crewai import AlithLLM
        >>> 
        >>> alith_agent = Agent(
        ...     model="llama-3.3-70b-versatile",
        ...     preamble="You are a helpful assistant"
        ... )
        >>> llm = AlithLLM(alith_agent)
    """
    
    def __init__(self, agent: Agent, **kwargs):
        """Initialize the Alith LLM wrapper.
        
        Args:
            agent: Alith agent instance
            **kwargs: Additional parameters for compatibility
        """
        self.agent = agent
        self.model = agent.model or "alith"
        self._kwargs = kwargs
    
    def call(
        self, 
        messages: Union[List[Dict[str, str]], str],
        **kwargs
    ) -> str:
        """Main inference method called by CrewAI.
        
        Args:
            messages: Either a list of message dicts or a string prompt
            **kwargs: Additional generation parameters
        
        Returns:
            The model's response as a string
        """
        # Convert messages to prompt format
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = self._messages_to_prompt(messages)
        
        # Call Alith agent
        response = self.agent.prompt(prompt)
        
        return response
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert CrewAI message format to Alith prompt format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
        
        Returns:
            Formatted prompt string
        """
        # Filter out system messages as they're handled by Alith's preamble
        user_messages = [
            msg for msg in messages 
            if msg.get("role") != "system"
        ]
        
        # If only one message, return its content
        if len(user_messages) == 1:
            return user_messages[0].get("content", "")
        
        # For multiple messages, format as conversation
        prompt_parts = []
        for msg in user_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(content)
        
        return "\n\n".join(prompt_parts)
    
    def supports_stop_words(self) -> bool:
        """Check if LLM supports stop words.
        
        Returns:
            False - Alith doesn't currently expose stop word support
        """
        return False
    
    def supports_function_calling(self) -> bool:
        """Check if LLM supportfunction calling.
        
        Returns:
            False - Using tools via Alith's tool system instead
        """
        return False
    
    def __str__(self) -> str:
        """String representation of the LLM."""
        return f"AlithLLM(model={self.model})"
    
    def __repr__(self) -> str:
        """Detailed representation of the LLM."""
        return f"AlithLLM(agent={self.agent}, model={self.model})"
