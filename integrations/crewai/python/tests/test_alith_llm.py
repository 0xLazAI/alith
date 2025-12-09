"""Unit tests for AlithLLM custom LLM provider."""

import os
from unittest.mock import MagicMock, patch

import pytest
from alith import Agent

from alith_crewai import AlithLLM


class TestAlithLLM:
    """Test suite for AlithLLM class."""
    
    def test_initialization(self):
        """Test AlithLLM initialization."""
        # Create mock agent
        agent = MagicMock(spec=Agent)
        agent.model = "test-model"
        
        # Create LLM
        llm = AlithLLM(agent)
        
        # Verify
        assert llm.agent == agent
        assert llm.model == "test-model"
    
    def test_initialization_with_no_model(self):
        """Test AlithLLM initialization when agent has no model."""
        agent = MagicMock(spec=Agent)
        agent.model = None
        
        llm = AlithLLM(agent)
        
        assert llm.model == "alith"
    
    def test_call_with_string(self):
        """Test calling LLM with string prompt."""
        # Create mock agent
        agent = MagicMock(spec=Agent)
        agent.model = "test-model"
        agent.prompt.return_value = "Test response"
        
        # Create LLM and call
        llm = AlithLLM(agent)
        response = llm.call("Test prompt")
        
        # Verify
        assert response == "Test response"
        agent.prompt.assert_called_once_with("Test prompt")
    
    def test_call_with_messages_single(self):
        """Test calling LLM with single message."""
        # Create mock agent
        agent = MagicMock(spec=Agent)
        agent.model = "test-model"
        agent.prompt.return_value = "Response"
        
        # Create LLM and call
        llm = AlithLLM(agent)
        messages = [{"role": "user", "content": "Hello"}]
        response = llm.call(messages)
        
        # Verify
        assert response == "Response"
        agent.prompt.assert_called_once_with("Hello")
    
    def test_call_with_messages_multiple(self):
        """Test calling LLM with multiple messages."""
        # Create mock agent
        agent = MagicMock(spec=Agent)
        agent.prompt.return_value = "Response"
        
        # Create LLM and call
        llm = AlithLLM(agent)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]
        response = llm.call(messages)
        
        # Verify system messages are filtered out
        expected_prompt = "User: Hello\n\nAssistant: Hi there\n\nUser: How are you?"
        agent.prompt.assert_called_once_with(expected_prompt)
    
    def test_messages_to_prompt(self):
        """Test message format conversion."""
        agent = MagicMock(spec=Agent)
        llm = AlithLLM(agent)
        
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second message"}
        ]
        
        prompt = llm._messages_to_prompt(messages)
        
        expected = "User: First message\n\nAssistant: First response\n\nUser: Second message"
        assert prompt == expected
    
    def test_str_representation(self):
        """Test string representation."""
        agent = MagicMock(spec=Agent)
        agent.model = "test-model"
        llm = AlithLLM(agent)
        
        assert str(llm) == "AlithLLM(model=test-model)"
    
    def test_repr(self):
        """Test detailed representation."""
        agent = MagicMock(spec=Agent)
        agent.model = "test-model"
        llm = AlithLLM(agent)
        
        repr_str = repr(llm)
        assert "AlithLLM" in repr_str
        assert "test-model" in repr_str
