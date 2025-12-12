"""Unit tests for tool conversion utilities."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from alith import Tool as AlithTool
from alith_crewai import convert_alith_tool_to_crewai, convert_crewai_tool_to_alith


class TestToolConversion:
    """Test suite for tool conversion functions."""
    
    def test_convert_alith_to_crewai_basic(self):
        """Test basic Alith to CrewAI tool conversion."""
        # Create Alith tool
        def handler(query: str) -> str:
            return f"Results for: {query}"
        
        alith_tool = AlithTool(
            name="search",
            description="Search the web",
            handler=handler
        )
        
        # Convert
        crewai_tool = convert_alith_tool_to_crewai(alith_tool)
        
        # Verify
        assert crewai_tool.name == "search"
        assert crewai_tool.description == "Search the web"
        assert hasattr(crewai_tool, "_run")
    
    def test_convert_alith_to_crewai_with_parameters(self):
        """Test Alith to CrewAI conversion with parameters."""
        # Create parameter schema
        class SearchParams(BaseModel):
            query: str = Field(description="Search query")
            limit: int = Field(default=10, description="Max results")
        
        def handler(query: str, limit: int = 10) -> str:
            return f"Found {limit} results for: {query}"
        
        alith_tool = AlithTool(
            name="search",
            description="Search tool",
            parameters=SearchParams,
            handler=handler
        )
        
        # Convert
        crewai_tool = convert_alith_tool_to_crewai(alith_tool)
        
        # Verify
        assert crewai_tool.name == "search"
        assert hasattr(crewai_tool, "args_schema")
        
        # Test execution
        result = crewai_tool._run(query="test", limit=5)
        assert "test" in result
        assert "5" in result
    
    def test_convert_crewai_to_alith(self):
        """Test CrewAI to Alith tool conversion."""
        # Create mock CrewAI tool
        crewai_tool = MagicMock()
        crewai_tool.name = "mock_tool"
        crewai_tool.description = "A mock tool"
        crewai_tool.args_schema = None
        crewai_tool._run.return_value = "Mock result"
        
        # Convert
        alith_tool = convert_crewai_tool_to_alith(crewai_tool)
        
        # Verify
        assert alith_tool.name == "mock_tool"
        assert alith_tool.description == "A mock tool"
        
        # Test handler
        result = alith_tool.handler()
        assert result == "Mock result"
    
    def test_convert_crewai_to_alith_with_schema(self):
        """Test CrewAI to Alith conversion with parameter schema."""
        # Create parameter schema
        class ToolParams(BaseModel):
            input_text: str
        
        # Create mock tool
        crewai_tool = MagicMock()
        crewai_tool.name = "process_text"
        crewai_tool.description = "Process text"
        crewai_tool.args_schema = ToolParams
        crewai_tool._run.return_value = "Processed"
        
        # Convert
        alith_tool = convert_crewai_tool_to_alith(crewai_tool)
        
        # Verify
        assert alith_tool.name == "process_text"
        assert alith_tool.parameters == ToolParams
        
        # Test handler
        result = alith_tool.handler(input_text="test")
        assert result == "Processed"
    
    def test_crewai_wrapper_json_type_conversion(self):
        """Test JSON type to Python type conversion."""
        from alith_crewai.tool_converter import CrewAIToolWrapper
        
        # Create mock tool
        alith_tool = MagicMock(spec=AlithTool)
        alith_tool.name = "test"
        alith_tool.description = "test"
        alith_tool.parameters = None
        
        wrapper = CrewAIToolWrapper(alith_tool)
        
        # Test type mappings
        assert wrapper._json_type_to_python("string") == str
        assert wrapper._json_type_to_python("number") == float
        assert wrapper._json_type_to_python("integer") == int
        assert wrapper._json_type_to_python("boolean") == bool
        assert wrapper._json_type_to_python("array") == list
        assert wrapper._json_type_to_python("object") == dict
    
    def test_tool_execution_returns_string(self):
        """Test that tool execution always returns string."""
        # Create Alith tool that returns non-string
        def handler() -> int:
            return 42
        
        alith_tool = AlithTool(
            name="number_tool",
            description="Returns a number",
            handler=handler
        )
        
        # Convert and execute
        crewai_tool = convert_alith_tool_to_crewai(alith_tool)
        result = crewai_tool._run()
        
        # Verify result is string
        assert isinstance(result, str)
        assert result == "42"
