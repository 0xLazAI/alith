"""Utilities for converting tools between CrewAI and Alith formats."""

from typing import Any, Callable, Dict, Optional

from alith import Tool as AlithTool
from pydantic import BaseModel, Field, create_model


class CrewAIToolWrapper:
    """Wrapper to make Alith tools compatible with CrewAI.
    
    This class wraps an Alith Tool to provide the interface expected
    by CrewAI's tool system.
    """
    
    def __init__(self, alith_tool: AlithTool):
        """Initialize the wrapper.
        
        Args:
            alith_tool: The Alith tool to wrap
        """
        self.name = alith_tool.name
        self.description = alith_tool.description
        self._handler = alith_tool.handler
        
        # Create Pydantic model for args_schema
        if alith_tool.parameters:
            # Extract properties from the parameters BaseModel
            schema = alith_tool.parameters.model_json_schema()
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            # Build field definitions for create_model
            fields = {}
            for field_name, field_info in properties.items():
                field_type = self._json_type_to_python(field_info.get("type", "string"))
                field_default = ... if field_name in required else None
                field_description = field_info.get("description", "")
                
                fields[field_name] = (
                    field_type,
                    Field(default=field_default, description=field_description)
                )
            
            self.args_schema = create_model(
                f"{self.name}Args",
                **fields
            )
        else:
            # Create empty schema
            self.args_schema = create_model(f"{self.name}Args")
    
    def _json_type_to_python(self, json_type: str) -> type:
        """Convert JSON schema type to Python type.
        
        Args:
            json_type: JSON schema type string
        
        Returns:
            Corresponding Python type
        """
        type_mapping = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return type_mapping.get(json_type, str)
    
    def _run(self, **kwargs) -> str:
        """Execute the tool.
        
        Args:
            **kwargs: Tool parameters
        
        Returns:
            Tool execution result as string
        """
        result = self._handler(**kwargs)
        
        # Ensure result is a string
        if not isinstance(result, str):
            return str(result)
        return result


def convert_alith_tool_to_crewai(alith_tool: AlithTool) -> CrewAIToolWrapper:
    """Convert an Alith tool to CrewAI-compatible format.
    
    Args:
        alith_tool: The Alith Tool instance to convert
    
    Returns:
        A CrewAI-compatible tool wrapper
    
    Example:
        >>> from alith import Tool
        >>> 
        >>> alith_tool = Tool(
        ...     name="search",
        ...     description="Search the web",
        ...     handler=lambda query: f"Results for: {query}"
        ... )
        >>> crewai_tool = convert_alith_tool_to_crewai(alith_tool)
    """
    return CrewAIToolWrapper(alith_tool)


def convert_crewai_tool_to_alith(crewai_tool: Any) -> AlithTool:
    """Convert a CrewAI tool to Alith format.
    
    Args:
        crewai_tool: A CrewAI BaseTool instance
    
    Returns:
        An Alith Tool instance
    
    Example:
        >>> from crewai_tools import SerperDevTool
        >>> 
        >>> crewai_tool = SerperDevTool()
        >>> alith_tool = convert_crewai_tool_to_alith(crewai_tool)
    """
    # Get tool attributes
    name = getattr(crewai_tool, "name", "unknown_tool")
    description = getattr(crewai_tool, "description", "")
    args_schema = getattr(crewai_tool, "args_schema", None)
    
    # Create handler that calls the CrewAI tool's _run method
    def handler(**kwargs) -> str:
        result = crewai_tool._run(**kwargs)
        if not isinstance(result, str):
            return str(result)
        return result
    
    # Create Alith tool
    return AlithTool(
        name=name,
        description=description,
        parameters=args_schema,
        handler=handler
    )
