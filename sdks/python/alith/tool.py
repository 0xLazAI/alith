from dataclasses import dataclass
from typing import Callable
from ._alith import DelegateTool as _DelegateTool
import json
import ctypes

CFUNC_TYPE = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_char_p)


@dataclass
class Tool:
    name: str
    description: str
    version: str
    author: str


def get_function_schema(func: Callable) -> str:
    """Generate a JSON schema for the function's parameters."""
    import inspect

    sig = inspect.signature(func)
    schema = {"type": "object", "properties": {}, "required": []}
    for name, param in sig.parameters.items():
        param_type = (
            str(param.annotation)
            if param.annotation != inspect.Parameter.empty
            else "any"
        )
        schema["properties"][name] = {"type": param_type}
        if param.default == inspect.Parameter.empty:
            schema["required"].append(name)
    return json.dumps(schema)


def create_delegate_tool(func: Callable) -> _DelegateTool:
    """Create a DelegateTool instance from a Python function."""
    # Get function name and description
    name = func.__name__
    description = func.__doc__.strip() if func.__doc__ else ""

    # Get function parameters as JSON schema
    parameters = get_function_schema(func)

    def wrapper(args: ctypes.c_char_p) -> ctypes.c_char_p:
        """Wrapper function to match the extern "C" signature."""
        args_str = ctypes.cast(args, ctypes.c_char_p).value.decode("utf-8")
        args_json = json.loads(args_str)
        result = func(**args_json)
        result_json = json.dumps(result)
        return ctypes.c_char_p(result_json.encode("utf-8"))

    cfunc_wrapper = CFUNC_TYPE(wrapper)
    # Get function address (C pointer)
    func_agent = ctypes.cast(cfunc_wrapper, ctypes.c_void_p).value

    # Create and return DelegateTool instance
    return _DelegateTool(
        name=name,
        version="1.0.0",  # Default version
        description=description,
        parameters=parameters,
        author="Unknown",  # Default author
        func_agent=func_agent,
    )
