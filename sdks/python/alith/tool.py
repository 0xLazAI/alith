from dataclasses import dataclass
from typing import Callable
from ._alith import DelegateTool as _DelegateTool
from inspect import Parameter
from pydantic import create_model
import json
import ctypes
import inspect


CFUNC_TYPE = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_char_p)


@dataclass
class Tool:
    name: str
    description: str
    version: str
    author: str


def get_function_schema(f: Callable) -> str:
    """Generate a JSON schema for the function's parameters."""
    kw = {
        n: (o.annotation, ... if o.default == Parameter.empty else o.default)
        for n, o in inspect.signature(f).parameters.items()
    }
    f_model = create_model(f"input for `{f.__name__}`", **kw)
    schema = {
        "name": f.__name__,
        "description": f.__doc__,
        "parameters": f_model.model_json_schema(),
    }
    return schema


def create_delegate_tool(func: Callable) -> _DelegateTool:
    """Create a DelegateTool instance from a Python function."""
    # Get function name and description

    # Get function parameters as JSON schema
    schema = get_function_schema(func)

    def wrapper(args: ctypes.c_char_p) -> bytes:
        """Wrapper function to match the extern "C" signature."""
        args_str = ctypes.cast(args, ctypes.c_char_p).value.decode("utf-8")
        print(args_str)
        args_json = json.loads(args_str)
        result = func(**args_json)
        result_json = json.dumps(result)
        return result_json.encode("utf-8")

    cfunc_wrapper = CFUNC_TYPE(wrapper)
    # Get function address (C pointer)
    func_agent = ctypes.cast(cfunc_wrapper, ctypes.c_void_p).value

    # Create and return DelegateTool instance
    return _DelegateTool(
        name=schema["name"],
        version="1.0.0",  # Default version
        description=schema["description"],
        parameters=json.dumps(schema["parameters"]),
        author="Unknown",  # Default author
        func_agent=func_agent,
    )
