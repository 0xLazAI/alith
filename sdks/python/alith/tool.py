from typing import Callable
from ._alith import DelegateTool as _DelegateTool
from inspect import Parameter
from pydantic import create_model, BaseModel, Field
import json
import ctypes
import inspect
import warnings

warnings.filterwarnings(action="ignore", category=RuntimeWarning)


CFUNC_TYPE = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_char_p)


class Tool(BaseModel):
    """Represents tool that can be performed by an agent."""

    name: str
    description: str
    parameters: type[BaseModel] | None = None
    version: str = "1.0.0"
    author: str = "Unknown"
    handler: Callable = Field(..., exclude=True)

    def to_delegate_tool(self) -> _DelegateTool:
        if self.parameters:
            parameters = self.parameters.model_json_schema()
        else:
            parameters = {}

        def wrapper(args: ctypes.c_char_p) -> bytes:
            """Wrapper function to match the extern "C" signature."""
            args_str = ctypes.cast(args, ctypes.c_char_p).value.decode("utf-8")
            args_json = json.loads(args_str)
            result = self.handler(**args_json)
            result_json = json.dumps(result)
            return result_json.encode("utf-8")

        cfunc_wrapper = CFUNC_TYPE(wrapper)
        # Get function address (C pointer)
        func_agent = ctypes.cast(cfunc_wrapper, ctypes.c_void_p).value

        # Create and return DelegateTool instance
        return _DelegateTool(
            name=self.name,
            version=self.version,
            description=self.description,
            parameters=json.dumps(parameters),
            author=self.author,
            func_agent=func_agent,
        )


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


def create_delegate_tool(
    func: Callable, version: str = "1.0.0", author: str = "Unknown"
) -> _DelegateTool:
    """Create a DelegateTool instance from a Python function."""
    # Get the function JSON schema
    schema = get_function_schema(func)
    # Get function name and description
    name, description = schema["name"], schema["description"]
    # Get function parameters as JSON schema
    parameters = schema["parameters"]

    def wrapper(args: ctypes.c_char_p) -> bytes:
        """Wrapper function to match the extern "C" signature."""

        args_str = ctypes.cast(args, ctypes.c_char_p).value.decode("utf-8")
        args_json = json.loads(args_str)
        result = func(**args_json)
        result_json = json.dumps(result)
        return result_json.encode("utf-8")

    cfunc_wrapper = CFUNC_TYPE(wrapper)
    # Get function address (C pointer)
    func_agent = ctypes.cast(cfunc_wrapper, ctypes.c_void_p).value

    # Create and return DelegateTool instance
    return _DelegateTool(
        name=name,
        version=version,
        description=description,
        parameters=json.dumps(parameters),
        author=author,
        func_agent=func_agent,
    )
