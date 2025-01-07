from dataclasses import dataclass, field
from typing import List, Union, Callable
from .tool import Tool, create_delegate_tool
from ._alith import DelegateAgent as _DelegateAgent
from ._alith import DelegateTool as _DelegateTool


@dataclass
class Agent:
    name: str
    model: str
    tools: List[Union[Tool, Callable, _DelegateTool]] = field(default_factory=list)

    def prompt(self, prompt: str) -> str:
        tools = [
            create_delegate_tool(tool) if isinstance(tool, Callable) else tool
            for tool in self.tools or []
        ]
        agent = _DelegateAgent(self.name, self.model, tools)
        return agent.prompt(prompt)
