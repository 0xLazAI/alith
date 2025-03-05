from dataclasses import dataclass, field
from typing import List, Union, Callable, Optional
from .tool import Tool, create_delegate_tool
from .store import Store
from ._alith import DelegateAgent as _DelegateAgent
from ._alith import DelegateTool as _DelegateTool


@dataclass
class Agent:
    name: str
    model: str
    preamble: Optional[str] = field(default_factory=str)
    api_key: Optional[str] = field(default_factory=str)
    base_url: Optional[str] = field(default_factory=str)
    tools: List[Union[Tool, Callable, _DelegateTool]] = field(default_factory=list)
    mcp_config_path: Optional[str] = field(default_factory=str)
    store: Optional[Store] = None

    def prompt(self, prompt: str) -> str:
        tools = [
            create_delegate_tool(tool) if isinstance(tool, Callable) else tool
            for tool in self.tools or []
        ]
        agent = _DelegateAgent(
            self.name,
            self.model,
            self.api_key,
            self.base_url,
            self.preamble,
            tools,
            self.mcp_config_path,
        )
        if self.store:
            docs = self.store.search(prompt)
            prompt = "{}\n\n<attachments>\n{}</attachments>\n".format(
                prompt, "".join(docs)
            )
        return agent.prompt(prompt)
