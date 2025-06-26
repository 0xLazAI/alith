from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

from .memory import Memory
from .store import Store
from .tool import Tool, create_delegate_tool
from .types import Headers


@dataclass
class Agent:
    model: Optional[str] = field(default_factory=str)
    name: Optional[str] = field(default_factory=str)
    preamble: Optional[str] = field(default_factory=str)
    api_key: Optional[str] = field(default_factory=str)
    base_url: Optional[str] = field(default_factory=str)
    tools: List[Union[Tool, Callable]] = field(default_factory=list)
    mcp_config_path: Optional[str] = field(default_factory=str)
    store: Optional[Store] = None
    memory: Optional[Memory] = None
    extra_headers: Optional[Headers] = None

    def prompt(self, prompt: str) -> str:
        from ._alith import DelegateAgent as _DelegateAgent

        tools = [
            (
                create_delegate_tool(tool)
                if isinstance(tool, Callable)
                else tool.to_delegate_tool() if isinstance(tool, Tool) else tool
            )
            for tool in self.tools or []
        ]
        agent = _DelegateAgent(
            self.name or "",
            self.model or "",
            self.api_key,
            self.base_url,
            self.preamble,
            tools,
            self.extra_headers or dict(),
            self.mcp_config_path,
        )
        if self.store:
            docs = self.store.search(prompt)
            prompt = "{}\n\n<attachments>\n{}</attachments>\n".format(
                prompt, "".join(docs)
            )
        if self.memory:
            result = agent.chat(prompt, self.memory.messages())
            self.memory.add_user_message(prompt)
            self.memory.add_ai_message(result)
            return result
        else:
            return agent.prompt(prompt)
