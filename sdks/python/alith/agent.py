import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

import requests

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


@dataclass
class MultimodalAgent(Agent):
    """Agent with image support for multimodal models."""

    def prompt(
        self, prompt: str, images: Optional[List[Union[str, Path, bytes]]] = None
    ) -> str:
        """Send prompt with optional images.
        
        Args:
            prompt: Text prompt to send to the agent.
            images: Optional list of image paths or bytes to include.
        
        Returns:
            Agent response as string.
        """
        if images:
            return self._prompt_with_images(prompt, images)
        return super().prompt(prompt)

    def _encode_image(self, image_input: Union[str, Path, bytes]) -> str:
        """Encode image to base64 data URL.
        
        Args:
            image_input: Image path (str/Path) or raw image bytes.
        
        Returns:
            Base64-encoded data URL string.
        
        Raises:
            FileNotFoundError: If image path does not exist.
        """
        if isinstance(image_input, bytes):
            image_data = image_input
            mime_type = 'image/png'
        else:
            image_path = Path(image_input)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            with open(image_path, "rb") as f:
                image_data = f.read()
            ext = image_path.suffix.lower()
            mime_types = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
            }
            mime_type = mime_types.get(ext, 'image/png')
        encoded = base64.b64encode(image_data).decode('utf-8')
        return f"data:{mime_type};base64,{encoded}"

    def _prompt_with_images(
        self, prompt: str, images: List[Union[str, Path, bytes]]
    ) -> str:
        """Handle multimodal prompt with images.
        
        Args:
            prompt: Text prompt to send.
            images: List of image paths or bytes to include.
        
        Returns:
            Agent response as string.
        """
        if self.store:
            docs = self.store.search(prompt)
            prompt = "{}\n\n<attachments>\n{}</attachments>\n".format(
                prompt, "".join(docs)
            )
        
        image_urls = [self._encode_image(img) for img in images]
        content = [{"type": "text", "text": prompt}]
        for image_url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.extra_headers:
            headers.update(self.extra_headers)
        messages = [{"role": "user", "content": content}]
        if self.preamble:
            messages.insert(0, {"role": "system", "content": self.preamble})
        if self.memory:
            for msg in self.memory.messages():
                messages.insert(-1, {"role": msg.role, "content": msg.content})
        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={"model": self.model, "messages": messages}
        )
        response.raise_for_status()
        result_content = response.json()["choices"][0]["message"]["content"]
        if self.memory:
            self.memory.add_user_message(prompt)
            self.memory.add_ai_message(result_content)
        return result_content
