import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Union


# Define the Message class
@dataclass
class MessageBuilder:
    @staticmethod
    def new_human_message(content: str):
        from ._alith import Message

        return Message(role="user", content=content)

    @staticmethod
    def new_system_message(content: str):
        from ._alith import Message

        return Message(role="system", content=content)

    @staticmethod
    def new_tool_message(content: str):
        from ._alith import Message

        return Message(role="tool", content=content)

    @staticmethod
    def new_ai_message(content: str):
        from ._alith import Message

        return Message(role="assistant", content=content)

    @staticmethod
    def messages_from_value(value: Union[str, Dict, List]):
        from ._alith import Message

        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, dict):
            value = [value]
        return [Message(**item) for item in value]

    @staticmethod
    def messages_to_string(messages) -> str:
        return "\n".join(f"{msg.role}: {msg.content}" for msg in messages)


# Define the Memory abstract class
class Memory(ABC):
    @abstractmethod
    def messages(self):
        pass

    def add_user_message(self, message: str):
        self.add_message(MessageBuilder.new_human_message(message))

    def add_ai_message(self, message: str):
        self.add_message(MessageBuilder.new_ai_message(message))

    @abstractmethod
    def add_message(self, message):
        pass

    @abstractmethod
    def clear(self):
        pass

    def to_string(self) -> str:
        return "\n".join(f"{msg.role}: {msg.content}" for msg in self.messages())


# Define the WindowBufferMemory class
class WindowBufferMemory(Memory):
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._messages = []

    def messages(self):
        return self._messages.copy()

    def add_message(self, message):
        if len(self._messages) >= self.window_size:
            self._messages.pop(0)
        self._messages.append(message)

    def clear(self):
        self._messages.clear()
