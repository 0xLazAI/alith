from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union, Dict


# Define the MessageType enum
class MessageType(Enum):
    SYSTEM = "system"
    HUMAN = "human"
    AI = "ai"
    TOOL = "tool"

    @classmethod
    def default(cls):
        return cls.SYSTEM

    def type_string(self) -> str:
        return {
            MessageType.SYSTEM: "system",
            MessageType.HUMAN: "user",
            MessageType.AI: "assistant",
            MessageType.TOOL: "tool",
        }[self]


# Define the Message class
@dataclass
class Message:
    content: str
    message_type: MessageType = MessageType.SYSTEM
    id: Optional[str] = None
    tool_calls: Optional[Dict] = field(default=None)

    @classmethod
    def new_human_message(cls, content: str) -> "Message":
        return cls(content=content, message_type=MessageType.HUMAN)

    @classmethod
    def new_system_message(cls, content: str) -> "Message":
        return cls(content=content, message_type=MessageType.SYSTEM)

    @classmethod
    def new_tool_message(cls, content: str, id: str) -> "Message":
        return cls(content=content, message_type=MessageType.TOOL, id=id)

    @classmethod
    def new_ai_message(cls, content: str) -> "Message":
        return cls(content=content, message_type=MessageType.AI)

    def with_tool_calls(self, tool_calls: Dict) -> "Message":
        self.tool_calls = tool_calls
        return self

    @staticmethod
    def messages_from_value(value: Union[str, Dict, List]) -> List["Message"]:
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, dict):
            value = [value]
        return [Message(**item) for item in value]

    @staticmethod
    def messages_to_string(messages: List["Message"]) -> str:
        return "\n".join(
            f"{msg.message_type.type_string()}: {msg.content}" for msg in messages
        )


# Define the Memory abstract class
class Memory(ABC):
    @abstractmethod
    def messages(self) -> List[Message]:
        pass

    def add_user_message(self, message: str):
        self.add_message(Message.new_human_message(message))

    def add_ai_message(self, message: str):
        self.add_message(Message.new_ai_message(message))

    @abstractmethod
    def add_message(self, message: Message):
        pass

    @abstractmethod
    def clear(self):
        pass

    def to_string(self) -> str:
        return "\n".join(
            f"{msg.message_type.type_string()}: {msg.content}"
            for msg in self.messages()
        )


# Define the WindowBufferMemory class
class WindowBufferMemory(Memory):
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._messages: List[Message] = []

    def messages(self) -> List[Message]:
        return self._messages.copy()

    def add_message(self, message: Message):
        if len(self._messages) >= self.window_size:
            self._messages.pop(0)
        self._messages.append(message)

    def clear(self):
        self._messages.clear()
