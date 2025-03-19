use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Represents the type of a message.
#[derive(PartialEq, Eq, Serialize, Deserialize, Debug, Clone)]
pub enum MessageType {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "human")]
    Human,
    #[serde(rename = "ai")]
    AI,
    #[serde(rename = "tool")]
    Tool,
}

impl Default for MessageType {
    /// Default message type is `SystemMessage`.
    fn default() -> Self {
        Self::System
    }
}

impl MessageType {
    /// Converts the `MessageType` to a string representation.
    pub fn type_string(&self) -> String {
        match self {
            MessageType::System => "system".to_owned(),
            MessageType::Human => "user".to_owned(),
            MessageType::AI => "assistant".to_owned(),
            MessageType::Tool => "tool".to_owned(),
        }
    }
}

/// Represents a message with content, type, optional ID, and optional tool calls.
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct Message {
    pub content: String,
    pub message_type: MessageType,
    pub id: Option<String>,
    pub tool_calls: Option<Value>,
}

impl Message {
    /// Creates a new human message with the given content.
    pub fn new_human_message<T: std::fmt::Display>(content: T) -> Self {
        Message {
            content: content.to_string(),
            message_type: MessageType::Human,
            id: None,
            tool_calls: None,
        }
    }

    /// Creates a new system message with the given content.
    pub fn new_system_message<T: std::fmt::Display>(content: T) -> Self {
        Message {
            content: content.to_string(),
            message_type: MessageType::System,
            id: None,
            tool_calls: None,
        }
    }

    /// Creates a new tool message with the given content and ID.
    pub fn new_tool_message<T: std::fmt::Display, S: Into<String>>(content: T, id: S) -> Self {
        Message {
            content: content.to_string(),
            message_type: MessageType::Tool,
            id: Some(id.into()),
            tool_calls: None,
        }
    }

    /// Creates a new =AI message with the given content.
    pub fn new_ai_message<T: std::fmt::Display>(content: T) -> Self {
        Message {
            content: content.to_string(),
            message_type: MessageType::AI,
            id: None,
            tool_calls: None,
        }
    }

    /// Adds tool calls to the message.
    pub fn with_tool_calls(mut self, tool_calls: Value) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    /// Deserializes a `Value` into a vector of `Message` objects.
    pub fn messages_from_value(value: &Value) -> Result<Vec<Message>, serde_json::error::Error> {
        serde_json::from_value(value.clone())
    }

    /// Converts a slice of `Message` objects into a single string representation.
    pub fn messages_to_string(messages: &[Message]) -> String {
        messages
            .iter()
            .map(|m| format!("{:?}: {}", m.message_type, m.content))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

/// A trait representing a memory storage for messages.
pub trait Memory: Send + Sync {
    /// Returns all messages stored in memory.
    fn messages(&self) -> Vec<Message>;

    /// Adds a user (human) message to the memory.
    fn add_user_message(&mut self, message: &dyn std::fmt::Display) {
        self.add_message(Message::new_human_message(message.to_string()));
    }

    /// Adds an AI (LLM) message to the memory.
    fn add_ai_message(&mut self, message: &dyn std::fmt::Display) {
        self.add_message(Message::new_ai_message(message.to_string()));
    }

    /// Adds a message to the memory.
    fn add_message(&mut self, message: Message);

    /// Clears all messages from memory.
    fn clear(&mut self);

    /// Converts the memory's messages to a string representation.
    fn to_string(&self) -> String {
        self.messages()
            .iter()
            .map(|msg| format!("{}: {}", msg.message_type.type_string(), msg.content))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

/// Converts a type implementing `Memory` into a boxed trait object.
impl<M> From<M> for Box<dyn Memory>
where
    M: Memory + 'static,
{
    fn from(memory: M) -> Self {
        Box::new(memory)
    }
}

/// A memory structure that stores messages in a sliding window buffer.
pub struct WindowBufferMemory {
    window_size: usize,
    messages: Vec<Message>,
}

impl Default for WindowBufferMemory {
    /// Creates a default `WindowBufferMemory` with a window size of 10.
    fn default() -> Self {
        Self::new(10)
    }
}

impl WindowBufferMemory {
    /// Creates a new `WindowBufferMemory` with the specified window size.
    pub fn new(window_size: usize) -> Self {
        Self {
            messages: Vec::new(),
            window_size,
        }
    }

    /// Get the window size.
    #[inline]
    pub fn window_size(&self) -> usize {
        self.window_size
    }
}

/// Converts `WindowBufferMemory` into an `Arc<dyn Memory>`.
impl From<WindowBufferMemory> for Arc<dyn Memory> {
    fn from(val: WindowBufferMemory) -> Self {
        Arc::new(val)
    }
}

/// Converts `WindowBufferMemory` into an `Arc<Mutex<dyn Memory>>`.
impl From<WindowBufferMemory> for Arc<Mutex<dyn Memory>> {
    fn from(val: WindowBufferMemory) -> Self {
        Arc::new(Mutex::new(val))
    }
}

impl Memory for WindowBufferMemory {
    /// Returns all messages in the buffer.
    fn messages(&self) -> Vec<Message> {
        self.messages.clone()
    }

    /// Adds a message to the buffer, removing the oldest message if the buffer is full.
    fn add_message(&mut self, message: Message) {
        if self.messages.len() >= self.window_size {
            self.messages.remove(0);
        }
        self.messages.push(message);
    }

    /// Clears all messages from the buffer.
    fn clear(&mut self) {
        self.messages.clear();
    }
}
