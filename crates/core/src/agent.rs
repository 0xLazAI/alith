use uuid::Uuid;

use crate::chat::Completion;
use crate::knowledge::Knowledge;
use crate::tool::Tool;

pub struct Agent<M: Completion> {
    /// The model to use.
    pub model: M,
    /// The tools to use.
    pub tools: Vec<Box<dyn Tool>>,
    /// The id of the agent.
    pub id: Uuid,
    /// The name of the agent.
    pub name: String,
    /// Role of the agent.
    pub role: String,
    /// System prompt
    pub preamble: String,
    /// System format for the agent.
    pub system_template: String,
    /// Prompt format for the agent.
    pub prompt_template: String,
    /// Response format for the agent.
    pub response_template: String,
    /// Enable or disable the verbose mode.
    pub verbose: bool,
    /// The maximum requests per minute for the completion text.
    pub max_rpm: Option<usize>,
    /// Maximum number of tokens for the completion.
    pub max_tokens: Option<usize>,
    /// Maximum execution time for an agent to execute a task.
    pub max_execution_time: Option<usize>,
    /// Knowledge sources for the agent.
    pub knowledges: Vec<Box<dyn Knowledge>>,
}
