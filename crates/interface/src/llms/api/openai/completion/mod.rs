mod req;
mod res;

pub use req::{CompletionRequestMessage, OpenAICompletionRequest, OpenAIToolDefinition};
pub use res::{
    ChatChoice, ChatCompletionResponseMessage, CompletionUsage, FinishReason,
    OpenAICompletionResponse, Role,
};
