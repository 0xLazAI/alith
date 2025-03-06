mod chat_template_prompt;
mod concatenator;
mod llm_prompt;
mod local_content;
mod openai_prompt;
mod prompt_message;
mod token_count;

#[allow(unused_imports)]
pub(crate) use anyhow::{anyhow, bail, Error, Result};

pub use chat_template_prompt::{apply_chat_template, ChatTemplatePrompt};
pub use concatenator::{TextConcatenator, TextConcatenatorTrait};
pub use llm_prompt::LLMPrompt;
pub use openai_prompt::OpenAIPrompt;
pub use prompt_message::{PromptMessage, PromptMessageType};
use std::sync::Arc;
pub use token_count::{check_and_get_max_tokens, MaxTokenState, RequestTokenLimitError};

/// Implement for your tokenizer to use with this library.
pub trait PromptTokenizer: Send + Sync {
    fn tokenize(&self, input: &str) -> Vec<u32>;

    fn count_tokens(&self, input: &str) -> u32;
}

impl PromptTokenizer for Arc<dyn PromptTokenizer> {
    fn tokenize(&self, input: &str) -> Vec<u32> {
        (**self).tokenize(input)
    }

    fn count_tokens(&self, input: &str) -> u32 {
        (**self).count_tokens(input)
    }
}
