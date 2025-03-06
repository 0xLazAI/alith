#[allow(unused_imports)]
pub(crate) use anyhow::{anyhow, bail, Error, Result};
#[allow(unused_imports)]
pub(crate) use tracing::{debug, error, info, span, trace, warn, Level};

pub mod llms;
pub mod requests;

pub struct LLMInterface {}

// These are examples and bare minimum implementations. For full featured implementation see the alith-client crate.
impl LLMInterface {
    pub fn openai() -> llms::api::openai::builder::OpenAIBackendBuilder {
        llms::api::openai::builder::OpenAIBackendBuilder::default()
    }

    pub fn anthropic() -> llms::api::anthropic::builder::AnthropicBackendBuilder {
        llms::api::anthropic::builder::AnthropicBackendBuilder::default()
    }

    pub fn perplexity() -> llms::api::perplexity::builder::PerplexityBackendBuilder {
        llms::api::perplexity::builder::PerplexityBackendBuilder::default()
    }
}
