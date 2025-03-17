#[allow(unused_imports)]
pub(crate) use anyhow::{Error, Result, anyhow, bail};
#[allow(unused_imports)]
pub(crate) use tracing::{Level, debug, error, info, span, trace, warn};

pub mod llms;
pub mod requests;

use llms::api::anthropic::builder::AnthropicBackendBuilder;
use llms::api::openai::builder::OpenAIBackendBuilder;
use llms::api::perplexity::builder::PerplexityBackendBuilder;

pub struct LLMInterface;

// These are examples and bare minimum implementations. For full featured implementation see the alith-client crate.
impl LLMInterface {
    #[inline]
    pub fn openai() -> OpenAIBackendBuilder {
        OpenAIBackendBuilder::default()
    }

    #[inline]
    pub fn anthropic() -> AnthropicBackendBuilder {
        AnthropicBackendBuilder::default()
    }

    #[inline]
    pub fn perplexity() -> PerplexityBackendBuilder {
        PerplexityBackendBuilder::default()
    }
}
