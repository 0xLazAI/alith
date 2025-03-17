pub mod backend_builders;
pub mod basic_completion;
pub mod components;
pub mod embeddings;
pub mod prelude;
pub mod primitives;
pub mod workflows;

#[allow(unused_imports)]
pub(crate) use alith_devices::logging::{i_ln, i_lns, i_nln, i_nlns};
#[allow(unused_imports)]
pub(crate) use anyhow::{Error, Result, anyhow, bail};
pub use prelude::*;
#[allow(unused_imports)]
pub(crate) use tracing::{Level, debug, error, info, span, trace, warn};

pub use alith_interface as interface;
pub use alith_interface::llms::LLMBackend;

use backend_builders::anthropic::AnthropicBackendBuilder;
use backend_builders::openai::OpenAIBackendBuilder;
use backend_builders::perplexity::PerplexityBackendBuilder;
use basic_completion::BasicCompletion;
use embeddings::Embeddings;
use std::sync::Arc;
use workflows::basic_primitive::BasicPrimitiveWorkflowBuilder;
use workflows::nlp::Nlp;
use workflows::reason::ReasonWorkflowBuilder;

#[derive(Clone)]
pub struct LLMClient {
    pub backend: Arc<LLMBackend>,
}

impl LLMClient {
    #[inline]
    pub fn new(backend: Arc<LLMBackend>) -> Self {
        Self { backend }
    }

    /// Creates a new instance of the [`OpenAIBackendBuilder`]. This builder that allows you to specify the model and other parameters. It is converted to an `LLMClient` instance using the `init` method.
    #[inline]
    pub fn openai() -> OpenAIBackendBuilder {
        OpenAIBackendBuilder::default()
    }

    /// Creates a new instance of the [`AnthropicBackendBuilder`]. This builder that allows you to specify the model and other parameters. It is converted to an `LLMClient` instance using the `init` method.
    #[inline]
    pub fn anthropic() -> AnthropicBackendBuilder {
        AnthropicBackendBuilder::default()
    }

    /// Creates a new instance of the [`PerplexityBackendBuilder`]. This builder that allows you to specify the model and other parameters. It is converted to an `LLMClient` instance using the `init` method.
    #[inline]
    pub fn perplexity() -> PerplexityBackendBuilder {
        PerplexityBackendBuilder::default()
    }

    #[inline]
    pub fn embeddings(&self) -> Embeddings {
        Embeddings::new(self.backend.clone())
    }

    #[inline]
    pub fn basic_completion(&self) -> BasicCompletion {
        BasicCompletion::new(self.backend.clone())
    }

    #[inline]
    pub fn basic_primitive(&self) -> BasicPrimitiveWorkflowBuilder {
        BasicPrimitiveWorkflowBuilder::new(self.backend.clone())
    }

    #[inline]
    pub fn reason(&self) -> ReasonWorkflowBuilder {
        ReasonWorkflowBuilder::new(self.backend.clone())
    }

    #[inline]
    pub fn nlp(&self) -> Nlp {
        Nlp::new(self.backend.clone())
    }

    #[inline]
    pub fn shutdown(&self) {
        self.backend.shutdown();
    }

    #[inline]
    pub fn completion_request(&self) -> CompletionRequest {
        CompletionRequest::new(self.backend.clone())
    }

    #[inline]
    pub fn embeddings_request(&self) -> EmbeddingsRequest {
        EmbeddingsRequest::new(self.backend.clone())
    }
}
