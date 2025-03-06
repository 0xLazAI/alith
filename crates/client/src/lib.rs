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
pub(crate) use anyhow::{anyhow, bail, Error, Result};
pub use prelude::*;
#[allow(unused_imports)]
pub(crate) use tracing::{debug, error, info, span, trace, warn, Level};

pub use alith_interface as interface;
pub use alith_interface::llms::LlmBackend;

use std::sync::Arc;

#[derive(Clone)]
pub struct LlmClient {
    pub backend: Arc<LlmBackend>,
}

impl LlmClient {
    pub fn new(backend: Arc<LlmBackend>) -> Self {
        Self { backend }
    }

    /// Creates a new instance of the [`OpenAiBackendBuilder`]. This builder that allows you to specify the model and other parameters. It is converted to an `LlmClient` instance using the `init` method.
    pub fn openai() -> backend_builders::openai::OpenAiBackendBuilder {
        backend_builders::openai::OpenAiBackendBuilder::default()
    }

    /// Creates a new instance of the [`AnthropicBackendBuilder`]. This builder that allows you to specify the model and other parameters. It is converted to an `LlmClient` instance using the `init` method.
    pub fn anthropic() -> backend_builders::anthropic::AnthropicBackendBuilder {
        backend_builders::anthropic::AnthropicBackendBuilder::default()
    }

    /// Creates a new instance of the [`PerplexityBackendBuilder`]. This builder that allows you to specify the model and other parameters. It is converted to an `LlmClient` instance using the `init` method.
    pub fn perplexity() -> backend_builders::perplexity::PerplexityBackendBuilder {
        backend_builders::perplexity::PerplexityBackendBuilder::default()
    }

    pub fn embeddings(&self) -> embeddings::Embeddings {
        embeddings::Embeddings::new(self.backend.clone())
    }

    pub fn basic_completion(&self) -> basic_completion::BasicCompletion {
        basic_completion::BasicCompletion::new(self.backend.clone())
    }

    pub fn basic_primitive(&self) -> workflows::basic_primitive::BasicPrimitiveWorkflowBuilder {
        workflows::basic_primitive::BasicPrimitiveWorkflowBuilder::new(self.backend.clone())
    }

    pub fn reason(&self) -> workflows::reason::ReasonWorkflowBuilder {
        workflows::reason::ReasonWorkflowBuilder::new(self.backend.clone())
    }

    pub fn nlp(&self) -> workflows::nlp::Nlp {
        workflows::nlp::Nlp::new(self.backend.clone())
    }

    pub fn shutdown(&self) {
        self.backend.shutdown();
    }

    pub fn completion_request(&self) -> CompletionRequest {
        CompletionRequest::new(self.backend.clone())
    }

    pub fn embeddings_request(&self) -> EmbeddingsRequest {
        EmbeddingsRequest::new(self.backend.clone())
    }
}
