pub mod extract;

use alith_interface::{llms::LLMBackend, requests::completion::CompletionRequest};
use extract::Extract;
use std::sync::Arc;

pub struct Nlp {
    pub base_req: CompletionRequest,
}

impl Nlp {
    pub fn new(backend: Arc<LLMBackend>) -> Self {
        Self {
            base_req: CompletionRequest::new(backend),
        }
    }

    pub fn extract(self) -> Extract {
        Extract::new(self.base_req)
    }
}
