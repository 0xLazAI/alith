pub mod extract;

use alith_interface::{llms::LlmBackend, requests::completion::CompletionRequest};
use extract::Extract;

pub struct Nlp {
    pub base_req: CompletionRequest,
}

impl Nlp {
    pub fn new(backend: std::sync::Arc<LlmBackend>) -> Self {
        Self {
            base_req: CompletionRequest::new(backend),
        }
    }

    pub fn extract(self) -> Extract {
        Extract::new(self.base_req)
    }
}
