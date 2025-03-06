use alith_interface::{
    llms::LLMBackend,
    requests::embeddings::{EmbeddingsRequest, EmbeddingsResponse},
};
use std::sync::Arc;

#[derive(Clone)]
pub struct Embeddings {
    pub req: EmbeddingsRequest,
}

impl Embeddings {
    pub fn new(backend: Arc<LLMBackend>) -> Self {
        Self {
            req: EmbeddingsRequest::new(backend),
        }
    }

    pub async fn run(&mut self) -> crate::Result<EmbeddingsResponse> {
        let res = self.req.request().await?;

        Ok(res)
    }

    #[inline]
    pub fn set_input(&mut self, input: Vec<String>) {
        self.req.input = input;
    }

    #[inline]
    pub fn set_model(&mut self, model: String) {
        self.req.model = model;
    }
}
