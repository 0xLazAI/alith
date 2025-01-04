use std::error::Error;

pub trait Completion: Clone + Send + Sync {
    type Request: From<String>;
    type Response: Send + Sync + ToString;

    /// Generates a completion response for the given completion request.
    fn completion(
        &self,
        request: Self::Request,
    ) -> impl std::future::Future<Output = Result<Self::Response, CompletionError>> + Send;
}

#[derive(Clone)]
pub struct StringCompletion;

impl Completion for StringCompletion {
    type Request = String;
    type Response = String;

    async fn completion(&self, request: Self::Request) -> Result<Self::Response, CompletionError> {
        Ok(format!("Processed: {}", request))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CompletionError {
    #[error("An normal completion error occurred: {0}")]
    Normal(String),
}

pub trait Prompt: Send + Sync {
    type PromptError: Error + Send + Sync;

    fn prompt(
        &self,
        prompt: &str,
    ) -> impl std::future::Future<Output = Result<String, Self::PromptError>> + Send;
}
