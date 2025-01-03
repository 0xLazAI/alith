use std::error::Error;

pub trait Completion: Clone + Send + Sync {
    type Request;
    type Response: Send + Sync;
    type CompletionError: Error + Send + Sync;

    /// Generates a completion response for the given completion request.
    fn completion(
        &self,
        request: Self::Request,
    ) -> impl std::future::Future<Output = Result<Self::Response, Self::CompletionError>> + Send;
}

pub trait Prompt: Send + Sync {
    type PromptError: Error + Send + Sync;

    fn prompt(
        &self,
        prompt: &str,
    ) -> impl std::future::Future<Output = Result<String, Self::PromptError>> + Send;
}
