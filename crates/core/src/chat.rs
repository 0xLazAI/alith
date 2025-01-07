use std::error::Error;

use llm_client::interface::requests::completion::ToolDefinition;

/// A trait representing a prompt-based interaction mechanism.
/// This trait defines the behavior of components that process user prompts
/// and return responses asynchronously.
///
/// # Associated Types
/// - `PromptError`: Represents errors that may occur during prompt processing.
///
/// # Requirements
/// Implementors of this trait must ensure thread safety (`Send` and `Sync`)
/// and provide an asynchronous implementation for the `prompt` method.
pub trait Prompt: Send + Sync {
    /// The error type associated with the `prompt` operation.
    type PromptError: Error + Send + Sync;

    /// Processes the given prompt and returns a response asynchronously.
    ///
    /// # Arguments
    /// - `prompt`: The input string provided by the user.
    ///
    /// # Returns
    /// A future that resolves to either:
    /// - `Ok(String)`: The generated response as a string.
    /// - `Err(Self::PromptError)`: An error that occurred during prompt processing.
    fn prompt(
        &self,
        prompt: &str,
    ) -> impl std::future::Future<Output = Result<String, Self::PromptError>> + Send;
}

/// Represents a request sent to a language model for completion.
///
/// A `Request` includes both the user-provided prompt and a system preamble.
/// Additionally, it allows customization of parameters such as the maximum token
/// limit and temperature for text generation.
#[derive(Debug, Clone)]
pub struct Request {
    /// The user-provided prompt to be completed by the language model.
    pub prompt: String,
    /// The system-defined preamble, which can be used to guide the model's behavior.
    pub preamble: String,
    /// Optional: The maximum number of tokens to use for the response.
    pub max_tokens: Option<usize>,
    /// Optional: The temperature value for text generation, controlling randomness.
    pub temperature: Option<f32>,
    /// The tools to be sent to the model
    pub tools: Vec<ToolDefinition>,
}

impl Request {
    /// Creates a new `Request` with the specified prompt and preamble.
    ///
    /// By default, the `max_tokens` and `temperature` fields are unset.
    ///
    /// # Arguments
    /// - `prompt`: The user-provided prompt.
    /// - `preamble`: The system-defined preamble to guide the model's behavior.
    ///
    /// # Returns
    /// A new instance of `Request`.
    pub fn new(prompt: String, preamble: String) -> Self {
        Self {
            prompt,
            preamble,
            max_tokens: None,
            temperature: None,
            tools: Vec::new(),
        }
    }
}

pub trait ResponseContent {
    fn content(&self) -> String;
}

/// A trait representing a completion engine for processing requests.
///
/// This trait defines the behavior of components that handle `Request` objects
/// and generate a response asynchronously. The response type is customizable
/// and must implement `Send`, `Sync`, and `ToString`.
///
/// # Associated Types
/// - `Response`: The type of the generated response.
pub trait Completion {
    /// The type of response returned by the `completion` method.
    type Response: Send + Sync + ResponseContent;

    /// Processes a completion request and returns the result asynchronously.
    ///
    /// # Arguments
    /// - `request`: The request object containing prompt and configuration details.
    ///
    /// # Returns
    /// A future that resolves to either:
    /// - `Ok(Self::Response)`: The generated response.
    /// - `Err(CompletionError)`: An error that occurred during completion.
    fn completion(
        &mut self,
        request: Request,
    ) -> impl std::future::Future<Output = Result<Self::Response, CompletionError>>;
}

/// An enumeration of possible errors that may occur during completion operations.
#[derive(Debug, thiserror::Error)]
pub enum CompletionError {
    /// A generic completion error.
    ///
    /// # Details
    /// - The error includes a message describing the cause of the failure.
    #[error("A normal completion error occurred: {0}")]
    Normal(String),
}
