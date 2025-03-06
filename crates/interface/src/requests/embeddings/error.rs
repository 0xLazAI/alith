#[derive(Debug, thiserror::Error)]
pub enum EmbeddingsError {
    // Break on these types
    #[error("RequestBuilderError: {0}")]
    RequestBuilderError(String),
    #[error("ClientError: {0}")]
    ClientError(#[from] crate::llms::api::error::ClientError),
    #[error("LocalClientError: {0}")]
    LocalClientError(String),
    #[error("ExceededRetryCount")]
    ExceededRetryCount {
        message: String,
        errors: Vec<EmbeddingsError>,
    },
    // Continue on these types
    #[error("ResponseContentEmpty: Response had no content")]
    ResponseContentEmpty,
    #[error("StopLimitRetry: stopped_limit == true && retry_stopped_limit == true")]
    StopLimitRetry,
    /// Json serialization or deserialization errors
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),
}
