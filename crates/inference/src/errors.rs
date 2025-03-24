use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("Engine error: {0}")]
    EngineError(String),
    #[error("Llamacpp error: {0}")]
    #[cfg(all(feature = "llamacpp", not(target_os = "windows")))]
    LlamaCpp(#[from] llama_cpp_2::LLamaCppError),
    #[error("MistralRs error: {0}")]
    #[cfg(feature = "mistralrs")]
    MistralRs(#[from] mistralrs::MistralRsError),
    #[error("Model load error: {0}")]
    ModelLoad(String),
    #[error("General error: {0}")]
    General(String),
    /// JSON error (e.g.: serialization, deserialization, etc.)
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}
