pub trait Knowledge: Send + Sync {
    fn load(&self) -> Result<String, KnowledgeError>;

    fn name(&self) -> &str {
        "default-knowledge"
    }

    fn description(&self) -> &str {
        "A default knowledge source"
    }

    fn enrich(&self, input: &str) -> String {
        format!("Enriched with {}: {}", self.name(), input)
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Knowledge error")]
pub enum KnowledgeError {
    #[error("Failed to load the knowledge source")]
    LoadError,

    #[error("An unknown error occurred: {0}")]
    Unknown(String),
}

pub struct DummyKnowledge;

impl Knowledge for DummyKnowledge {
    fn load(&self) -> Result<String, KnowledgeError> {
        Ok("Dummy knowledge loaded successfully".to_string())
    }

    fn name(&self) -> &str {
        "dummy-knowledge"
    }

    fn description(&self) -> &str {
        "A dummy knowledge source for testing"
    }
}
