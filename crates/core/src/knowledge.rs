pub trait Knowledge {
    fn load(&self) -> Result<String, KnowledgeError>;
}

#[derive(Debug, thiserror::Error)]
#[error("Knowledge error")]
pub enum KnowledgeError {}
