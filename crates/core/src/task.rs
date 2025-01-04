use crate::{agent::Agent, chat::Completion};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

#[derive(Clone)]
pub struct Task<M: Completion> {
    pub id: Uuid,
    pub description: String,
    pub output: Option<String>,
    pub agent: Arc<RwLock<Agent<M>>>, // Use RwLock for thread-safe mutable access
    pub metadata: Option<TaskMetadata>,
}

#[derive(Debug, Clone)]
pub struct TaskMetadata {
    pub priority: usize,
    pub tags: Vec<String>,
    pub created_at: u64, // Timestamp in seconds since the UNIX epoch
}

impl<M: Completion> Task<M> {
    /// Creates a new task.
    pub fn new(agent: Arc<RwLock<Agent<M>>>, description: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            description,
            output: None,
            agent,
            metadata: None,
        }
    }

    /// Attaches metadata to the task.
    pub fn with_metadata(mut self, metadata: TaskMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Executes the task using the agent.
    pub async fn execute(&mut self) -> Result<String, TaskError> {
        // Get a write lock for mutable access to the Agent
        let mut agent = self.agent.write().map_err(|_| TaskError::LockError)?;

        // Call `execute_task` on the Agent
        let result = agent
            .execute_task(self.clone(), None, None)
            .await
            .map_err(|_| TaskError::ExecutionError)?;

        // Set the output of the task
        self.output = Some(result.clone());
        Ok(result)
    }
}

impl TaskMetadata {
    /// Creates a new `TaskMetadata` instance.
    pub fn new(priority: usize, tags: Vec<String>) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(); // Get current time as seconds since UNIX epoch

        Self {
            priority,
            tags,
            created_at,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TaskError {
    #[error("Failed to execute the task")]
    ExecutionError,

    #[error("Failed to acquire lock on the agent")]
    LockError,

    #[error("An unknown error occurred: {0}")]
    Unknown(String),
}
