use crate::chat::{Completion, Document, Message, Request};
use crate::executor::Executor;
use crate::knowledge::Knowledge;
use crate::store::{Storage, VectorStoreError};
use crate::task::TaskError;
use crate::tool::Tool;
use futures::{stream, StreamExt, TryStreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

pub struct Agent<M: Completion> {
    /// The model to use.
    pub model: Arc<RwLock<M>>,
    /// Indexed storage for the agent.
    pub store_indices: Vec<(usize, Box<dyn Storage>)>,
    /// The tools to use.
    pub tools: Arc<Vec<Box<dyn Tool>>>,
    /// Knowledge sources for the agent.
    pub knowledges: Arc<Vec<Box<dyn Knowledge>>>,
    /// The unique ID of the agent.
    pub id: Uuid,
    /// The name of the agent.
    pub name: String,
    /// System prompt for the agent.
    pub preamble: String,
    /// System format for the agent.
    pub system_template: String,
    /// Prompt format for the agent.
    pub prompt_template: String,
    /// Response format for the agent.
    pub response_template: String,
    /// Enable or disable verbose mode.
    pub verbose: bool,
    /// The maximum requests per minute for completions.
    pub max_rpm: Option<usize>,
    /// Temperature of the model.
    pub temperature: Option<f32>,
    /// Maximum number of tokens for the completion.
    pub max_tokens: Option<usize>,
    /// Maximum execution time for the agent to complete a task.
    pub max_execution_time: Option<usize>,
    /// Whether to respect the context window.
    pub respect_context_window: bool,
    /// Whether code execution is allowed.
    pub allow_code_execution: bool,
}

impl<M: Completion> Agent<M>
where
    M: Completion,
{
    /// Creates a new agent.
    pub fn new<I>(name: impl ToString, model: M, tools: I) -> Agent<M>
    where
        I: IntoIterator<Item = Box<dyn Tool>>,
    {
        Agent {
            model: Arc::new(RwLock::new(model)),
            tools: Arc::new(tools.into_iter().collect()),
            store_indices: vec![],
            id: Uuid::new_v4(),
            name: name.to_string(),
            preamble: String::new(),
            system_template: String::new(),
            prompt_template: String::new(),
            response_template: String::new(),
            verbose: false,
            max_rpm: None,
            temperature: None,
            max_tokens: None,
            max_execution_time: None,
            knowledges: Arc::new(Vec::new()),
            respect_context_window: false,
            allow_code_execution: false,
        }
    }

    /// Adds a storage index to the agent.
    pub fn store_index(&mut self, sample: usize, store: impl Storage + 'static) -> &mut Self {
        self.store_indices.push((sample, Box::new(store)));
        self
    }

    /// Processes a prompt using the agent.
    pub async fn prompt(&mut self, prompt: &str) -> Result<String, TaskError> {
        self.chat(prompt, vec![]).await
    }

    /// Processes a prompt using the agent.
    pub async fn chat(&mut self, prompt: &str, history: Vec<Message>) -> Result<String, TaskError> {
        let mut executor = Executor::new(
            self.model.clone(),
            self.knowledges.clone(),
            self.tools.clone(),
        );
        let mut req = Request::new(prompt.to_string(), self.preamble.clone());
        req.history = history;
        req.max_tokens = self.max_tokens;
        req.temperature = self.temperature;
        req.tools = self
            .tools
            .iter()
            .map(|tool| tool.definition())
            .collect::<Vec<_>>();
        req.documents = stream::iter(self.store_indices.iter())
            .then(|(num_sample, storage)| async {
                Ok::<_, VectorStoreError>(
                    storage
                        .search(prompt, *num_sample, 1000.0)
                        .await?
                        .into_iter()
                        .map(|(id, text, _)| Document {
                            id,
                            text,
                            additional_props: HashMap::new(),
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .try_fold(vec![], |mut acc, docs| async {
                acc.extend(docs);
                Ok(acc)
            })
            .await
            .map_err(|err| TaskError::ExecutionError(err.to_string()))?;

        let response = executor
            .invoke(req)
            .await
            .map_err(|err| TaskError::ExecutionError(err.to_string()))?;

        Ok(response)
    }
}
