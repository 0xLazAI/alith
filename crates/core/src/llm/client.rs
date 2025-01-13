use std::sync::Arc;

use crate::chat::CallFunction;
use crate::chat::Completion;
use crate::chat::CompletionError;
use crate::chat::Request;
use crate::chat::ResponseContent;
use crate::chat::ResponseToolCalls;
use crate::chat::ToolCall;
use crate::embeddings::Embeddings as EmbeddingsTrait;
use crate::embeddings::EmbeddingsData;
use crate::embeddings::EmbeddingsError;
use anyhow::Result;

use async_trait::async_trait;
pub use llm_client::basic_completion::BasicCompletion;
pub use llm_client::embeddings::Embeddings;
pub use llm_client::interface::requests::completion::{CompletionRequest, CompletionResponse};
pub use llm_client::models::api_model::ApiLlmModel;
pub use llm_client::prelude::*;
pub use llm_client::LlmClient;

impl ResponseContent for CompletionResponse {
    fn content(&self) -> String {
        self.content.to_string()
    }
}

pub struct Client {
    pub(crate) client: LlmClient,
}

impl Clone for Client {
    fn clone(&self) -> Self {
        Self {
            client: LlmClient::new(Arc::clone(&self.client.backend)),
        }
    }
}

impl Client {
    pub fn from_model_name(model: &str) -> Result<Client> {
        if model.starts_with("gpt") {
            let mut builder = LlmClient::openai();
            builder.model = ApiLlmModel::openai_model_from_model_id(model);
            let client = builder.init()?;
            Ok(Client { client })
        } else if model.starts_with("claude") {
            let mut builder = LlmClient::anthropic();
            builder.model = ApiLlmModel::anthropic_model_from_model_id(model);
            let client = builder.init()?;
            Ok(Client { client })
        } else {
            Err(anyhow::anyhow!("unknown model {model}"))
        }
    }

    pub fn openai_compatible_client(api_key: &str, base_url: &str, model: &str) -> Result<Client> {
        let mut builder = LlmClient::openai();
        builder.model = ApiLlmModel::gpt_4();
        builder.model.model_base.model_id = model.to_string();
        builder.config.api_config.api_key = Some(api_key.to_string().into());
        builder.config.api_config.host = base_url.to_string();
        let client = builder.init()?;
        Ok(Client { client })
    }
}

impl ResponseToolCalls for CompletionResponse {
    fn toolcalls(&self) -> Vec<ToolCall> {
        self.tool_calls
            .as_ref()
            .unwrap_or(&Vec::new())
            .iter()
            .map(|call| ToolCall {
                id: call.id.clone(),
                r#type: call.r#type.clone(),
                function: CallFunction {
                    name: call.function.name.clone(),
                    arguments: call.function.arguments.clone(),
                },
            })
            .collect()
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        self.client.shutdown();
    }
}

impl Completion for Client {
    type Response = CompletionResponse;

    async fn completion(&mut self, request: Request) -> Result<Self::Response, CompletionError> {
        let mut completion = self.client.basic_completion();
        if let Some(temperature) = request.temperature {
            completion.temperature(temperature);
        }
        if let Some(max_tokens) = request.max_tokens {
            completion.max_tokens(max_tokens.try_into().unwrap());
        }

        let prompt = completion.prompt();
        // Add preamble if provided
        if !request.preamble.trim().is_empty() {
            prompt
                .add_system_message()
                .map_err(|err| CompletionError::Normal(err.to_string()))?
                .set_content(&request.preamble);
        }

        // Add knowledge sources if provided
        for knowledge in &request.knowledges {
            prompt
                .add_system_message()
                .map_err(|err| CompletionError::Normal(err.to_string()))?
                .set_content(knowledge);
        }

        // Add user prompt with or without context
        if request.documents.is_empty() {
            prompt
                .add_user_message()
                .map_err(|err| CompletionError::Normal(err.to_string()))?
                .set_content(&request.prompt);
        } else {
            prompt
                .add_user_message()
                .map_err(|err| CompletionError::Normal(err.to_string()))?
                .set_content(request.prompt_with_context());
        }

        // Add conversation history
        for msg in &request.history {
            let result = match msg.role.as_str() {
                "system" => prompt.add_system_message(),
                "user" => prompt.add_user_message(),
                "assistant" => prompt.add_assistant_message(),
                _ => continue, // Just skip unknown roles
            };
            result
                .map_err(|err| CompletionError::Normal(err.to_string()))?
                .set_content(&msg.content);
        }
        completion.base_req.tools.append(&mut request.tools.clone());

        // Execute the completion request
        completion
            .run()
            .await
            .map_err(|err| CompletionError::Normal(err.to_string()))
    }
}

#[async_trait]
impl EmbeddingsTrait for Client {
    const MAX_DOCUMENTS: usize = 1024;

    async fn embed_texts(
        &self,
        input: Vec<String>,
    ) -> Result<Vec<EmbeddingsData>, EmbeddingsError> {
        let mut embeddings = self.client.embeddings();
        embeddings.set_input(input);
        embeddings
            .run()
            .await
            .map(|resp| {
                resp.data
                    .iter()
                    .map(|data| EmbeddingsData {
                        document: data.object.clone(),
                        vec: data.embedding.clone(),
                    })
                    .collect()
            })
            .map_err(|err| EmbeddingsError::ResponseError(err.to_string()))
    }
}
