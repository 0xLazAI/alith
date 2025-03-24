use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;

use crate::chat::CallFunction;
use crate::chat::Completion;
use crate::chat::CompletionError;
use crate::chat::Request;
use crate::chat::ResponseContent;
use crate::chat::ResponseTokenUsage;
use crate::chat::ResponseToolCalls;
use crate::chat::ToolCall;
use crate::embeddings::EmbeddingsData;
use crate::embeddings::EmbeddingsError;
use alith_interface::requests::completion::TokenUsage;
use anyhow::Result;

pub use alith_client as client;
pub use alith_client::LLMClient;
pub use alith_client::basic_completion::BasicCompletion;
pub use alith_client::embeddings::Embeddings;
pub use alith_client::prelude::*;
pub use alith_interface::requests::completion::{CompletionRequest, CompletionResponse};
pub use alith_models::api_model::ApiLLMModel;

pub struct Client {
    pub(crate) client: LLMClient,
}

impl Deref for Client {
    type Target = LLMClient;

    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

impl DerefMut for Client {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.client
    }
}

impl Clone for Client {
    fn clone(&self) -> Self {
        Self {
            client: LLMClient::new(Arc::clone(&self.client.backend)),
        }
    }
}

impl Client {
    pub fn from_model_name(model: &str) -> Result<Client> {
        if model.starts_with("gpt") {
            let mut builder = LLMClient::openai();
            builder.model = ApiLLMModel::openai_model_from_model_id(model);
            let client = builder.init()?;
            Ok(Client { client })
        } else if model.starts_with("claude") {
            let mut builder = LLMClient::anthropic();
            builder.model = ApiLLMModel::anthropic_model_from_model_id(model);
            let client = builder.init()?;
            Ok(Client { client })
        } else if model.starts_with("llama") || model.starts_with("sonar") {
            let mut builder = LLMClient::perplexity();
            builder.model = ApiLLMModel::perplexity_model_from_model_id(model);
            let client = builder.init()?;
            Ok(Client { client })
        } else {
            Err(anyhow::anyhow!("unknown model {model}"))
        }
    }

    pub fn openai_compatible_client(api_key: &str, base_url: &str, model: &str) -> Result<Client> {
        let mut builder = LLMClient::openai();
        builder.model = ApiLLMModel::gpt_4();
        builder.model.model_base.model_id = model.to_string();
        builder.config.api_config.api_key = Some(api_key.to_string().into());
        builder.config.api_config.host = base_url.to_string();
        builder.config.logging_config.logger_name = "generic".to_string();
        let client = builder.init()?;
        Ok(Client { client })
    }
}

impl ResponseContent for CompletionResponse {
    fn content(&self) -> String {
        self.content.to_string()
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

impl ResponseTokenUsage for CompletionResponse {
    fn token_usage(&self) -> TokenUsage {
        self.token_usage.clone()
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
        // New the complation request
        let mut completion = self.client.basic_completion();
        if let Some(temperature) = request.temperature {
            completion.temperature(temperature);
        }
        if let Some(max_tokens) = request.max_tokens {
            completion.max_tokens(max_tokens.try_into().unwrap());
        }
        if let Some(top_p) = request.top_p {
            completion.top_p(top_p);
        }
        // Construct the prompt
        let prompt = completion.prompt();
        // Add preamble if provided
        if !request.preamble.trim().is_empty() {
            prompt
                .add_system_message()
                .map_err(|err| CompletionError::Normal(err.to_string()))?
                .set_content(&request.preamble);
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
        prompt
            .add_user_message()
            .map_err(|err| CompletionError::Normal(err.to_string()))?
            .set_content(request.effective_prompt().as_str());
        // Add custom tools
        completion.base_req.tools.append(&mut request.tools.clone());
        // Execute the completion request
        completion
            .run()
            .await
            .map_err(|err| CompletionError::Normal(err.to_string()))
    }
}

impl Client {
    pub async fn embed_texts(
        &self,
        model: &str,
        input: Vec<String>,
    ) -> Result<Vec<EmbeddingsData>, EmbeddingsError> {
        let mut embeddings = self.client.embeddings();
        embeddings.set_input(input.clone());
        embeddings.set_model(model.to_string());
        embeddings
            .run()
            .await
            .map(|resp| {
                resp.data
                    .iter()
                    .zip(input)
                    .map(|(data, document)| EmbeddingsData {
                        document,
                        vec: data.embedding.clone(),
                    })
                    .collect()
            })
            .map_err(|err| EmbeddingsError::ResponseError(err.to_string()))
    }
}
