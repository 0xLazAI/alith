use crate::chat::Completion;
use crate::chat::CompletionError;
use crate::chat::Request;
use crate::chat::ResponseContent;
use anyhow::Result;

pub use llm_client::basic_completion::BasicCompletion;
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
    pub(crate) completion: BasicCompletion,
}

impl Client {
    pub fn from_model_name(name: &str) -> Result<Client> {
        if name.starts_with("gpt") {
            let mut builder = LlmClient::openai();
            builder.model = ApiLlmModel::openai_model_from_model_id(name);
            let client = builder.init()?;
            let completion = client.basic_completion();
            Ok(Client { client, completion })
        } else if name.starts_with("claude") {
            let mut builder = LlmClient::anthropic();
            builder.model = ApiLlmModel::anthropic_model_from_model_id(name);
            let client = builder.init()?;
            let completion = client.basic_completion();
            Ok(Client { client, completion })
        } else {
            Err(anyhow::anyhow!("unknown model {name}"))
        }
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
        self.completion
            .prompt()
            .add_user_message()
            .map_err(|err| CompletionError::Normal(err.to_string()))?
            .set_content(&request.prompt);
        self.completion
            .run()
            .await
            .map_err(|err| CompletionError::Normal(err.to_string()))
    }
}
