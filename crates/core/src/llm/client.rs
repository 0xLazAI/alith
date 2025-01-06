use anyhow::Result;

pub use llm_client::basic_completion::BasicCompletion;
pub use llm_client::prelude::*;
pub use llm_client::LlmClient;
pub use llm_models::api_model::ApiLlmModel;

use crate::chat::Completion;
use crate::chat::CompletionError;

pub struct Client {
    pub(crate) client: LlmClient,
    pub(crate) completion: BasicCompletion,
}

impl Client {
    pub fn from(name: &str) -> Result<Client> {
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
    type Request = String;
    type Response = String;

    async fn completion(
        &mut self,
        request: Self::Request,
    ) -> Result<Self::Response, CompletionError> {
        self.completion
            .prompt()
            .add_user_message()
            .map_err(|err| CompletionError::Normal(err.to_string()))?
            .set_content(&request);
        self.completion
            .run()
            .await
            .map(|resp| resp.content)
            .map_err(|err| CompletionError::Normal(err.to_string()))
    }
}
