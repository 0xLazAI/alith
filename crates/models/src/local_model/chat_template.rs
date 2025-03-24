use std::collections::HashMap;

use super::metadata::tokenizer::TokenizerMetadata;
use anyhow::Context;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize, Clone, PartialEq)]
pub struct LLMChatTemplate {
    pub chat_template: String,
    pub bos_token: Option<String>,
    pub eos_token: String,
    pub unk_token: Option<String>,
    pub base_generation_prefix: Option<String>,
}

impl LLMChatTemplate {
    pub fn from_local_path(tokenizer_config_local_path: &PathBuf) -> crate::Result<Self> {
        let file = std::fs::File::open(tokenizer_config_local_path)?;
        let reader = std::io::BufReader::new(file);
        let mut chat_template: LLMChatTemplate = serde_json::from_reader(reader)?;
        chat_template.set_generation_prefix()?;
        Ok(chat_template)
    }

    pub fn from_gguf_tokenizer(tokenizer: &TokenizerMetadata) -> crate::Result<Self> {
        let chat_template = if let Some(chat_template) = &tokenizer.chat_template {
            chat_template
        } else {
            anyhow::bail!("chat_template not found.");
        };
        let ggml = if let Some(ggml) = &tokenizer.ggml {
            ggml
        } else {
            anyhow::bail!("GGML tokenizer model not found.");
        };

        let bos_token = ggml
            .tokens
            .get(ggml.bos_token_id as usize)
            .map(ToString::to_string)
            .with_context(|| format!("Token not found for ID: {}", ggml.bos_token_id))?;

        let eos_token = ggml
            .tokens
            .get(ggml.eos_token_id as usize)
            .map(ToString::to_string)
            .with_context(|| format!("Token not found for ID: {}", ggml.eos_token_id))?;

        let unk_token = if let Some(unk_token_id) = ggml.unknown_token_id {
            Some(
                ggml.tokens
                    .get(unk_token_id as usize)
                    .map(ToString::to_string)
                    .with_context(|| format!("Token not found for ID: {}", unk_token_id))?,
            )
        } else {
            None
        };

        let mut chat_template = LLMChatTemplate {
            chat_template: chat_template.to_owned(),
            bos_token: Some(bos_token),
            eos_token,
            unk_token,
            base_generation_prefix: None,
        };
        chat_template.set_generation_prefix()?;
        Ok(chat_template)
    }

    /// Applies a chat template to a message, given a message and a chat template.
    #[inline]
    pub fn apply(
        &self,
        messages: &[HashMap<String, String>],
        add_generation_prompt: bool,
    ) -> String {
        alith_prompt::apply_chat_template(
            messages,
            &self.chat_template,
            self.bos_token.as_deref(),
            &self.eos_token,
            self.unk_token.as_deref(),
            add_generation_prompt,
        )
    }

    fn set_generation_prefix(&mut self) -> crate::Result<()> {
        let user_message_1 = HashMap::from([
            ("role".to_string(), "user".to_string()),
            ("content".to_string(), "test_user_message_1".to_string()),
        ]);
        let assistant_message_1 = HashMap::from([
            ("role".to_string(), "assistant".to_string()),
            (
                "content".to_string(),
                "test_assistant_message_1".to_string(),
            ),
        ]);

        let message_1 = alith_prompt::apply_chat_template(
            &[user_message_1.clone()],
            &self.chat_template,
            self.bos_token.as_deref(),
            &self.eos_token,
            self.unk_token.as_deref(),
            true,
        );
        let message_1 = message_1
            .trim_end_matches(self.eos_token.as_str())
            .to_owned();
        let message_2 = alith_prompt::apply_chat_template(
            &[user_message_1, assistant_message_1],
            &self.chat_template,
            self.bos_token.as_deref(),
            &self.eos_token,
            self.unk_token.as_deref(),
            true,
        );

        // Find the point where the outputs start to differ
        let diff_index = message_1
            .chars()
            .zip(message_2.chars())
            .position(|(a, b)| a != b)
            .unwrap_or(message_1.len());

        // Extract the differing part
        let diff_part = &message_2[diff_index..];

        // Find the start of the assistant content
        if let Some(content_index) = diff_part.find("test_assistant_message_1") {
            // The prefix is everything before the content
            self.base_generation_prefix = Some(
                diff_part[..content_index]
                    .trim_start_matches(self.eos_token.as_str())
                    .to_string(),
            );
        } else {
            crate::bail!("Error finding base_generation_prefix");
        }
        Ok(())
    }
}

impl std::fmt::Debug for LLMChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("LLMChatTemplate");
        debug_struct.field("chat_template", &"string too long to print nicely");
        debug_struct.field("bos_token", &self.bos_token);
        debug_struct.field("eos_token", &self.eos_token);
        debug_struct.field("unk_token", &self.unk_token);
        debug_struct.finish()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use super::LLMChatTemplate;

    #[test]
    fn test_apply_chat_template() {
        let path = PathBuf::new()
            .join("src")
            .join("local_model")
            .join("gguf")
            .join("preset")
            .join("llama")
            .join("llama3_1_8b_instruct")
            .join("tokenizer_config.json");
        let template = LLMChatTemplate::from_local_path(&path).unwrap();
        let messages = &[
            HashMap::from([
                ("role".to_string(), "system".to_string()),
                ("content".to_string(), "test_system_message".to_string()),
            ]),
            HashMap::from([
                ("role".to_string(), "user".to_string()),
                ("content".to_string(), "test_user_message_1".to_string()),
            ]),
            HashMap::from([
                ("role".to_string(), "assistant".to_string()),
                ("content".to_string(), "test_user_message_2".to_string()),
            ]),
        ];
        let result = template.apply(messages, false);
        assert_eq!(
            result,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\ntest_system_message<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ntest_user_message_1<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\ntest_user_message_2<|eot_id|>"
        );
    }
}
