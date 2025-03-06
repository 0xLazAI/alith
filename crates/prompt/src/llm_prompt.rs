use std::{
    cell::{Ref, RefMut},
    collections::HashMap,
};

use crate::{
    chat_template_prompt::ChatTemplatePrompt, openai_prompt::OpenAIPrompt, PromptMessage,
    PromptMessageType, PromptTokenizer, TextConcatenator, TextConcatenatorTrait,
};
use std::sync::Arc;

#[derive(Clone)]
pub enum LLMPrompt {
    ChatTemplatePrompt(ChatTemplatePrompt),
    OpenAIPrompt(OpenAIPrompt),
}

impl LLMPrompt {
    pub fn new_chat_template_prompt(
        chat_template: &str,
        bos_token: Option<&str>,
        eos_token: &str,
        unk_token: Option<&str>,
        base_generation_prefix: Option<&str>,
        tokenizer: Arc<dyn PromptTokenizer>,
    ) -> Self {
        LLMPrompt::ChatTemplatePrompt(ChatTemplatePrompt::new(
            chat_template,
            bos_token,
            eos_token,
            unk_token,
            base_generation_prefix,
            tokenizer,
        ))
    }

    pub fn new_openai_prompt(
        tokens_per_message: Option<u32>,
        tokens_per_name: Option<i32>,
        tokenizer: Arc<dyn PromptTokenizer>,
    ) -> Self {
        LLMPrompt::OpenAIPrompt(OpenAIPrompt::new(
            tokens_per_message,
            tokens_per_name,
            tokenizer,
        ))
    }

    // Setter functions
    pub fn add_system_message(&self) -> crate::Result<Ref<PromptMessage>> {
        if !self.messages_ref().is_empty() {
            crate::bail!("System message must be first message.");
        };
        let message = PromptMessage::new(PromptMessageType::System, self.concatenator_ref());
        self.messages_mut().push(message);
        self.clear_built_prompt();
        Ok(Ref::map(self.messages_ref(), |msgs| msgs.last().unwrap()))
    }

    pub fn add_user_message(&self) -> crate::Result<Ref<PromptMessage>> {
        if !self.messages_ref().is_empty()
            && self.messages_ref().last().unwrap().message_type == PromptMessageType::User
        {
            crate::bail!("Cannot add user message when previous message is user message.");
        }
        let message = PromptMessage::new(PromptMessageType::User, self.concatenator_ref());
        self.messages_mut().push(message);
        self.clear_built_prompt();
        Ok(Ref::map(self.messages_ref(), |msgs| msgs.last().unwrap()))
    }

    pub fn add_assistant_message(&self) -> crate::Result<Ref<PromptMessage>> {
        if self.messages_ref().is_empty() {
            crate::bail!("Cannot add assistant message as first message.");
        } else if self.messages_ref().last().unwrap().message_type == PromptMessageType::Assistant {
            crate::bail!(
                "Cannot add assistant message when previous message is assistant message."
            );
        };
        let message = PromptMessage::new(PromptMessageType::Assistant, self.concatenator_ref());
        self.messages_mut().push(message);
        self.clear_built_prompt();
        Ok(Ref::map(self.messages_ref(), |msgs| msgs.last().unwrap()))
    }

    pub fn set_generation_prefix<T: AsRef<str>>(&self, generation_prefix: T) {
        self.clear_built_prompt();
        match self {
            LLMPrompt::ChatTemplatePrompt(p) => p.set_generation_prefix(generation_prefix),
            LLMPrompt::OpenAIPrompt(_) => {}
        }
    }

    pub fn clear_generation_prefix(&self) {
        match self {
            LLMPrompt::ChatTemplatePrompt(p) => p.clear_generation_prefix(),
            LLMPrompt::OpenAIPrompt(_) => {}
        }
    }

    pub fn reset_prompt(&self) {
        self.messages_mut().clear();
        self.clear_built_prompt();
    }

    pub fn clear_built_prompt(&self) {
        match self {
            LLMPrompt::ChatTemplatePrompt(p) => p.clear_built_prompt(),
            LLMPrompt::OpenAIPrompt(p) => p.clear_built_prompt(),
        }
    }

    // Getter functions
    pub fn get_total_prompt_tokens(&self) -> crate::Result<u64> {
        if self.total_prompt_tokens().is_none() {
            self.build_prompt()?;
        };
        if let Some(ref total_prompt_tokens) = *self.total_prompt_tokens() {
            Ok(*total_prompt_tokens)
        } else {
            crate::bail!("total_prompt_tokens is None after building!");
        }
    }

    pub fn get_built_prompt_string(&self) -> crate::Result<String> {
        if self.built_prompt_string().is_none() {
            self.build_prompt()?;
        }
        if let Some(ref built_prompt_string) = *self.built_prompt_string() {
            Ok(built_prompt_string.clone())
        } else {
            crate::bail!("built_prompt_string is None after building!");
        }
    }

    pub fn get_built_prompt_as_tokens(&self) -> crate::Result<Vec<u32>> {
        if self.built_prompt_as_tokens().is_none() {
            self.build_prompt()?;
        };
        if let Some(ref built_prompt_as_tokens) = *self.built_prompt_as_tokens() {
            Ok(built_prompt_as_tokens.clone())
        } else {
            crate::bail!("total_prompt_tokens is None after building!");
        }
    }

    pub fn get_built_prompt_hashmap(&self) -> crate::Result<Vec<HashMap<String, String>>> {
        if self.built_prompt_hashmap().is_none() {
            self.build_prompt()?;
        }
        if let Some(ref built_prompt_hashmap) = *self.built_prompt_hashmap() {
            Ok(built_prompt_hashmap.clone())
        } else {
            crate::bail!("built_prompt_hashmap is None after building!");
        }
    }

    // Builder functions
    fn build_prompt(&self) -> crate::Result<()> {
        self.precheck_build()?;
        match self {
            LLMPrompt::ChatTemplatePrompt(p) => {
                p.build_prompt();
            }
            LLMPrompt::OpenAIPrompt(p) => {
                p.build_prompt();
            }
        }
        Ok(())
    }

    fn precheck_build(&self) -> crate::Result<()> {
        if let Some(last) = self.messages_ref().last() {
            if last.message_type == PromptMessageType::Assistant {
                crate::bail!(
                    "Cannot build prompt when the current inference message is PromptMessageType::Assistant"
                )
            } else if last.message_type == PromptMessageType::System {
                crate::bail!("Cannot build prompt when the current inference message is PromptMessageType::System")
            } else {
                Ok(())
            }
        } else {
            crate::bail!("Cannot build prompt when there are no messages.")
        }
    }

    // Helper functions
    fn messages_ref(&self) -> Ref<Vec<PromptMessage>> {
        match self {
            LLMPrompt::ChatTemplatePrompt(p) => p.messages.borrow(),
            LLMPrompt::OpenAIPrompt(p) => p.messages.borrow(),
        }
    }

    fn messages_mut(&self) -> RefMut<Vec<PromptMessage>> {
        match self {
            LLMPrompt::ChatTemplatePrompt(p) => p.messages.borrow_mut(),
            LLMPrompt::OpenAIPrompt(p) => p.messages.borrow_mut(),
        }
    }

    fn concatenator_ref(&self) -> &TextConcatenator {
        match self {
            LLMPrompt::ChatTemplatePrompt(p) => &p.concatenator,
            LLMPrompt::OpenAIPrompt(p) => &p.concatenator,
        }
    }

    fn concatenator_mut(&mut self) -> &mut TextConcatenator {
        match self {
            LLMPrompt::ChatTemplatePrompt(p) => &mut p.concatenator,
            LLMPrompt::OpenAIPrompt(p) => &mut p.concatenator,
        }
    }

    fn built_prompt_string(&self) -> Ref<Option<String>> {
        match self {
            LLMPrompt::ChatTemplatePrompt(p) => p.built_prompt_string.borrow(),
            LLMPrompt::OpenAIPrompt(_) => unimplemented!(),
        }
    }

    fn built_prompt_hashmap(&self) -> Ref<Option<Vec<HashMap<String, String>>>> {
        match self {
            LLMPrompt::ChatTemplatePrompt(_) => unimplemented!(),
            LLMPrompt::OpenAIPrompt(p) => p.built_prompt_hashmap.borrow(),
        }
    }

    fn total_prompt_tokens(&self) -> Ref<Option<u64>> {
        match self {
            LLMPrompt::ChatTemplatePrompt(p) => p.total_prompt_tokens.borrow(),
            LLMPrompt::OpenAIPrompt(p) => p.total_prompt_tokens.borrow(),
        }
    }

    fn built_prompt_as_tokens(&self) -> Ref<Option<Vec<u32>>> {
        match self {
            LLMPrompt::ChatTemplatePrompt(p) => p.built_prompt_as_tokens.borrow(),
            LLMPrompt::OpenAIPrompt(_) => unimplemented!(),
        }
    }
}

impl TextConcatenatorTrait for LLMPrompt {
    fn concatenator_mut(&mut self) -> &mut TextConcatenator {
        self.concatenator_mut()
    }

    fn clear_built(&self) {
        self.clear_built_prompt();
    }
}

impl std::fmt::Display for LLMPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;

        match self {
            LLMPrompt::ChatTemplatePrompt(p) => {
                p.build_prompt();
            }
            LLMPrompt::OpenAIPrompt(p) => {
                p.build_prompt();
            }
        }

        match self {
            LLMPrompt::ChatTemplatePrompt(p) => write!(f, "{}", p),
            LLMPrompt::OpenAIPrompt(p) => write!(f, "{}", p),
        }
    }
}
