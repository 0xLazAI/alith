use metadata::LocalLLMMetadata;

pub mod chat_template;
pub mod gguf;
pub mod hf_loader;
pub mod metadata;

use super::LLMModelBase;
pub use chat_template::LLMChatTemplate;
pub use gguf::{GgufLoader, GgufLoaderTrait, preset::GgufPresetTrait};
pub use hf_loader::HfTokenTrait;
use std::path::PathBuf;

pub struct LocalLLMModel {
    pub model_base: LLMModelBase,
    pub local_model_path: PathBuf,
    pub model_metadata: LocalLLMMetadata,
    pub chat_template: LLMChatTemplate,
}

impl Default for LocalLLMModel {
    fn default() -> Self {
        let mut loader = GgufLoader::default();
        loader.load().expect("Failed to load LLMPreset")
    }
}

impl std::fmt::Debug for LocalLLMModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("LocalLLMModel");
        debug_struct.field("model_id", &self.model_base.model_id);
        debug_struct.field("local_model_path", &self.local_model_path);
        debug_struct.field("model_metadata", &self.model_metadata);
        debug_struct.field("chat_template", &self.chat_template);
        debug_struct.finish()
    }
}
