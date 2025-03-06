use alith_interface::{llms::LLMBackend, requests::completion::CompletionRequest};
use std::sync::Arc;
use subject_of_text::ClassifySubjectOfText;

pub mod hierarchical_classification;
pub mod subject_of_text;

pub struct Classify {
    backend: Arc<LLMBackend>,
}

impl Classify {
    pub fn new(backend: Arc<LLMBackend>) -> Self {
        Self { backend }
    }

    pub fn subject_of_text<T: AsRef<str>>(self, content: T) -> ClassifySubjectOfText {
        ClassifySubjectOfText::new(CompletionRequest::new(self.backend), content)
    }
}
