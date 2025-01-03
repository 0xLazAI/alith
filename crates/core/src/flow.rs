use crate::{chat::Completion, task::Task};

pub struct Flow<M: Completion> {
    pub name: Option<String>,
    pub tasks: Vec<Task<M>>,
}
