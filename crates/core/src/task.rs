use uuid::Uuid;

use crate::{agent::Agent, chat::Completion};

pub struct Task<M: Completion> {
    pub id: Uuid,
    pub description: String,
    pub output: String,
    pub agent: Agent<M>,
}
