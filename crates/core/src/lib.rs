pub mod agent;
pub mod chat;
pub mod chunking;
pub mod embeddings;
pub mod executor;
pub mod extractor;
pub mod flow;
pub mod knowledge;
pub mod llm;
pub mod mcp;
pub mod splitting;
pub use llm_client;
pub mod memory;
pub mod store;
pub mod task;
pub mod tool;

use std::sync::Arc;
use tokio::sync::RwLock;

pub type Ref<T> = Arc<RwLock<T>>;

#[inline]
pub fn make_ref<T>(t: T) -> Ref<T> {
    Arc::new(RwLock::new(t))
}
