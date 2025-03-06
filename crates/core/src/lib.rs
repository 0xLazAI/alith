pub mod agent;
pub mod chat;
pub mod chunking;
pub mod cleaner;
pub mod concatenator;
pub mod embeddings;
pub mod executor;
pub mod extractor;
pub mod flow;
pub mod json;
pub mod knowledge;
pub mod llm;
pub mod mcp;
pub mod memory;
pub mod parser;
pub mod splitting;
pub mod store;
pub mod task;
pub mod tool;

pub use alith_client as client;
pub use alith_interface as interface;
pub use alith_models as models;

use std::sync::Arc;
use tokio::sync::RwLock;

pub type Ref<T> = Arc<RwLock<T>>;

#[inline]
pub fn make_ref<T>(t: T) -> Ref<T> {
    Arc::new(RwLock::new(t))
}
