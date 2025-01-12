use crate::embeddings::EmbeddingsError;
use futures::future::BoxFuture;
use serde_json::Value;
use std::sync::{Arc, Mutex};

#[derive(Debug, thiserror::Error)]
pub enum VectorStoreError {
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingsError),
    /// JSON error (e.g.: serialization, deserialization, etc.)
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Datastore error: {0}")]
    DatastoreError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("Missing Id: {0}")]
    MissingIdError(String),
}

pub type TopNResults = Result<Vec<(f64, String, Value)>, VectorStoreError>;

/// Trait representing a storage backend.
pub trait Storage: Send + Sync {
    /// Saves a value into the storage.
    fn save(&self, value: String);
    /// Searches the storage with a query, limiting the results and applying a threshold.
    fn search(&self, query: &str, limit: usize, threshold: f32) -> BoxFuture<'static, TopNResults>;
    /// Resets the storage by clearing all stored data.
    fn reset(&self);
}

/// In-memory storage implementation.
#[derive(Debug, Clone)]
pub struct InMemoryStorage {
    data: Arc<Mutex<Vec<String>>>, // Simple in-memory vector to store data.
}

impl InMemoryStorage {
    /// Creates a new instance of `InMemoryStorage`.
    pub fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl Storage for InMemoryStorage {
    fn save(&self, value: String) {
        let mut data = self.data.lock().unwrap();
        data.push(value);
    }

    fn search(
        &self,
        query: &str,
        _limit: usize,
        _threshold: f32,
    ) -> BoxFuture<'static, TopNResults> {
        let _data = self.data.lock().unwrap().clone(); // Clone data for safe async use.
        let _query = query.to_string(); // Clone the query string for async move.

        Box::pin(async move {
            // TODO: use hnsw and bm25 to search
            let results = vec![];
            Ok(results)
        })
    }

    fn reset(&self) {
        let mut data = self.data.lock().unwrap();
        data.clear();
    }
}

/// Factory to create different types of vector stores.
pub struct StoreFactory;

impl StoreFactory {
    /// Returns a store instance based on the specified type.
    pub fn get_store(store_type: &str) -> impl Storage {
        match store_type {
            "memory" => InMemoryStorage::new(),
            _ => panic!("Unknown store type: {}", store_type),
        }
    }
}
