use alith_core::store::DocumentId;
pub use chromadb::client::{
    ChromaAuthMethod, ChromaClient, ChromaClientOptions, ChromaTokenHeader,
};
pub use chromadb::collection::{
    ChromaCollection, CollectionEntries, GetOptions, GetResult, QueryOptions,
};
use std::sync::Arc;

use alith_core::{
    embeddings::{Embeddings, EmbeddingsData},
    store::{Storage, TopNResults, VectorStoreError},
};
use async_trait::async_trait;

pub const DEFAULT_CHROMADB_COLLECTION_NAME: &str = "alith";
pub const DEFAULT_CHROMADB_DIM: i64 = 768;
pub const DEFAULT_CHROMADB_URL: &str = "http://localhost:8000";

/// Chroma storage implementation.
pub struct ChromaStorage<E: Embeddings> {
    _client: ChromaClient,
    embeddings: Arc<E>,
    collection: ChromaCollection,
}

impl<E: Embeddings> ChromaStorage<E> {
    /// Creates a new instance of `ChromaStorage`.
    pub async fn from_documents(
        client: ChromaClient,
        embeddings: E,
        documents: Vec<EmbeddingsData>,
    ) -> Result<Self, VectorStoreError> {
        let collection = client
            .get_or_create_collection(DEFAULT_CHROMADB_COLLECTION_NAME, None)
            .await
            .map_err(|err| VectorStoreError::CustomError(err.to_string()))?;

        let ids: Vec<String> = documents.iter().map(|d| hash_id(&d.document)).collect();
        let ids = ids.iter().map(|i| i.as_str()).collect();

        let collection_entries = CollectionEntries {
            ids,
            embeddings: Some(documents.iter().map(|d| d.f32_vec()).collect()),
            metadatas: None,
            documents: Some(documents.iter().map(|d| d.document.as_str()).collect()),
        };

        collection
            .upsert(collection_entries, None)
            .await
            .map_err(|err| VectorStoreError::CustomError(err.to_string()))?;

        Ok(Self {
            _client: client,
            collection,
            embeddings: Arc::new(embeddings),
        })
    }

    /// Creates a new instance of `ChromaStorage`.
    pub async fn from_multiple_documents<T>(
        client: ChromaClient,
        embeddings: E,
        documents: Vec<(T, Vec<EmbeddingsData>)>,
    ) -> Result<Self, VectorStoreError> {
        let documents = documents.iter().flat_map(|d| d.1.clone()).collect();
        Self::from_documents(client, embeddings, documents).await
    }

    /// Generate the embed vector for the Chroma store.
    pub async fn generate_embed_vector(&self, value: &str) -> Result<Vec<f32>, VectorStoreError> {
        let vec = self
            .embeddings
            .embed_texts(vec![value.to_string()])
            .await?
            .first()
            .map(|e| e.vec.clone())
            .unwrap_or_default();
        Ok(vec.iter().map(|&x| x as f32).collect())
    }
}

#[async_trait]
impl<E: Embeddings> Storage for ChromaStorage<E> {
    async fn save(&self, value: String) -> Result<(), VectorStoreError> {
        let embeddings = self.generate_embed_vector(&value).await?;
        let id = hash_id(&value);
        let collection_entries = CollectionEntries {
            ids: vec![id.as_str()],
            embeddings: Some(vec![embeddings]),
            metadatas: None,
            documents: Some(vec![value.as_str()]),
        };

        self.collection
            .upsert(collection_entries, None)
            .await
            .map_err(|err| VectorStoreError::CustomError(err.to_string()))?;
        Ok(())
    }

    async fn search(&self, query: &str, limit: usize, _threshold: f32) -> TopNResults {
        let query_vectors = self.generate_embed_vector(query).await?;
        let result = self
            .collection
            .query(
                QueryOptions {
                    query_embeddings: Some(vec![query_vectors]),
                    n_results: Some(limit),
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|err| VectorStoreError::CustomError(err.to_string()))?;
        if let Some(ids) = result.ids.first() {
            let docs: Vec<String> = result
                .documents
                .unwrap_or_default()
                .first()
                .cloned()
                .unwrap_or_default();
            let distances: Vec<f32> = result
                .distances
                .unwrap_or_default()
                .first()
                .cloned()
                .unwrap_or_default();
            let result = ids
                .iter()
                .zip(docs)
                .zip(distances)
                .map(|((id, doc), distance)| (DocumentId(id.clone()), doc, distance))
                .collect();
            Ok(result)
        } else {
            Ok(vec![])
        }
    }

    async fn reset(&self) -> Result<(), VectorStoreError> {
        self.collection
            .delete(None, None, None)
            .await
            .map_err(|err| VectorStoreError::CustomError(err.to_string()))?;
        Ok(())
    }
}

fn hash_id<S: AsRef<str>>(input: S) -> String {
    use blake3::Hasher;
    let mut hasher = Hasher::new();
    hasher.update(input.as_ref().as_bytes());
    let hash = hasher.finalize();
    hash.to_string()
}
