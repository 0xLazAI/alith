use alith_core::{
    embeddings::{Embeddings, EmbeddingsData},
    store::{DocumentId, Storage, TopNResults, VectorStoreError},
};
use async_trait::async_trait;
pub use qdrant_client::{
    Qdrant as QdrantClient, QdrantBuilder, QdrantError,
    qdrant::{
        CreateCollectionBuilder, DeletePointsBuilder, Distance, PointId, PointStruct, Query,
        QueryPoints, QueryPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
        point_id::PointIdOptions,
    },
};
use serde_json::Map;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

pub const DEFAULT_COLLECTION_NAME: &str = "alith";
static ID: AtomicUsize = AtomicUsize::new(0);

/// In-memory storage implementation.
pub struct QdrantStorage<E: Embeddings> {
    client: QdrantClient,
    embeddings: Arc<E>,
}

impl<E: Embeddings> QdrantStorage<E> {
    /// Creates a new instance of `QdrantStorage`.
    pub async fn from_documents(
        client: QdrantClient,
        embeddings: E,
        documents: Vec<EmbeddingsData>,
    ) -> Result<Self, VectorStoreError> {
        let points = Self::documents_to_points(documents);

        client
            .upsert_points(UpsertPointsBuilder::new(DEFAULT_COLLECTION_NAME, points))
            .await
            .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?;

        Ok(Self {
            client,
            embeddings: Arc::new(embeddings),
        })
    }

    /// Creates a new instance of `QdrantStorage`.
    pub async fn from_multiple_documents<T>(
        client: QdrantClient,
        embeddings: E,
        documents: Vec<(T, Vec<EmbeddingsData>)>,
    ) -> Result<Self, VectorStoreError> {
        let documents = documents.iter().flat_map(|d| d.1.clone()).collect();
        Self::from_documents(client, embeddings, documents).await
    }

    /// Generate the query vector for the qdrant store.
    pub async fn generate_query_vector(&self, query: &str) -> Result<Vec<f32>, VectorStoreError> {
        let vec = self
            .embeddings
            .embed_texts(vec![query.to_string()])
            .await?
            .first()
            .map(|e| e.vec.clone())
            .unwrap_or_default();
        Ok(vec.iter().map(|&x| x as f32).collect())
    }

    /// Convert documents to points
    pub fn documents_to_points(
        documents: impl IntoIterator<Item = EmbeddingsData>,
    ) -> Vec<PointStruct> {
        documents
            .into_iter()
            .map(|data| {
                let vec: Vec<f32> = data.vec.iter().map(|&x| x as f32).collect();
                ID.fetch_add(1, Ordering::SeqCst);
                let mut object = Map::new();
                object.insert("document".to_string(), data.document.into());
                PointStruct::new(ID.load(Ordering::SeqCst) as u64, vec, object)
            })
            .collect()
    }
}

#[async_trait]
impl<E: Embeddings> Storage for QdrantStorage<E> {
    async fn save(&self, value: String) -> Result<(), VectorStoreError> {
        let embeddings = self
            .embeddings
            .embed_texts(vec![value])
            .await
            .map_err(VectorStoreError::EmbeddingError)?;
        let points = Self::documents_to_points(embeddings);

        self.client
            .upsert_points(UpsertPointsBuilder::new(DEFAULT_COLLECTION_NAME, points))
            .await
            .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?;

        Ok(())
    }

    async fn search(&self, query: &str, limit: usize, threshold: f32) -> TopNResults {
        let query = Query::new_nearest(self.generate_query_vector(query).await?);
        let params = QueryPointsBuilder::new(DEFAULT_COLLECTION_NAME)
            .score_threshold(threshold)
            .with_payload(true)
            .query(query)
            .limit(limit as u64);

        let points = self
            .client
            .query(params)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?
            .result;

        let result = points
            .into_iter()
            .flat_map(|point| {
                let id_usize =
                    id_usize(point.id.ok_or_else(|| {
                        VectorStoreError::DatastoreError("Missing point ID".into())
                    })?)?;
                let document = point
                    .payload
                    .get("document")
                    .map(|v| v.as_str().cloned().unwrap_or_default())
                    .unwrap_or_default();
                Ok::<(DocumentId, std::string::String, f32), VectorStoreError>((
                    DocumentId(id_usize),
                    document,
                    point.score,
                ))
            })
            .collect();
        Ok(result)
    }

    async fn reset(&self) -> Result<(), VectorStoreError> {
        self.client
            .delete_points(DeletePointsBuilder::new(DEFAULT_COLLECTION_NAME))
            .await
            .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?;

        Ok(())
    }
}

fn id_usize(point_id: PointId) -> Result<usize, VectorStoreError> {
    match point_id.point_id_options {
        Some(PointIdOptions::Num(num)) => Ok(num as usize),
        _ => Err(VectorStoreError::DatastoreError(
            "Invalid point ID format".into(),
        )),
    }
}
