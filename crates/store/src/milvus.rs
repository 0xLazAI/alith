use alith_core::store::DocumentId;
pub use milvus::index::{IndexParams as MilvusIndexParams, IndexType as MilvusIndexType};
pub use milvus::options::LoadOptions as MilvusLoadOptions;
pub use milvus::query::QueryOptions as MilvusQueryOptions;
use milvus::query::SearchOptions;
pub use milvus::schema::{
    CollectionSchema as MilvusCollectionSchema,
    CollectionSchemaBuilder as MilvusCollectionSchemaBuilder,
};
pub use milvus::{
    self, client::Client as MilvusClient, collection::Collection as MilvusCollection,
    data::FieldColumn as MilvusFieldColumn, error::Error as MilvusError,
    schema::FieldSchema as MilvusFieldSchema, value::Value as MilvusValue,
};
use std::sync::Arc;

use alith_core::{
    embeddings::{Embeddings, EmbeddingsData},
    store::{Storage, TopNResults, VectorStoreError},
};
use async_trait::async_trait;

pub const DEFAULT_MILVUS_COLLECTION_SCHEMA_NAME: &str = "alith";
pub const DEFAULT_MILVUS_ID_FIELD: &str = "id";
pub const DEFAULT_MILVUS_VEC_FIELD: &str = "vector";
pub const DEFAULT_MILVUS_TEXT_FIELD: &str = "text";
pub const DEFAULT_MILVUS_DIM: i64 = 768;
pub const DEFAULT_MILVUS_URL: &str = "localhost:19530";

/// Milvus storage implementation.
pub struct MilvusStorage<E: Embeddings> {
    client: MilvusClient,
    embeddings: Arc<E>,
    collection: MilvusCollectionSchema,
}

impl<E: Embeddings> MilvusStorage<E> {
    /// Creates a new instance of `MilvusStorage`.
    pub async fn from_documents(
        client: MilvusClient,
        embeddings: E,
        documents: Vec<EmbeddingsData>,
    ) -> Result<Self, VectorStoreError> {
        let collection =
            MilvusCollectionSchemaBuilder::new(DEFAULT_MILVUS_COLLECTION_SCHEMA_NAME, "")
                .add_field(MilvusFieldSchema::new_primary_int64(
                    DEFAULT_MILVUS_ID_FIELD,
                    "primary key field",
                    true,
                ))
                .add_field(MilvusFieldSchema::new_float_vector(
                    DEFAULT_MILVUS_VEC_FIELD,
                    "embeddiings vector field",
                    DEFAULT_MILVUS_DIM,
                ))
                .add_field(MilvusFieldSchema::new_string(
                    DEFAULT_MILVUS_TEXT_FIELD,
                    "text field",
                ))
                .build()
                .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?;

        client
            .create_collection(collection.clone(), None)
            .await
            .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?;

        for document in &documents {
            let embed_column = MilvusFieldColumn::new(
                collection.get_field(DEFAULT_MILVUS_VEC_FIELD).unwrap(),
                document.f32_vec(),
            );
            let text_column = MilvusFieldColumn::new(
                collection.get_field(DEFAULT_MILVUS_TEXT_FIELD).unwrap(),
                vec![document.document.clone()],
            );
            client
                .insert(collection.name(), vec![embed_column, text_column], None)
                .await
                .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?;
        }

        Ok(Self {
            client,
            collection,
            embeddings: Arc::new(embeddings),
        })
    }

    /// Creates a new instance of `MilvusStorage`.
    pub async fn from_multiple_documents<T>(
        client: MilvusClient,
        embeddings: E,
        documents: Vec<(T, Vec<EmbeddingsData>)>,
    ) -> Result<Self, VectorStoreError> {
        let documents = documents.iter().flat_map(|d| d.1.clone()).collect();
        Self::from_documents(client, embeddings, documents).await
    }

    /// Generate the embed vector for the Milvus store.
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
impl<E: Embeddings> Storage for MilvusStorage<E> {
    async fn save(&self, value: String) -> Result<(), VectorStoreError> {
        let embeddings = self.generate_embed_vector(&value).await?;

        let embed_column = MilvusFieldColumn::new(
            self.collection.get_field(DEFAULT_MILVUS_VEC_FIELD).unwrap(),
            embeddings,
        );
        let text_column = MilvusFieldColumn::new(
            self.collection
                .get_field(DEFAULT_MILVUS_TEXT_FIELD)
                .unwrap(),
            vec![value],
        );
        self.client
            .insert(
                self.collection.name(),
                vec![embed_column, text_column],
                None,
            )
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
        self.client
            .flush(self.collection.name())
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
        Ok(())
    }

    async fn search(&self, query: &str, limit: usize, _threshold: f32) -> TopNResults {
        let query_vectors = self.generate_embed_vector(query).await?;
        let results = self
            .client
            .search(
                self.collection.name(),
                query_vectors
                    .iter()
                    .map(|v| MilvusValue::Float(*v))
                    .collect(),
                DEFAULT_MILVUS_VEC_FIELD,
                &SearchOptions::default()
                    .output_fields(vec![
                        DEFAULT_MILVUS_ID_FIELD.to_string(),
                        DEFAULT_MILVUS_TEXT_FIELD.to_string(),
                    ])
                    .limit(limit),
            )
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
        Ok(results
            .iter()
            .map(|r| {
                let id: Vec<i64> = r.field[0].clone().value.try_into().unwrap();
                let text: Vec<String> = r.field[1].clone().value.try_into().unwrap();
                (
                    DocumentId(id[0].clone().to_string()),
                    text[0].clone(),
                    r.score[0],
                )
            })
            .collect())
    }

    async fn reset(&self) -> Result<(), VectorStoreError> {
        self.client
            .drop_collection(self.collection.name())
            .await
            .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?;

        Ok(())
    }
}
