use std::{fmt::Display, sync::Arc};

use alith_core::{
    embeddings::{Embeddings, EmbeddingsData},
    store::{DocumentId, Storage, TopNResult, TopNResults, VectorStoreError},
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use sqlx::{self, PgPool, migrate, postgres::PgPoolOptions};
pub use uuid::Uuid;

pub const DEFAULT_PG_VECTOR_TABLE_NAME: &str = "alith";

/// Postgres vector storage implementation.
pub struct PgVectorStorage<E: Embeddings> {
    table: String,
    pg_pool: PgPool,
    distance_function: PgVectorDistanceFunction,
    embeddings: Arc<E>,
}

/// PgVector supported distance functions
/// <+> - L1 distance.
/// <-> - L2 distance
/// <#> - (negative) inner product
/// <=> - cosine distance.
/// <~> - Hamming distance
/// <%> - Jaccard distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum PgVectorDistanceFunction {
    L1,
    L2,
    InnerProduct,
    #[default]
    Cosine,
    Hamming,
    Jaccard,
}

impl Display for PgVectorDistanceFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PgVectorDistanceFunction::L1 => write!(f, "<+>"),
            PgVectorDistanceFunction::L2 => write!(f, "<->"),
            PgVectorDistanceFunction::InnerProduct => write!(f, "<#>"),
            PgVectorDistanceFunction::Cosine => write!(f, "<=>"),
            PgVectorDistanceFunction::Hamming => write!(f, "<~>"),
            PgVectorDistanceFunction::Jaccard => write!(f, "<%>"),
        }
    }
}

#[derive(Debug, Deserialize, sqlx::FromRow)]
pub struct SearchResult {
    id: Uuid,
    document: Value,
    distance: f64,
}

impl SearchResult {
    pub fn into_result(self) -> TopNResult {
        let document: String =
            serde_json::from_value(self.document).map_err(VectorStoreError::JsonError)?;
        Ok((
            DocumentId(self.id.to_string()),
            document,
            self.distance as f32,
        ))
    }
}

impl<E: Embeddings> PgVectorStorage<E> {
    /// Creates a new instance of `PgVectorStorage`.
    pub async fn from_documents(
        pg_pool: PgPool,
        embeddings: E,
        documents: Vec<EmbeddingsData>,
    ) -> Result<Self, VectorStoreError> {
        let store = Self {
            pg_pool,
            distance_function: Default::default(),
            table: DEFAULT_PG_VECTOR_TABLE_NAME.to_string(),
            embeddings: Arc::new(embeddings),
        };
        store.insert_documents(documents).await?;
        Ok(store)
    }

    /// Creates a new instance of `PgVectorStorage`.
    pub async fn from_multiple_documents<T>(
        pg_pool: PgPool,
        embeddings: E,
        documents: Vec<(T, Vec<EmbeddingsData>)>,
    ) -> Result<Self, VectorStoreError> {
        let documents = documents.iter().flat_map(|d| d.1.clone()).collect();
        Self::from_documents(pg_pool, embeddings, documents).await
    }

    /// Set the table name of the storage.
    pub fn with_table(self, table: String) -> Result<Self, VectorStoreError> {
        Ok(Self {
            pg_pool: self.pg_pool,
            distance_function: self.distance_function,
            table,
            embeddings: self.embeddings,
        })
    }

    /// Set the distance function of the storage.
    pub fn with_distance_function(
        self,
        distance_function: PgVectorDistanceFunction,
    ) -> Result<Self, VectorStoreError> {
        Ok(Self {
            pg_pool: self.pg_pool,
            distance_function,
            table: self.table,
            embeddings: self.embeddings,
        })
    }

    /// Generate the query vector for the pg vector store.
    pub async fn generate_query_vector(
        &self,
        query: &str,
    ) -> Result<pgvector::Vector, VectorStoreError> {
        let vec = self
            .embeddings
            .embed_texts(vec![query.to_string()])
            .await?
            .first()
            .map(|e| e.vec.clone())
            .unwrap_or_default();
        Ok(vec.iter().map(|&x| x as f32).collect::<Vec<f32>>().into())
    }

    /// Insert documents into the storage
    pub async fn insert_documents(
        &self,
        documents: Vec<EmbeddingsData>,
    ) -> Result<(), VectorStoreError> {
        for doc in documents {
            let id = Uuid::new_v4();
            let json_document = serde_json::to_value(&doc.document).unwrap();
            let embedding_text = doc.document;
            let embedding: Vec<f64> = doc.vec;
            sqlx::query(
                    format!(
                        "INSERT INTO {} (id, document, embedded_text, embedding) VALUES ($1, $2, $3, $4)",
                        self.table
                    )
                    .as_str(),
                )
                .bind(id)
                .bind(&json_document)
                .bind(&embedding_text)
                .bind(&embedding)
                .execute(&self.pg_pool)
                .await
                .map_err(|e| VectorStoreError::DatastoreError(e.into()))?;
        }

        Ok(())
    }

    fn search_query(&self, with_document: bool) -> String {
        let document = if with_document { ", document" } else { "" };
        format!(
            "
            SELECT id{}, distance FROM ( \
              SELECT DISTINCT ON (id) id{}, embedding {} $1 as distance \
              FROM {} \
              ORDER BY id, distance \
            ) as d \
            ORDER BY distance \
            LIMIT $2",
            document, document, self.distance_function, self.table
        )
    }

    fn reset_query(&self) -> String {
        format!("TRUNCATE {}", self.table)
    }
}

#[async_trait]
impl<E: Embeddings> Storage for PgVectorStorage<E> {
    async fn save(&self, value: String) -> Result<(), VectorStoreError> {
        let embeddings = self
            .embeddings
            .embed_texts(vec![value])
            .await
            .map_err(VectorStoreError::EmbeddingError)?;
        self.insert_documents(embeddings).await?;
        Ok(())
    }

    async fn search(&self, query: &str, limit: usize, _threshold: f32) -> TopNResults {
        let embedded_query = self.generate_query_vector(query).await?;

        let rows: Vec<SearchResult> = sqlx::query_as(self.search_query(true).as_str())
            .bind(embedded_query)
            .bind(limit as i64)
            .fetch_all(&self.pg_pool)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        Ok(rows
            .into_iter()
            .flat_map(SearchResult::into_result)
            .collect())
    }

    async fn reset(&self) -> Result<(), VectorStoreError> {
        sqlx::query(self.reset_query().as_str())
            .execute(&self.pg_pool)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
        Ok(())
    }
}
