use async_trait::async_trait;
use futures::stream;
use futures::stream::StreamExt;
use futures::stream::TryStreamExt;
use serde::{Deserialize, Serialize};
use std::cmp::max;
use std::collections::HashMap;

/// Struct representing an embedding
#[derive(Clone, Default, Deserialize, Serialize, Debug)]
pub struct EmbeddingsData {
    pub document: String,
    pub vec: Vec<f64>,
}

/// Trait for embeddings
#[async_trait]
pub trait Embeddings: Clone + Send + Sync {
    const MAX_DOCUMENTS: usize = 1024;

    /// Generate embeddings for a list of texts
    async fn embed_texts(&self, input: Vec<String>)
        -> Result<Vec<EmbeddingsData>, EmbeddingsError>;
}

// Trait that defines the embedding process for a document
pub trait Embed {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError>;
}

impl Embed for String {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.clone());
        Ok(())
    }
}

impl Embed for &str {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

// A simple struct to hold text data for embedding
#[derive(Default)]
pub struct TextEmbedder {
    pub texts: Vec<String>,
}

impl TextEmbedder {
    /// Adds input `text` string to the list of texts in the [TextEmbedder] that need to be embedded.
    pub fn embed(&mut self, text: String) {
        self.texts.push(text);
    }
}

// Errors related to embedding
#[derive(Debug)]
pub enum EmbedError {
    Custom(String),
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingsError {
    /// Json error (e.g.: serialization, deserialization)
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),
    /// Error processing the document for embedding
    #[error("DocumentError: {0}")]
    DocumentError(Box<dyn std::error::Error + Send + Sync + 'static>),
    /// Error parsing the completion response
    #[error("ResponseError: {0}")]
    ResponseError(String),
    /// Error returned by the embedding model provider
    #[error("ProviderError: {0}")]
    ProviderError(String),
}

/// The main builder struct for generating embeddings
pub struct EmbeddingsBuilder<M: Embeddings, T: Embed> {
    model: M,
    documents: Vec<(T, Vec<String>)>,
}

impl<M: Embeddings, T: Embed> EmbeddingsBuilder<M, T> {
    /// Create a new embedding builder with the given model
    pub fn new(model: M) -> Self {
        Self {
            model,
            documents: vec![],
        }
    }

    /// Add a single document to the builder
    pub fn document(mut self, document: T) -> Result<Self, EmbedError> {
        let mut embedder = TextEmbedder::default();
        document.embed(&mut embedder)?;

        self.documents.push((document, embedder.texts));
        Ok(self)
    }

    /// Add multiple documents to the builder
    pub fn documents(self, documents: impl IntoIterator<Item = T>) -> Result<Self, EmbedError> {
        documents
            .into_iter()
            .try_fold(self, |builder, doc| builder.document(doc))
    }
}

impl<M: Embeddings, T: Embed + Send> EmbeddingsBuilder<M, T> {
    /// Generate embeddings for all documents
    pub async fn build(self) -> Result<Vec<(T, Vec<EmbeddingsData>)>, EmbeddingsError> {
        // Create lookup stores for documents and their corresponding texts
        let mut docs = HashMap::new();
        let mut texts = HashMap::new();

        for (i, (doc, doc_texts)) in self.documents.into_iter().enumerate() {
            docs.insert(i, doc);
            texts.insert(i, doc_texts);
        }

        // Compute embeddings for the texts
        let mut embeddings = stream::iter(texts.into_iter())
            .flat_map(|(i, texts)| stream::iter(texts.into_iter().map(move |text| (i, text))))
            .chunks(M::MAX_DOCUMENTS)
            .map(|chunk| async {
                let (ids, docs): (Vec<_>, Vec<_>) = chunk.into_iter().unzip();

                let embeddings = self.model.embed_texts(docs).await?;
                Ok::<_, EmbeddingsError>(ids.into_iter().zip(embeddings).collect::<Vec<_>>())
            })
            .buffer_unordered(max(1, 1024 / M::MAX_DOCUMENTS))
            .try_fold(
                HashMap::new(),
                |mut acc: HashMap<_, Vec<EmbeddingsData>>, embeddings| async move {
                    embeddings.into_iter().for_each(|(i, embedding)| {
                        acc.entry(i)
                            .and_modify(|embeds| embeds.push(embedding.clone()))
                            .or_insert(vec![embedding]);
                    });

                    Ok(acc)
                },
            )
            .await?;

        // Merge the embeddings back with their respective documents
        Ok(docs
            .into_iter()
            .map(|(i, doc)| {
                (
                    doc,
                    embeddings
                        .remove(&i)
                        .expect("Document embeddings should be present"),
                )
            })
            .collect())
    }
}
