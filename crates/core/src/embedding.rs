use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::cmp::max;
use futures::stream;
use futures::stream::StreamExt;
use futures::stream::TryStreamExt;

// Struct representing an embedding
#[derive(Clone, Default, Deserialize, Serialize, Debug)]
pub struct Embedding {
    pub document: String,
    pub vec: Vec<f64>,
}

// Trait for an embedding model
pub trait EmbeddingModel: Clone + Send + Sync {
    const MAX_DOCUMENTS: usize;

    /// Generate embeddings for a list of texts
    fn embed_texts(
        &self,
        texts: Vec<String>,
    ) -> futures::future::BoxFuture<'static, Result<Vec<Embedding>, EmbeddingError>>;
}

// Trait that defines the embedding process for a document
pub trait Embed {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError>;
}

// A simple struct to hold text data for embedding
#[derive(Default)]
pub struct TextEmbedder {
    pub texts: Vec<String>,
}

// Errors related to embedding
#[derive(Debug)]
pub enum EmbedError {
    Custom(String),
}

// Errors related to the embedding model
#[derive(Debug)]
pub enum EmbeddingError {
    Custom(String),
}

// Enum to handle one or multiple embeddings
#[derive(Clone)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

impl<T: Clone> OneOrMany<T> {
    /// Create an instance with a single value
    pub fn one(value: T) -> Self {
        OneOrMany::One(value)
    }

    /// Push a new value into the structure
    pub fn push(&mut self, value: T) {
        match self {
            OneOrMany::One(existing) => {
                *self = OneOrMany::Many(vec![existing.clone(), value]);
            }
            OneOrMany::Many(existing) => existing.push(value),
        }
    }
}

// The main builder struct for generating embeddings
pub struct EmbeddingsBuilder<M: EmbeddingModel, T: Embed> {
    model: M,
    documents: Vec<(T, Vec<String>)>,
}

impl<M: EmbeddingModel, T: Embed> EmbeddingsBuilder<M, T> {
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

impl<M: EmbeddingModel, T: Embed + Send> EmbeddingsBuilder<M, T> {
    /// Generate embeddings for all documents
    pub async fn build(self) -> Result<Vec<(T, OneOrMany<Embedding>)>, EmbeddingError> {
        // Create lookup stores for documents and their corresponding texts
        let mut docs = HashMap::new();
        let mut texts = HashMap::new();

        for (i, (doc, doc_texts)) in self.documents.into_iter().enumerate() {
            docs.insert(i, doc);
            texts.insert(i, doc_texts);
        }

        // Compute embeddings for the texts
        let mut embeddings = stream::iter(texts.into_iter())
            .flat_map(|(i, texts)| {
                stream::iter(texts.into_iter().map(move |text| (i, text)))
            })
            .chunks(M::MAX_DOCUMENTS)
            .map(|chunk| async {
                let (ids, docs): (Vec<_>, Vec<_>) = chunk.into_iter().unzip();

                let embeddings = self.model.embed_texts(docs).await?;
                Ok::<_, EmbeddingError>(
                    ids.into_iter().zip(embeddings).collect::<Vec<_>>(),
                )
            })
            .buffer_unordered(max(1, 1024 / M::MAX_DOCUMENTS))
            .try_fold(
                HashMap::new(),
                |mut acc: HashMap<_, OneOrMany<Embedding>>, embeddings| async move {
                    embeddings.into_iter().for_each(|(i, embedding)| {
                        acc.entry(i)
                            .and_modify(|embeds| embeds.push(embedding.clone()))
                            .or_insert(OneOrMany::one(embedding));
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