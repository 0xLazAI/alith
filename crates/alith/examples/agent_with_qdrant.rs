use alith::store::qdrant::{
    CreateCollectionBuilder, DEFAULT_COLLECTION_NAME, Distance, QdrantClient, QdrantStorage,
    VectorParamsBuilder,
};
use alith::{Agent, Chat, EmbeddingsBuilder, LLM};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::from_model_name("gpt-4")?;
    let embeddings_model = model.embeddings_model("text-embedding-3-small");
    let data = EmbeddingsBuilder::new(embeddings_model.clone())
        .documents(vec!["doc0", "doc1", "doc2"])
        .unwrap()
        .build()
        .await?;

    let client = QdrantClient::from_url("http://localhost:6334").build()?;

    if !client.collection_exists(DEFAULT_COLLECTION_NAME).await? {
        client
            .create_collection(
                CreateCollectionBuilder::new(DEFAULT_COLLECTION_NAME)
                    .vectors_config(VectorParamsBuilder::new(1536, Distance::Cosine)),
            )
            .await?;
    }

    let storage = QdrantStorage::from_multiple_documents(client, embeddings_model, data).await?;

    let agent = Agent::new("simple agent", model)
        .preamble(
            r#"
You are a dictionary assistant here to assist the user in understanding the meaning of words.
You will find additional non-standard word definitions that could be useful below.
"#,
        )
        .store_index(1, storage);
    let response = agent.prompt("What does \"glarb-glarb\" mean?").await?;

    println!("{}", response);

    Ok(())
}
