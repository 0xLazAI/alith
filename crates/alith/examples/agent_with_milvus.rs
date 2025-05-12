use alith::store::milvus::{MilvusClient, MilvusStorage};
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

    let client = MilvusClient::new("http://localhost:19530").await?;

    let storage = MilvusStorage::from_multiple_documents(client, embeddings_model, data).await?;

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
