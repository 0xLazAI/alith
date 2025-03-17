use alith::{Agent, EmbeddingsBuilder, InMemoryStorage, LLM, chunk_text};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::from_model_name("gpt-4")?;
    let embeddings_model = model.embeddings_model("text-embedding-3-small");
    let docs = chunk_text(include_str!("../../../README.md"), 200, None)?.unwrap_or_default();
    let data = EmbeddingsBuilder::new(embeddings_model.clone())
        .documents(docs)
        .unwrap()
        .build()
        .await?;
    // Or you can use other vertor storage, e.g., milvus, qdrant, etc. Here we use a memory store for example.
    let storage = InMemoryStorage::from_multiple_documents(embeddings_model, data);
    let agent = Agent::new("simple agent", model)
        .preamble(
            r#"
You are a dictionary assistant here to assist the user in understanding the meaning of words.
You will find additional non-standard word definitions that could be useful below.
"#,
        )
        .store_index(1, storage);
    let response = agent.prompt("What is Alith?").await?;

    println!("{}", response);

    Ok(())
}
