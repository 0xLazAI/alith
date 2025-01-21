use alith::{Agent, InMemoryStorage, LLM};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::from_model_name("gpt-4")?;
    let embeddingds_model = model.embeddings_model("text-embedding-ada-002");
    let storage = InMemoryStorage::from_multiple_documents::<()>(embeddingds_model, vec![]);
    let mut agent = Agent::new("simple agent", model, vec![]);
    agent.preamble = r#"
You are a dictionary assistant here to assist the user in understanding the meaning of words.
You will find additional non-standard word definitions that could be useful below.
"#
    .to_string();
    agent.store_index(1, storage);
    let response = agent.prompt("What does \"glarb-glarb\" mean?").await?;

    println!("{}", response);

    Ok(())
}
