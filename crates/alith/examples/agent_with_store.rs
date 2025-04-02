use alith::{Agent, Chat, InMemoryStorage, LLM};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::from_model_name("gpt-4")?;
    let embeddings_model = model.embeddings_model("text-embedding-3-small");
    let storage = InMemoryStorage::from_multiple_documents::<()>(embeddings_model, vec![]);
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
