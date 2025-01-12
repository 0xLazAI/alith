use alith::{Agent, EmbeddingsBuilder, Storage, StoreFactory, LLM};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::from_model_name("gpt4o")?;
    let embeddings = EmbeddingsBuilder::new(model.embeddings_model("text-embedding-ada-002"))
        .documents(vec!["doc0", "doc1", "doc2"])
        .unwrap()
        .build()
        .await?;
    let storage = StoreFactory::get_store("memory");
    for (_, embedding) in embeddings {
        for data in embedding {
            storage.save(data.document);
        }
    }

    let mut agent = Agent::new("simple agent", model, vec![]);
    agent.preamble =
        "
            You are a dictionary assistant here to assist the user in understanding the meaning of words.
            You will find additional non-standard word definitions that could be useful below.
        ".to_string();
    agent.store_index(1, storage);
    let response = agent.prompt("What does \"glarb-glarb\" mean?").await?;

    println!("{}", response);

    Ok(())
}
