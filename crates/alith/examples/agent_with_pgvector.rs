use alith::store::pgvector::{PgPoolOptions, PgVectorStorage};
use alith::{Agent, Chat, EmbeddingsBuilder, LLM};
use sqlx::{migrate, query};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::from_model_name("gpt-4")?;
    let embeddings_model = model.embeddings_model("text-embedding-3-small");
    let data = EmbeddingsBuilder::new(embeddings_model.clone())
        .documents(vec!["doc0", "doc1", "doc2"])
        .unwrap()
        .build()
        .await?;

    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL not set");
    let pool = PgPoolOptions::new()
        .max_connections(50)
        .idle_timeout(std::time::Duration::from_secs(5))
        .connect(&database_url)
        .await
        .expect("Failed to create postgres pool");
    // Make sure database is setup
    migrate!("examples/pgvector_migarations").run(&pool).await?;
    // Delete documents from table to have a clean start (optional, not recommended for production)
    query("TRUNCATE alith").execute(&pool).await?;
    let storage = PgVectorStorage::from_multiple_documents(pool, embeddings_model, data).await?;

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
