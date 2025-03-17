use alith::{Embed, EmbedError, EmbeddingsBuilder, LLM, TextEmbedder};

#[derive(Debug)]
pub struct Foo {
    pub message: String,
}

impl Embed for Foo {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(&self.message);
        Ok(())
    }
}

impl Foo {
    pub fn new<S: AsRef<str>>(message: S) -> Self {
        Self {
            message: message.as_ref().to_string(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::from_model_name("gpt-4")?;
    let embeddings_model = model.embeddings_model("text-embedding-3-small");
    let data = EmbeddingsBuilder::new(embeddings_model.clone())
        .documents(vec![Foo::new("doc0"), Foo::new("doc1"), Foo::new("doc2")])
        .unwrap()
        .build()
        .await?;

    println!("{:?}", data);

    Ok(())
}
