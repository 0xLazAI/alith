#[cfg(feature = "inference")]
use alith::{EmbeddingsBuilder, FastEmbeddingsModel};

#[cfg(feature = "inference")]
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let embeddingds_model = FastEmbeddingsModel::try_default().unwrap();
    let data = EmbeddingsBuilder::new(embeddingds_model.clone())
        .documents(vec!["doc0", "doc1", "doc2"])
        .unwrap()
        .build()
        .await?;
    println!("{:?}", data);
    Ok(())
}

#[cfg(not(feature = "inference"))]
fn main() {
    println!("This example need to set the inference feature.")
}
