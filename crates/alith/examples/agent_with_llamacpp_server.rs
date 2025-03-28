#[cfg(not(target_os = "windows"))]
use alith::inference::{LlamaEngine, serve::run};

#[cfg(not(target_os = "windows"))]
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    run(
        None,
        LlamaEngine::new("/root/models/qwen2.5-1.5b-instruct-q5_k_m.gguf").await?,
    )
    .await?;
    // Run the server and run the following command to test the server
    /*
    curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "your-model-name",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is the capital of France?"}
      ],
      "temperature": 0.7,
      "max_tokens": 100
    }'
    */
    // ```
    Ok(())
}

#[cfg(target_os = "windows")]
fn main() {}
