use alith::HuggingFaceLoader;

fn main() -> Result<(), anyhow::Error> {
    let _path = HuggingFaceLoader::new().load_file("model.safetensors", "gpt2")?;
    Ok(())
}
