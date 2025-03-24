use alith::{GgufLoader, GgufLoaderTrait};

fn main() -> Result<(), anyhow::Error> {
    let _model = GgufLoader::default()
        .hf_quant_file_url("https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
        .load()?;

    // By default we attempt to extract everything we need from the GGUF file.
    // If you need to specifiy the tokenizer or chat template to use, you can add a hf repo to load from.
    let _model = GgufLoader::default()
        .hf_quant_file_url("https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
        .hf_config_repo_id("meta-llama/Meta-Llama-3-8B-Instruct")
        .load()?;

    // We can also load the model from the local path
    let _model = GgufLoader::default()
        .local_quant_file_path("/root/models/qwen2.5-1.5b-instruct-q5_k_m.gguf")
        .load()?;
    Ok(())
}
