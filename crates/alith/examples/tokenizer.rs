use alith::Tokenizer;

fn main() -> Result<(), anyhow::Error> {
    let _tokenizer = Tokenizer::new_tiktoken("gpt4")?;
    let _tokenizer = Tokenizer::new_from_hf_repo(None, "gpt-2")?;
    let _tokenizer = Tokenizer::new_from_tokenizer_json("tokenizer.json")?;
    let _vector = _tokenizer.tokenize("Hello world");
    Ok(())
}
