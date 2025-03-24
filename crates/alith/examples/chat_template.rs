use alith::{GgufLoader, GgufLoaderTrait};
use std::collections::HashMap;

fn main() -> Result<(), anyhow::Error> {
    let model = GgufLoader::default()
        .local_quant_file_path("/root/models/qwen2.5-1.5b-instruct-q5_k_m.gguf")
        .load()?;
    let messages = &[
        HashMap::from([
            ("role".to_string(), "system".to_string()),
            ("content".to_string(), "test_system_message".to_string()),
        ]),
        HashMap::from([
            ("role".to_string(), "user".to_string()),
            ("content".to_string(), "test_user_message_1".to_string()),
        ]),
    ];
    println!("{}", model.chat_template.apply(messages, true));
    Ok(())
}
