pub use llm_client::utils::splitting::{
    split_text_into_indices, Separator, SeparatorGroup, TextSplit, TextSplitter,
};

#[inline]
pub fn split_text(text: &str) -> Vec<String> {
    match TextSplitter::new().split_text(text) {
        Some(splits) => splits
            .iter()
            .map(|split| split.text().to_string())
            .collect(),
        None => vec![],
    }
}
