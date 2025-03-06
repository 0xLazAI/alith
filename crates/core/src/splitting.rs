pub use alith_utils::concatenator::TextConcatenator;
pub use alith_utils::extract::extract_urls;
pub use alith_utils::splitting::{
    split_text_into_indices, Separator, SeparatorGroup, TextSplit, TextSplitter,
};
pub use alith_utils::TextCleaner;

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
