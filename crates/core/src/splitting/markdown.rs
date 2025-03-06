use super::SplitError;
use text_splitter::{ChunkConfig, MarkdownSplitter};

/// Generate a list of chunks from a given markdown text.
#[inline]
pub fn split_markdown(
    text: &str,
    size: usize,
    overlap: usize,
    trim: bool,
) -> Result<Vec<String>, SplitError> {
    let chunk_config = ChunkConfig::new(size)
        .with_overlap(overlap)?
        .with_trim(trim);
    Ok(MarkdownSplitter::new(chunk_config)
        .chunks(text)
        .map(|x| x.to_string())
        .collect())
}
