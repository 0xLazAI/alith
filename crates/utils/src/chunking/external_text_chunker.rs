use std::{
    collections::VecDeque,
    sync::{atomic::AtomicBool, Arc},
};

use alith_models::tokenizer::LlmTokenizer;
use text_splitter::ChunkConfig;
use text_splitter::TextSplitter as TextSplitterExternal;
use tiktoken_rs::cl100k_base;

use crate::splitting::Separator;

use super::{Chunk, ChunkerConfig, ChunkerResult, ABSOLUTE_LENGTH_MIN_DEFAULT_RATIO};

/// Chunk incoming text using the [text-splitter](https://github.com/benbrandt/text-splitter) crate.
/// This is a dev-dependency for comparing the performance of the text-splitter crate with the TextChunker.
/// In the future we may integrate the text-splitter crate's markdown and code chunking capabilities into the [`TextChunker`].
pub fn chunk_text_with_text_splitter(
    incoming_text: &str,
    max_chunk_token_size: u32,
    overlap_percent: Option<f32>,
) -> Option<ChunkerResult> {
    let tiktoken_tokenizer = cl100k_base().unwrap();
    let chunking_start_time = std::time::Instant::now();
    let config = if let Some(overlap_percent) = overlap_percent {
        let overlap = (max_chunk_token_size as f32 * overlap_percent).floor() as u32;
        if overlap >= max_chunk_token_size {
            eprintln!("Overlap is greater than or equal to max_chunk_token_size");
            return None;
        }
        let max_chunk_token_size = max_chunk_token_size - overlap;
        ChunkConfig::new(max_chunk_token_size as usize)
            .with_trim(true)
            .with_sizer(tiktoken_tokenizer)
            .with_overlap(overlap as usize)
            .unwrap()
    } else {
        ChunkConfig::new(max_chunk_token_size as usize)
            .with_trim(true)
            .with_sizer(tiktoken_tokenizer)
    };
    let splitter = TextSplitterExternal::new(config);
    let text_chunks = splitter
        .chunks(incoming_text)
        .map(|s| s.to_string())
        .collect::<Vec<String>>();

    let tokenizer = LlmTokenizer::new_tiktoken("gpt-4").unwrap();

    let dummy_config = Arc::new(ChunkerConfig {
        chunks_found: Arc::new(AtomicBool::new(true)),
        absolute_length_max: max_chunk_token_size,
        length_max: max_chunk_token_size as f32,
        absolute_length_min: (max_chunk_token_size as f32 * ABSOLUTE_LENGTH_MIN_DEFAULT_RATIO)
            as u32,
        overlap_percent,
        tokenizer: Arc::new(tokenizer),
        base_text: Arc::from(incoming_text),
        initial_separator: Separator::None,
        initial_splits: VecDeque::new(),
    });
    let mut chunks = Vec::new();
    for chunk in text_chunks.iter() {
        chunks.push(Chunk::dummy_chunk(&Arc::clone(&dummy_config), chunk));
    }
    Some(ChunkerResult::new(
        incoming_text,
        &dummy_config,
        chunking_start_time,
        chunks,
    ))
}
