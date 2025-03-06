pub mod markdown;
pub mod rule_based;

use crate::cleaner::TextCleaner;
use regex::Regex;
use std::{
    collections::VecDeque,
    ops::Range,
    sync::{Arc, LazyLock},
};
use text_splitter::ChunkConfigError;
use thiserror::Error;

pub use markdown::split_markdown;
pub use rule_based::split_text_into_indices;

#[derive(Error, Debug)]
pub enum SplitError {
    #[error("Chunk config error: {0}")]
    ChunkConfigError(#[from] ChunkConfigError),
}

#[derive(Default)]
pub struct TextSplitter {
    pub split_separator: Separator,
    pub recursive: bool,
    pub clean_text: bool,
}

impl TextSplitter {
    pub fn new() -> Self {
        Self {
            split_separator: Separator::TwoPlusEoL,
            recursive: true,
            clean_text: true,
        }
    }

    pub fn split_text(&self, text: &str) -> Option<VecDeque<TextSplit>> {
        let base_text: Arc<str> = if self.clean_text {
            Arc::from(self.split_separator.clean_text(text.as_ref()))
        } else {
            Arc::from(text)
        };

        let mut split_separator = self.split_separator.clone();
        let split_indices = if self.recursive {
            loop {
                let split_indices = split_separator.split_text_into_indices(&base_text);
                if split_indices.len() > 1 {
                    break split_indices;
                } else {
                    split_separator = split_separator.next()?;
                }
            }
        } else {
            split_separator.split_text_into_indices(&base_text)
        };
        if split_indices.len() < 2 {
            return None;
        }

        Some(
            split_indices
                .into_iter()
                .map(|indices| TextSplit::new(&indices, &split_separator, &base_text))
                .collect(),
        )
    }

    pub fn on_two_plus_newline(mut self) -> Self {
        self.split_separator = Separator::TwoPlusEoL;
        self
    }

    pub fn on_single_newline(mut self) -> Self {
        self.split_separator = Separator::SingleEol;
        self
    }

    pub fn on_sentences_rule_based(mut self) -> Self {
        self.split_separator = Separator::SentencesRuleBased;
        self
    }

    pub fn on_sentences_unicode(mut self) -> Self {
        self.split_separator = Separator::SentencesUnicode;
        self
    }

    pub fn on_words_unicode(mut self) -> Self {
        self.split_separator = Separator::WordsUnicode;
        self
    }

    pub fn on_graphemes_unicode(mut self) -> Self {
        self.split_separator = Separator::GraphemesUnicode;
        self
    }

    pub fn on_separator(mut self, split_separator: &Separator) -> Self {
        self.split_separator = split_separator.clone();
        self
    }

    pub fn recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    pub fn clean_text(mut self, clean_text: bool) -> Self {
        self.clean_text = clean_text;
        self
    }

    pub fn split_split(
        self,
        base_text: &Arc<str>,
        split_indices: &Range<usize>,
    ) -> Option<VecDeque<TextSplit>> {
        let start_offset = split_indices.start;
        let split_text = &base_text[split_indices.clone()];

        let mut split_separator = self.split_separator.clone();
        let split_indices = loop {
            let split_indices = split_separator.split_text_into_indices(split_text);
            if split_indices.len() > 1 {
                break split_indices;
            } else {
                split_separator = split_separator.next()?;
            }
        };
        Some(
            split_indices
                .into_iter()
                .map(|indices| {
                    let start = start_offset + indices.start;
                    let end = start_offset + indices.end;
                    TextSplit::new(&Range { start, end }, &split_separator, base_text)
                })
                .collect(),
        )
    }

    pub fn splits_to_text(splits: &VecDeque<TextSplit>, with_seperator: bool) -> String {
        let mut text = String::new();
        let mut last_separator = Separator::None;
        for (i, split) in splits.iter().enumerate() {
            if last_separator == Separator::GraphemesUnicode
                && split.split_separator != Separator::GraphemesUnicode
            {
                text.push(' ');
            };
            last_separator = split.split_separator.clone();
            match split.split_separator {
                Separator::TwoPlusEoL => {
                    text.push_str(split.text());
                    if with_seperator {
                        text.push_str("\n\n");
                    } else if i < splits.len() - 1 {
                        text.push(' ');
                    }
                }
                Separator::SingleEol => {
                    text.push_str(split.text());
                    if with_seperator {
                        text.push('\n');
                    } else if i < splits.len() - 1 {
                        text.push(' ');
                    }
                }
                Separator::SentencesRuleBased
                | Separator::SentencesUnicode
                | Separator::WordsUnicode => {
                    text.push_str(split.text());
                    if i < splits.len() - 1 {
                        text.push(' ');
                    }
                }
                Separator::GraphemesUnicode => {
                    text.push_str(split.text());
                }
                Separator::None => unreachable!(),
            }
        }
        text
    }
}

#[derive(Debug, Clone)]
pub struct TextSplit {
    pub indices: Range<usize>,
    pub split_separator: Separator,
    pub base_text: Arc<str>,
    pub token_count: Option<u32>,
}

impl TextSplit {
    fn new(indices: &Range<usize>, split_separator: &Separator, base_text: &Arc<str>) -> Self {
        Self {
            indices: indices.clone(),
            split_separator: split_separator.clone(),
            base_text: Arc::clone(base_text),

            token_count: None,
        }
    }

    pub fn char_count(&mut self) -> usize {
        self.text().chars().count()
    }

    pub fn text(&self) -> &str {
        &self.base_text[self.indices.clone()]
    }

    pub fn split(&self) -> Option<VecDeque<TextSplit>> {
        TextSplitter::default()
            .on_separator(&self.split_separator.next()?)
            .split_split(&self.base_text, &self.indices)
    }
}

#[derive(PartialEq)]
pub enum SeparatorGroup {
    Semantic,
    Syntactic,
}
impl SeparatorGroup {
    pub fn get(&self) -> Vec<Separator> {
        match self {
            Self::Semantic => vec![
                Separator::TwoPlusEoL,
                Separator::SingleEol,
                Separator::SentencesRuleBased,
                Separator::SentencesUnicode,
            ],
            Self::Syntactic => vec![Separator::WordsUnicode, Separator::GraphemesUnicode],
        }
    }
}

#[derive(PartialEq, Debug, Clone, Default)]
pub enum Separator {
    #[default]
    TwoPlusEoL,
    SingleEol,
    SentencesRuleBased,
    SentencesUnicode,
    WordsUnicode,
    GraphemesUnicode,
    None,
}

impl Separator {
    pub fn get_all() -> Vec<Self> {
        vec![
            Self::TwoPlusEoL,
            Self::SingleEol,
            Self::SentencesRuleBased,
            Self::SentencesUnicode,
            Self::WordsUnicode,
            // Self::GraphemesUnicode,
        ]
    }

    pub fn group(&self) -> SeparatorGroup {
        match self {
            Self::TwoPlusEoL
            | Self::SingleEol
            | Self::SentencesRuleBased
            | Self::SentencesUnicode => SeparatorGroup::Semantic,
            Self::WordsUnicode | Self::GraphemesUnicode => SeparatorGroup::Syntactic,
            Self::None => unreachable!(),
        }
    }

    pub fn clean_text(&self, text: &str) -> String {
        match self {
            Self::TwoPlusEoL => TextCleaner::new()
                .reduce_newlines_to_double_newline()
                .run(text),
            Self::SingleEol => TextCleaner::new()
                .reduce_newlines_to_single_newline()
                .run(text),
            Self::SentencesRuleBased
            | Self::SentencesUnicode
            | Self::WordsUnicode
            | Self::GraphemesUnicode => TextCleaner::new()
                .reduce_newlines_to_single_space()
                .run(text),
            Self::None => unreachable!(),
        }
    }

    pub fn split_text_into_indices<T: AsRef<str>>(&self, text: T) -> Vec<Range<usize>> {
        let mut split_indices: Vec<Range<usize>> = Vec::new();
        match self {
            Self::TwoPlusEoL | Self::SingleEol => {
                let pattern_matches = match self {
                    Self::TwoPlusEoL => TWO_PLUS_NEWLINE_REGEX.find_iter(text.as_ref()),
                    Self::SingleEol => SINGLE_NEWLINE_REGEX.find_iter(text.as_ref()),
                    _ => unreachable!(),
                };
                let mut last_end = 0;
                for m in pattern_matches {
                    let start = m.start();
                    let end = m.end();
                    if start > last_end {
                        split_indices.push(Range {
                            start: last_end,
                            end: start,
                        });
                    }
                    split_indices.push(Range { start, end });
                    last_end = end;
                }
                if last_end < text.as_ref().len() {
                    split_indices.push(Range {
                        start: last_end,
                        end: text.as_ref().len(),
                    });
                }
            }
            Self::SentencesRuleBased => {
                split_indices = split_text_into_indices(text.as_ref(), true);
            }
            Self::SentencesUnicode | Self::WordsUnicode | Self::GraphemesUnicode => {
                let indices: Vec<(usize, &str)> = match self {
                    Self::SentencesUnicode => {
                        unicode_segmentation::UnicodeSegmentation::split_sentence_bound_indices(
                            text.as_ref(),
                        )
                        .collect()
                    }
                    Self::WordsUnicode => {
                        unicode_segmentation::UnicodeSegmentation::unicode_word_indices(
                            text.as_ref(),
                        )
                        .collect()
                    }
                    Self::GraphemesUnicode => {
                        unicode_segmentation::UnicodeSegmentation::grapheme_indices(
                            text.as_ref(),
                            true,
                        )
                        .collect()
                    }
                    _ => unreachable!(),
                };
                for i in 0..indices.len() {
                    let end_index = if i == indices.len() - 1 {
                        text.as_ref().len()
                    } else {
                        indices[i + 1].0
                    };
                    split_indices.push(Range {
                        start: indices[i].0,
                        end: end_index,
                    });
                }
            }
            Self::None => unreachable!(),
        }
        split_indices
            .into_iter()
            .filter_map(|indices| self.trim_range(&indices, text.as_ref()))
            .collect()
    }

    pub fn next(&self) -> Option<Self> {
        match self {
            Self::TwoPlusEoL => Some(Self::SingleEol),
            Self::SingleEol => Some(Self::SentencesRuleBased),
            Self::SentencesRuleBased => Some(Self::SentencesUnicode),
            Self::SentencesUnicode => Some(Self::WordsUnicode),
            Self::WordsUnicode => Some(Self::GraphemesUnicode),
            Self::GraphemesUnicode => None,
            Self::None => unreachable!(),
        }
    }
    fn trim_range<T: AsRef<str>>(&self, indices: &Range<usize>, text: T) -> Option<Range<usize>> {
        let (start, end) = match self {
            Self::TwoPlusEoL
            | Self::SingleEol
            | Self::SentencesRuleBased
            | Self::SentencesUnicode => {
                let start = text.as_ref()[indices.start..indices.end]
                    .char_indices()
                    .find(|(_, c)| !c.is_whitespace())
                    .map(|(i, _)| indices.start + i)
                    .unwrap_or(indices.end);
                let end = if indices.end == text.as_ref().len() {
                    text.as_ref().len()
                } else {
                    text.as_ref()[indices.start..indices.end]
                        .char_indices()
                        .rev()
                        .find(|(_, c)| !c.is_whitespace())
                        .map(|(i, c)| indices.start + i + c.len_utf8())
                        .unwrap_or(start)
                };
                (start, end)
            }
            Self::WordsUnicode => {
                let start = text.as_ref()[..indices.start]
                    .char_indices()
                    .rev()
                    .find(|(_, c)| c.is_whitespace())
                    .map(|(i, c)| i + c.len_utf8())
                    .unwrap_or(indices.start);
                let end = if indices.end == text.as_ref().len() {
                    text.as_ref().len()
                } else {
                    text.as_ref()[indices.start..indices.end]
                        .char_indices()
                        .find(|(_, c)| c.is_whitespace())
                        .map(|(i, _)| indices.start + i)
                        .unwrap_or(start)
                };
                (start, end)
            }
            Self::GraphemesUnicode => (indices.start, indices.end),
            Self::None => unreachable!(),
        };

        if start >= end {
            None
        } else {
            Some(Range { start, end })
        }
    }
}

pub static TWO_PLUS_NEWLINE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\n{2,}").unwrap());
pub static SINGLE_NEWLINE_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n").unwrap());

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
