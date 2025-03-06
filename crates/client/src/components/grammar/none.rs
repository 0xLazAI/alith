use super::{Grammar, GrammarError};

#[derive(Clone, PartialEq, Default)]
pub struct NoneGrammar {
    pub stop_word_done: Option<String>,
    pub stop_word_no_result: Option<String>,
}

impl NoneGrammar {
    #[inline]
    pub fn wrap(self) -> Grammar {
        Grammar::NoneGrammar(self)
    }

    #[inline]
    pub fn grammar_string(&self) -> String {
        String::new()
    }

    #[inline]
    pub fn validate_clean(&self, content: &str) -> Result<String, GrammarError> {
        Ok(content.to_owned())
    }

    #[inline]
    pub fn grammar_parse(&self, content: &str) -> Result<String, GrammarError> {
        Ok(content.to_owned())
    }
}
