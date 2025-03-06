use regex::Regex;
use serde_json::{Error as JsonError, Value};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum JsonParseError {
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] JsonError),
    #[error("Missing expected key: {0}")]
    MissingKey(String),
}

pub fn parse_partial_json(s: &str) -> Result<Value, JsonParseError> {
    let mut chars: Vec<char> = s.chars().collect();
    let mut stack = Vec::new();
    let mut is_inside_string = false;
    let mut escaped = false;

    for i in 0..chars.len() {
        let c = chars[i];
        if is_inside_string {
            if c == '"' && !escaped {
                is_inside_string = false;
            } else if c == '\n' && !escaped {
                chars[i] = '\\';
                chars.insert(i + 1, 'n');
            }
            escaped = c == '\\' && !escaped;
        } else {
            match c {
                '"' => {
                    is_inside_string = true;
                    escaped = false;
                }
                '{' => stack.push('}'),
                '[' => stack.push(']'),
                '}' | ']' => {
                    if let Some(last) = stack.last() {
                        if *last == c {
                            stack.pop();
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let mut completed: String = chars.into_iter().collect();
    completed.extend(stack.into_iter().rev());
    serde_json::from_str(&completed)
        .or_else(|_| serde_json::from_str(s))
        .map_err(JsonParseError::from)
}

pub fn parse_json_markdown(text: &str) -> Result<Value, JsonParseError> {
    let re = Regex::new(r"(?s)```(json)?(.*?)(```|$)").unwrap();
    let json_str = if let Some(caps) = re.captures(text) {
        caps[2].trim().to_owned()
    } else {
        text.trim().to_owned()
    };
    parse_partial_json(&json_str)
}

pub fn parse_and_check_json_markdown(
    text: &str,
    expected_keys: &[&str],
) -> Result<Value, JsonParseError> {
    let json_obj = parse_json_markdown(text)?;

    if let Value::Object(map) = &json_obj {
        for key in expected_keys {
            if !map.contains_key(*key) {
                return Err(JsonParseError::MissingKey(key.to_string()));
            }
        }
    }

    Ok(json_obj)
}
