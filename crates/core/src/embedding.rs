use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Deserialize, Serialize, Debug)]
pub struct Embedding {
    pub document: String,
    pub vec: Vec<f64>,
}
