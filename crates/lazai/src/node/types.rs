use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, bon::Builder)]
pub struct ProofRequest {
    job_id: usize,
    file_id: usize,
    file_url: String,
    encryption_key: String,
    proof_url: Option<String>,
    encryption_seed: Option<String>,
    nonce: Option<usize>,
}
