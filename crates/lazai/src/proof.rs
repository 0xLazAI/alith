pub struct ProofData {
    pub file_url: String,
    pub proof_url: Option<String>,
    pub id: usize,
    pub instruction: String,
}

pub struct Proof {
    pub signature: Option<String>,
    pub data: ProofData,
}
