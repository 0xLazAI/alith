from pydantic import BaseModel
from typing import Optional


class ProofRequest(BaseModel):
    job_id: int
    file_id: int
    file_url: str
    encryption_key: str
    proof_url: Optional[str]
    encryption_seed: Optional[str] = None
    nonce: Optional[int] = None
