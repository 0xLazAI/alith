from typing import Optional

from pydantic import BaseModel


class ProofRequest(BaseModel):
    job_id: int
    file_id: int
    file_url: str
    encryption_key: str
    proof_url: Optional[str] = None
    encryption_seed: Optional[str] = None
    nonce: Optional[int] = None
