from pydantic import BaseModel
from eth_abi import encode


class ProofData(BaseModel):
    id: int
    file_url: str
    proof_url: str

    def abi_encode(self) -> bytes:
        return encode(
            ["(uint256,string,string)"], [(self.id, self.file_url, self.proof_url)]
        )


class SettlementProofData(BaseModel):
    id: str
    user: str
    cost: int
    nonce: int
    user_signature: str

    def abi_encode(self) -> bytes:
        return encode(
            ["(string,address,uint256,uint256,bytes)"],
            [(self.id, self.user, self.cost, self.nonce, self.user_signature)],
        )
