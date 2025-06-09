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
    id: int
    user: str
    cost: int
    nonce: int

    def abi_encode(self) -> bytes:
        return encode(
            ["(uint256,address,uint256,uint256)"],
            [(self.id, self.user, self.cost, self.nonce)],
        )
