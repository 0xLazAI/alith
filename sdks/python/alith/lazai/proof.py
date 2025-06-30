from eth_abi import encode
from pydantic import BaseModel
from web3 import Web3


class ProofData(BaseModel):
    id: int
    score: int
    file_url: str
    proof_url: str

    def abi_encode(self) -> bytes:
        return encode(
            ["(uint256,uint256,string,string)"],
            [(self.id, self.score, self.file_url, self.proof_url)],
        )


class SettlementData(BaseModel):
    id: str
    user: str
    cost: int
    nonce: int
    user_signature: str

    def abi_encode(self) -> bytes:
        return encode(
            ["(string,address,uint256,uint256,bytes)"],
            [
                (
                    self.id,
                    self.user,
                    self.cost,
                    self.nonce,
                    Web3.to_bytes(hexstr=self.user_signature),
                )
            ],
        )
