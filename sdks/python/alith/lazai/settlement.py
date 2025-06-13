from pydantic import BaseModel
from web3 import Web3
from typing import Dict
from eth_account.messages import encode_defunct
from eth_abi import encode
from eth_account import Account
from hexbytes import HexBytes
from .request import USER_HEADER, NONCE_HEADER, SIGNATURE_HEADER


class SettlementSignature(BaseModel):
    """SettlementSignature contains headers related to the AI inference or
    training request."""

    user: str
    nonce: int
    signature: str

    def to_request_headers(self) -> Dict[str, str]:
        return {
            USER_HEADER: self.user,
            NONCE_HEADER: str(self.nonce),
            SIGNATURE_HEADER: self.signature,
        }


class SettlementRequest(BaseModel):
    """Represents an abstract settlement request, which contains the node
    address providing AI services including inference and training, the
    user address and nonce, which will be used to request signature."""

    nonce: int
    user: str
    node: str

    def abi_encode(self) -> bytes:
        return encode(["(uint256,address,address)"], [(self.nonce, self.user, self.node)])

    def generate_signature(
        self,
        private_key: str,
    ) -> SettlementSignature:
        """
        Generates an Ethereum signature for a SettlementRequest object.

        Args:
            private_key: The user's Ethereum private key for signing.

        Returns:
            A SettlementSignature object containing signature information.
        """
        packed_data = self.abi_encode()
        message_hash = Web3.keccak(packed_data)
        eth_message = Web3.keccak(b"\x19Ethereum Signed Message:\n32" + message_hash)
        signed_message = Account.signHash(eth_message, private_key)
        signature = signed_message.signature
        signature = HexBytes(signature).hex()

        return SettlementSignature(
            user=self.user,
            nonce=self.nonce,
            signature=signature,
        )
