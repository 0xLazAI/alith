from typing import Dict

from eth_abi import encode
from eth_account.messages import encode_defunct
from hexbytes import HexBytes
from pydantic import BaseModel
from web3 import Web3

from .request import NONCE_HEADER, SIGNATURE_HEADER, USER_HEADER, FILE_ID_HEADER


class SettlementSignature(BaseModel):
    """SettlementSignature contains headers related to the AI inference or
    training request."""

    user: str
    nonce: int
    signature: str
    file_id: int | None = None

    def to_request_headers(self) -> Dict[str, str]:
        return {
            USER_HEADER: self.user,
            NONCE_HEADER: str(self.nonce),
            SIGNATURE_HEADER: self.signature,
            FILE_ID_HEADER: str(self.file_id) if self.file_id is not None else "",
        }


class SettlementRequest(BaseModel):
    """Represents an abstract settlement request, which contains the node
    address providing AI services including inference and training, the
    user address and nonce, which will be used to request signature."""

    nonce: int
    user: str
    node: str
    file_id: int | None = None

    def abi_encode(self) -> bytes:
        return encode(
            ["uint256", "address", "address"], [self.nonce, self.user, self.node]
        )

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
        message_hash = Web3.keccak(self.abi_encode())
        eth_message = encode_defunct(primitive=message_hash)
        signature = (
            Web3().eth.account.sign_message(eth_message, private_key).signature.hex()
        )

        return SettlementSignature(
            user=self.user,
            nonce=self.nonce,
            signature=HexBytes(signature).hex(),
            file_id=self.file_id,
        )
