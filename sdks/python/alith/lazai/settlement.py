from pydantic import BaseModel
from web3 import Web3
from typing import Dict
from eth_account.messages import encode_defunct
import json
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
        # 1. Convert the request object to a dictionary
        message_dict = self.model_dump()
        # 2. Sort dictionary keys for consistent message serialization
        sorted_dict = dict(sorted(message_dict.items()))
        # 3. Convert to JSON string (ensure_ascii=False preserves Chinese characters)
        # The message string should be the JSON format e.g., {"node":"0xABCDE02F9bB4E4C8836e38DF4320D4a79106F194","nonce":1234567,"user":"0xBCDEE02F9bB4E4C8836e38DF4320D4a79106F194"}
        message_str = json.dumps(sorted_dict, separators=(",", ":"), ensure_ascii=False)
        # 4. Initialize Web3 instance
        w3 = Web3()
        # 5. Sign the message with the private key.
        signature = w3.eth.account.sign_message(
            encode_defunct(text=message_str),
            private_key=private_key,
        ).signature.hex()

        return SettlementSignature(
            user=self.user,
            nonce=self.nonce,
            signature=signature,
        )
