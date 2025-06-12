from fastapi import Request
from .client import Client
from web3 import Web3
from eth_account.messages import encode_defunct
import json

USER_HEADER = "X-LazAI-User"
NONCE_HEADER = "X-LazAI-Nonce"
SIGNATURE_HEADER = "X-LazAI-Signature"

TRAINING_TYPE = 1
INFERENCE_TYPE = 2


def validate_request(
    request: Request, type: int = TRAINING_TYPE, client: Client | None = None
):
    """Validate the request user and signature in the request headers"""
    user = request.headers[USER_HEADER]
    nonce = request.headers[NONCE_HEADER]
    signature = request.headers[SIGNATURE_HEADER]
    validate_account_and_signature(user, nonce, signature, type, client)


def validate_account_and_signature(
    user: str,
    nonce: int,
    signature: str,
    type: int = TRAINING_TYPE,
    client: Client | None = None,
):
    """Validate the request user and signature with the user address, nonce and signature"""
    client = client or Client()
    account = (
        client.get_training_account(user, client.wallet.address)
        if type == TRAINING_TYPE
        else client.get_inference_account(user, client.wallet.address)
    )
    if not account or account[0] != user:
        raise Exception(f"Account {user} does not exist or is unauthorized")
    last_nonce = account[2]
    if nonce <= last_nonce:
        raise Exception(
            f"Invalid nonce: {nonce}. Must be greater than last nonce: {last_nonce}"
        )
    message_dict = {"nonce": nonce, "user": user, "node": client.wallet.address}
    sorted_dict = dict(sorted(message_dict.items()))
    message_str = json.dumps(sorted_dict, separators=(",", ":"), ensure_ascii=False)
    w3 = Web3()
    message = encode_defunct(text=message_str)
    recovered_address = w3.eth.account.recover_message(message, signature=signature)
    if recovered_address.lower() != user.lower():
        raise Exception("Signature verification failed: address mismatch")
