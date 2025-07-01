from eth_abi import encode
from eth_account.messages import encode_defunct
from fastapi import Request
from web3 import Web3

USER_HEADER = "X-LazAI-User"
NONCE_HEADER = "X-LazAI-Nonce"
SIGNATURE_HEADER = "X-LazAI-Signature"
TOKEN_ID_HEADER = "X-LazAI-Token-ID"
FILE_ID_HEADER = "X-LazAI-File-ID"

QUERY_TYPE = 0
INFERENCE_TYPE = 1
TRAINING_TYPE = 2


def validate_request(request: Request, type: int = QUERY_TYPE, client=None):
    """Validate the request user and signature in the request headers"""
    user = request.headers[USER_HEADER]
    nonce = request.headers[NONCE_HEADER]
    signature = request.headers[SIGNATURE_HEADER]
    validate_account_and_signature(user, int(nonce), signature, type, client)


def validate_account_and_signature(
    user: str,
    nonce: int,
    signature: str,
    type: int = QUERY_TYPE,
    client=None,
):
    """Validate the request user and signature with the user address, nonce and signature"""
    from .client import Client

    client = client or Client()
    node = client.wallet.address
    account = (
        client.get_training_account(user, node)
        if type == TRAINING_TYPE
        else (
            client.get_inference_account(user, node)
            if type == INFERENCE_TYPE
            else client.get_query_account(user, node)
        )
    )
    if not account or account[0] != user:
        raise Exception(f"Account {user} does not exist or is unauthorized")
    last_nonce = account[2]
    if nonce <= last_nonce:
        raise Exception(
            f"Invalid nonce: {nonce}. Must be greater than last nonce: {last_nonce}"
        )
    recovered_address = recover_address(
        nonce,
        user,
        node,
        signature,
    )
    if recovered_address.lower() != user.lower():
        raise Exception("Signature verification failed: address mismatch")


def recover_address(
    nonce: int,
    user: str,
    node: str,
    signature: str,
) -> str:
    message_hash = Web3.keccak(
        encode(["uint256", "address", "address"], (nonce, user, node))
    )
    eth_message = encode_defunct(primitive=message_hash)
    recovered_address = Web3().eth.account.recover_message(
        eth_message, signature=signature
    )
    return recovered_address
