"""
Start the node with the PRIVATE_KEY and RSA_PRIVATE_KEY_BASE64 environment variable set to the base64 encoded RSA private key.
python3 -m alith.lazai.node.validator
"""

import base64
import logging
import os
import pathlib
import sys
import json

import rsa
from fastapi import FastAPI, Response, status

from alith.data import decrypt, download_file
from alith.lazai import Client, ProofData, ProofRequest

# Logging configuration

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Environment variables

enable_decrypt_file = os.getenv("ENABLE_DECRYPT_FILE", "")
rsa_private_key_base64 = os.getenv("RSA_PRIVATE_KEY_BASE64", "")
rsa_private_key = (
    base64.b64decode(rsa_private_key_base64).decode()
    if rsa_private_key_base64
    else os.getenv("RSA_PRIVATE_KEY", "")
)

# FastAPI app and LazAI client initialization

app = FastAPI(title="Alith Validator Node", version="1.0.0")
client = Client()


def decrypt_file_url(url: str, encryption_key: str) -> bytes:
    """Download the encrypted file and use the encryption_key hex format to decrypt it."""
    file = download_file(url)
    content = pathlib.Path(file).read_bytes()
    priv_key = rsa.PrivateKey.load_pkcs1(rsa_private_key.strip().encode())
    password = rsa.decrypt(bytes.fromhex(encryption_key.removeprefix("0x")), priv_key)
    return decrypt(content, password=password.decode())


@app.post("/proof")
async def process_proof(req: ProofRequest):
    try:
        # Decrypt the file and check it
        if enable_decrypt_file:
            decrypt_file_url(req.file_url, req.encryption_key)
        client.complete_job(req.job_id)
        client.add_proof(
            req.file_id,
            ProofData(
                id=req.file_id,
                score=1,
                file_url=req.file_url,
                proof_url=req.proof_url or "",
            ),
        )
        client.claim()
        logger.info(f"Successfully processed request for file_id: {req.file_id}")
        return {"success": True}
    except Exception as e:
        logger.error(
            f"Error processing request for file_id: {req.file_id}. Error: {str(e)}"
        )
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=json.dumps(
                {
                    "error": {
                        "message": f"Error processing request for req: {req}. Error: {str(e)}",
                        "type": "internal_error",
                    }
                }
            ),
        )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting node server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
