"""
Start the node with the PRIVATE_KEY and RSA_PRIVATE_KEY_BASE64 environment variable set to the base64 encoded RSA private key.
python3 -m alith.lazai.node.fake
"""

from flask import Flask, request, jsonify
from alith.lazai import (
    Client,
    ProofData,
    ProofRequest,
)
import os
import logging
import base64

# Logging configuration

logging.basicConfig(
    filename="node.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Environment variables

rsa_private_key_base64 = os.getenv("RSA_PRIVATE_KEY_BASE64", "")
rsa_private_key = (
    base64.b64decode(rsa_private_key_base64).decode() if rsa_private_key_base64 else ""
)

# Flask app and LazAI client initialization

app = Flask(__name__)
client = Client()


@app.route("/proof", methods=["POST"])
def proof():
    try:
        data = request.get_json()
        req = ProofRequest(**data)
        # We do not check the encryption key and file in the fake node just for simplicity.
        client.complete_job(req.job_id)
        client.add_proof(
            req.file_id,
            ProofData(
                id=req.file_id, file_url=req.file_url, proof_url=req.proof_url or ""
            ),
        )
        client.claim()
        logger.info(f"Successfully processed request for file_id: {req.file_id}")
        return jsonify({"success": True}), 200
    except Exception as e:
        logger.error(
            f"Error processed request for file_id: {req.file_id} and error {str(e)}"
        )
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    logger.info("Starting node server...")
    app.run(host="0.0.0.0", port=8000, debug=True)
