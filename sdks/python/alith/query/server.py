"""
Start the node with the PRIVATE_KEY and RSA_PRIVATE_KEY_BASE64 environment variable set to the base64 encoded RSA private key.
python3 -m alith.query.server

curl http://localhost:8000/query \
-H "Content-Type: application/json" \
-H "X-LazAI-User: 0x34d9E02F9bB4E4C8836e38DF4320D4a79106F194" \
-H "X-LazAI-Nonce: 123456" \
-H "X-LazAI-Signature: HSDGYUSDOWP123" \
-d '{
  "query": "What is Alith?",
  "file_id": 1
}'
"""

import logging
import sys
import json
import uvicorn
import argparse

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from alith.lazai import Client
from alith.lazai.node.middleware import HeaderValidationMiddleware
from alith.lazai.node.validator import decrypt_file_url
from alith.store import MilvusStore
from .types import QueryRequest
from .settlement import QueryBillingMiddleware

# Logging configuration

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
client = Client()
app = FastAPI(title="Alith LazAI Privacy Data Query Node", version="1.0.0")


@app.post("/query/rag")
async def process_proof(req: QueryRequest):
    try:
        file_id = req.file_id
        if req.file_url:
            file_id = client.get_file_id_by_url(req.file_url)
        if file_id:
            file = client.get_file(file_id)
        else:
            return Response(
                status_code=400,
                content=json.dumps(
                    {
                        "error": {
                            "message": "File ID or URL is required",
                            "type": "invalid_request_error",
                        }
                    }
                ),
            )
        owner, file_url, file_hash = file[1], file[2], file[3]
        encryption_key = client.get_file_permission(
            file_id, client.contract_config.data_registry_address
        )
        data = decrypt_file_url(file_url, encryption_key)
        logger.info(f"Successfully processed request for file: {file}")
        return {
            "data": data.decode("utf-8") if req.query else data.decode("utf-8"),
            "owner": owner,
            "file_id": file_id,
            "file_url": file_url,
            "file_hash": file_hash,
        }
    except Exception as e:
        return Response(
            status_code=400,
            content=json.dumps(
                {
                    "error": {
                        "message": f"Error processing request for req: {req}. Error: {str(e)}",
                        "type": "internal_error",
                    }
                }
            ),
        )


def run(host: str = "localhost", port: int = 8000, *, settlement: bool = False):

    # FastAPI app and LazAI client initialization

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if settlement:
        app.add_middleware(HeaderValidationMiddleware)
        app.add_middleware(QueryBillingMiddleware)

    return uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    description = "Alith data query server. Host your own embedding models and support language query!"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--host",
        type=str,
        help="Server host",
        default="localhost",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Server port",
        default=8000,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name or path",
        default="/root/models/qwen2.5-1.5b-instruct-q5_k_m.gguf",
    )
    args = parser.parse_args()

    run(host=args.host, port=args.port, settlement=True)
