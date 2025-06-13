import argparse

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .service import router


def run(
    host: str = "localhost",
    port: int = 8000,
):
    """Run an training server with the given address"""

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/v1/training", tags=["training"])

    return uvicorn.run(
        app,
        host=host,
        port=port,
    )


if __name__ == "__main__":
    description = "Alith training server. Host your own trained LLMs!🚀"
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
    args = parser.parse_args()

    run(host=args.host, port=args.port)
