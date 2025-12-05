"""
Start the node with the PRIVATE_KEY and RSA_PRIVATE_KEY_BASE64 environment variable set to the base64 encoded RSA private key.
python3 -m alith.inference.server
"""

import argparse
import uvicorn


def run(
    host: str = "localhost",
    port: int = 8000,
    engine_type: str = "llamacpp",
    *,
    model: str,
    settlement: bool = False,
    graceful_settlement: bool = False,
    
    store_type: str="chromadb",
    llm_base_url: str,
    llm_api_key: str,
    private_key: str
):
    """Run an inference server with the given address and engine type.
    Args:
        host: The host to run the server on (e.g. localhost).
        port: The port to run the server on (e.g. 8000).
        engine_type: The type of engine to use (available: llamacpp, openai).
        model: The model to use (e.g. deepseek/deepseek-r1-0528).
        settlement: Whether to enable the settlement middleware (e.g. True).
        store_type: The type of store to use (available: chromadb, milvus, faiss).
        llm_base_url: The base URL of the LLM (e.g. https://api.groq.com/openai/v1).
        llm_api_key: The API key of the LLM (e.g. gsk_LQYB...HFZa4z3Fh).
        private_key: The private key to use (e.g. 0x1234567890a...67890a...).
    """
    
    # Set environment variables from parameters if provided
    import os
    if store_type:
        os.environ["ALITH_STORE_TYPE"] = store_type
    if llm_base_url:
        os.environ["LLM_BASE_URL"] = llm_base_url
    if llm_api_key:
        os.environ["LLM_API_KEY"] = llm_api_key
    if private_key:
        os.environ["PRIVATE_KEY"] = private_key

    if engine_type == "llamacpp":
        from llama_cpp.server.app import create_app
        from llama_cpp.server.settings import ModelSettings, ServerSettings

        server_settings = ServerSettings(host=host, port=port)
        model_settings = [ModelSettings(model=model)]
        app = create_app(server_settings=server_settings, model_settings=model_settings)
        if settlement:
            from .settlement import TokenBillingMiddleware
            from .query import DataQueryMiddleware
            from ..lazai.node.middleware import HeaderValidationMiddleware
            from ..lazai.request import INFERENCE_TYPE

            app.add_middleware(HeaderValidationMiddleware, type=INFERENCE_TYPE)
            app.add_middleware(DataQueryMiddleware)
            app.add_middleware(TokenBillingMiddleware, graceful=graceful_settlement)

        return uvicorn.run(
            app,
            host=server_settings.host,
            port=int(server_settings.port),
            ssl_keyfile=server_settings.ssl_keyfile,
            ssl_certfile=server_settings.ssl_certfile,
        )
    elif engine_type == "openai":
        from .proxy_server import app

        if settlement:
            from .settlement import TokenBillingMiddleware
            from .query import DataQueryMiddleware
            from ..lazai.node.middleware import HeaderValidationMiddleware
            from ..lazai.request import INFERENCE_TYPE

            app.add_middleware(HeaderValidationMiddleware, type=INFERENCE_TYPE)
            app.add_middleware(DataQueryMiddleware)
            app.add_middleware(TokenBillingMiddleware, graceful=graceful_settlement)

        return uvicorn.run(
            app,
            host=host,
            port=port,
        )
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    description = "Alith inference server. Host your own or remote LLMs!🚀"
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
    parser.add_argument(
        "--engine",
        type=str,
        help="Engine type",
        default="llamacpp",
    )
    parser.add_argument(
        "--settlement",
        type=bool,
        help="Enable the settlement middleware",
        default=True,
    )
    args = parser.parse_args()

    run(
        host=args.host,
        port=args.port,
        engine_type=args.engine,
        model=args.model,
        settlement=args.settlement,
    )
