import uvicorn
import argparse


def run(
    host: str = "localhost",
    port: int = 8000,
    engine_type: str = "llamacpp",
    *,
    model: str
):
    """Run an inference server with the given address and engine type."""
    if engine_type not in ["llamacpp"]:
        raise
    if engine_type == "llamacpp":
        from llama_cpp.server.app import create_app
        from llama_cpp.server.settings import (
            ServerSettings,
            ModelSettings,
        )

        server_settings = ServerSettings(host=host, port=port)
        model_settings = [ModelSettings(model=model)]
        app = create_app(server_settings=server_settings, model_settings=model_settings)
        return uvicorn.run(
            app,
            host=server_settings.host,
            port=int(server_settings.port),
            ssl_keyfile=server_settings.ssl_keyfile,
            ssl_certfile=server_settings.ssl_certfile,
        )
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    description = "Alith inference server. Host your own LLMs!🚀"
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

    run(host=args.host, port=args.port, model=args.model)
