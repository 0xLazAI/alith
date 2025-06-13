from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Awaitable
from ..lazai.request import validate_request, USER_HEADER, NONCE_HEADER
from ..lazai.client import Client
from .config import Config
import json
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ValidationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, client: Client = Client()):
        super().__init__(app)
        self.client: Client = client

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable]
    ) -> Response:
        try:
            validate_request(request, client=self.client)
            response = await call_next(request)
            return response
        except Exception as e:
            return Response(
                status_code=401,
                content=json.dumps(
                    {
                        "error": {
                            "message": "Validate the request header failed: " + str(e),
                            "type": "authentication_error",
                        }
                    }
                ),
            )


class TokenBillingMiddleware(BaseHTTPMiddleware):
    """Token consumption billing middleware for /v1/chat/completions endpoint."""

    def __init__(self, app, client: Client = Client(), config: Config = Config()):
        super().__init__(app)
        self.client: Client = client
        self.config: Config = config

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Process the request and get the response
        response = await call_next(request)
        # Only process successful responses to /v1/chat/completions
        # TODO: Settlement for other request including embeddings, completions, etc.
        if request.url.path == "/v1/chat/completions" and response.status_code == 200:
            try:
                # Read the response body
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk
                # Parse response to extract token usage
                response_data = json.loads(response_body.decode("utf-8"))
                id = response_data.get("id", 0)
                usage = response_data.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
                user, cost = calculate_billing(
                    request, id, total_tokens, self.config.price_per_token, self.client
                )
                logger.info(
                    f"User {user} consumed {total_tokens} tokens on /v1/chat/completions, billing: {cost}"
                )
                # Reconstruct the response body to maintain original data
                response.body = response_body
                response.init_content()

            except json.JSONDecodeError:
                logger.warning("Failed to parse response for token billing")
                return Response(
                    status_code=400,
                    content=json.dumps(
                        {
                            "error": {
                                "message": "Failed to parse response for token billing",
                                "type": "invalid_request_error",
                            }
                        }
                    ),
                )
            except Exception as e:
                logger.error(f"Error in token billing process: {str(e)}")
                return Response(
                    status_code=400,
                    content=json.dumps(
                        {
                            "error": {
                                "message": f"Error in token billing process: {str(e)}",
                                "type": "invalid_request_error",
                            }
                        }
                    ),
                )

        return response


def calculate_billing(
    request: Request,
    id: str,
    total_tokens: int,
    price_per_token: int,
    client: Client = Client(),
) -> tuple[int, int]:
    user = request.headers[USER_HEADER]
    nonce = request.headers[NONCE_HEADER]
    cost = total_tokens * price_per_token
    client.inference_settlement_fees(user, cost, id, nonce)
    return user, cost
