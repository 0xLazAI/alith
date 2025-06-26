import json
import logging
from typing import Awaitable, Callable

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from ..lazai.client import Client, SettlementData
from ..lazai.request import NONCE_HEADER, USER_HEADER, SIGNATURE_HEADER
from .config import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
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

                new_response = Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
                new_response.headers["content-length"] = str(len(response_body))
                return new_response

            except json.JSONDecodeError:
                logger.warning("Failed to parse response for token billing")
                return Response(
                    status_code=status.HTTP_400_BAD_REQUEST,
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
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content=json.dumps(
                        {
                            "error": {
                                "message": f"Error in token billing process: {str(e)}",
                                "type": "internal_error",
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
    signature = request.headers[SIGNATURE_HEADER]
    cost = total_tokens * price_per_token
    client.inference_settlement_fees(
        SettlementData(
            id=id, user=user, cost=cost, nonce=int(nonce), user_signature=signature
        )
    )
    return user, cost
