import json
from typing import Awaitable, Callable

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from ..client import Client
from ..request import validate_request, QUERY_TYPE


class HeaderValidationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, type: int = QUERY_TYPE, client: Client = Client()):
        super().__init__(app)
        self.type: int = type
        self.client: Client = client

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable]
    ) -> Response:
        try:
            validate_request(request, type=self.type, client=self.client)
            response = await call_next(request)
            return response
        except Exception as e:
            return Response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=json.dumps(
                    {
                        "error": {
                            "message": "Validate the request header failed: " + str(e),
                            "type": "authentication_error",
                        }
                    }
                ),
            )
