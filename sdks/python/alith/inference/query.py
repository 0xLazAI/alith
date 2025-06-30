import json
from typing import Awaitable, Callable
import logging

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from ..lazai.client import Client
from ..lazai.request import FILE_ID_HEADER
from ..query.server import QueryRequest, query_rag


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class DataQueryMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, client: Client = Client()):
        super().__init__(app)
        self.client: Client = client

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable]
    ) -> Response:
        try:
            file_id = request.headers.get(FILE_ID_HEADER, "")
            # If file_id is set, do the RAG query and insert the results into the prompt
            if request.url.path == "/v1/chat/completions" and file_id:
                # Get the prompt from the request body
                body = await request.body()
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    return Response(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content=json.dumps(
                            {
                                "error": {
                                    "message": "Invalid JSON format in request body",
                                    "type": "bad_request_error",
                                }
                            }
                        ),
                    )

                prompt = ""
                # Chat completions can have different structures
                if "messages" in data:
                    if data["messages"] and isinstance(data["messages"], list):
                        if (
                            isinstance(data["messages"][-1], dict)
                            and "content" in data["messages"][-1]
                        ):
                            prompt = data["messages"][-1]["content"]
                elif "prompt" in data:
                    prompt = data["prompt"]

                resp = await query_rag(QueryRequest(file_id=int(file_id), query=prompt))

                # Deal error response from query_rag
                if isinstance(resp, Response):
                    return resp

                docs = resp.get("data", [])
                if not isinstance(docs, list):
                    docs = [docs]

                updated_prompt = "{}\n\n<attachments>\n{}</attachments>\n".format(
                    prompt, "".join(docs)
                )
                # Update the request body with the new prompt
                if "messages" in data:
                    data["messages"][-1]["content"] = updated_prompt
                elif "prompt" in data:
                    data["prompt"] = updated_prompt

                new_body = json.dumps(data).encode("utf-8")
                request._body = new_body
                return await call_next(request)

            return await call_next(request)
        except Exception as e:
            return Response(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=json.dumps(
                    {
                        "error": {
                            "message": "Data query failed: " + str(e),
                            "type": "bad_request_error",
                        }
                    }
                ),
            )
