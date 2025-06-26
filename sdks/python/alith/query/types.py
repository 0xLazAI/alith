from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    file_id: Optional[int] = None
    file_url: Optional[str] = None
    limit: int = 3
    query: str
