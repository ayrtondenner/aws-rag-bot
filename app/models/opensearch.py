from __future__ import annotations

from pydantic import BaseModel


class IndexExistsResponse(BaseModel):
    index_name: str
    exists: bool


class HybridSearchResponse(BaseModel):
    phrases: list[str]
    documents: list[str]
    phrases_length: int
    documents_length: int
