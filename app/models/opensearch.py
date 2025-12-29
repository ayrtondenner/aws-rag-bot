from __future__ import annotations

from pydantic import BaseModel


class IndexExistsResponse(BaseModel):
    index_name: str
    exists: bool


class IndexSageMakerDocsResponse(BaseModel):
    search_index_name: str
    vector_index_name: str
    documents_indexed: int
    chunks_indexed: int
