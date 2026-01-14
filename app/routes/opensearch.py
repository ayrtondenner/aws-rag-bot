from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query

from app.models.opensearch import (
    HybridSearchResponse,
    IndexExistsResponse,
)
from app.services.dependencies import (
    get_opensearch_service,
)
from app.services.opensearch_service import OpenSearchService

router = APIRouter(prefix="/opensearch", tags=["opensearch"])

@router.get("/indexes/{index_name}/exists", response_model=IndexExistsResponse)
async def index_exists(
    index_name: str = Path(..., description="OpenSearch index name"),
    svc: OpenSearchService = Depends(get_opensearch_service),
) -> IndexExistsResponse:
    """Check whether an OpenSearch index exists."""

    exists = await svc.index_exists(index_name=index_name)
    return IndexExistsResponse(index_name=index_name, exists=exists)



@router.get("/hybrid-search", response_model=HybridSearchResponse)
async def hybrid_search(
    q: str = Query(..., description="Text query"),
    k_text: int = Query(5, ge=1, le=50, description="Top-k lexical results"),
    k_vector: int = Query(5, ge=1, le=50, description="Top-k vector results"),
    svc: OpenSearchService = Depends(get_opensearch_service),
) -> HybridSearchResponse:
    """Run hybrid search (lexical + vector) and return relevant phrases + source documents."""

    phrases, documents = await svc.hybrid_search(query=q, k_text=k_text, k_vector=k_vector)
    return HybridSearchResponse(
        phrases=phrases,
        documents=documents,
        phrases_length=len(phrases),
        documents_length=len(documents),
    )
