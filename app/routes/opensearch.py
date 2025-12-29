from __future__ import annotations

from fastapi import APIRouter, Depends, Path
from starlette.concurrency import run_in_threadpool

from app.models.opensearch import IndexExistsResponse, IndexSageMakerDocsResponse
from app.services.dependencies import get_opensearch_service, get_sagemaker_docs_opensearch_index_service
from app.services.opensearch_service import OpenSearchService
from app.services.sagemaker_docs_opensearch_index_service import SageMakerDocsOpenSearchIndexService


router = APIRouter(prefix="/opensearch", tags=["opensearch"])


@router.get("/indexes/{index_name}/exists", response_model=IndexExistsResponse)
async def index_exists(
    index_name: str = Path(..., description="OpenSearch index name"),
    svc: OpenSearchService = Depends(get_opensearch_service),
) -> IndexExistsResponse:
    exists = await run_in_threadpool(svc.index_exists, index_name=index_name)
    return IndexExistsResponse(index_name=index_name, exists=exists)


@router.post("/sagemaker-docs/index", response_model=IndexSageMakerDocsResponse)
async def index_sagemaker_docs(
    svc: SageMakerDocsOpenSearchIndexService = Depends(get_sagemaker_docs_opensearch_index_service),
) -> IndexSageMakerDocsResponse:
    documents_indexed, chunks_indexed = await run_in_threadpool(svc.index_local_docs)
    return IndexSageMakerDocsResponse(
        search_index_name=svc.search_index_name,
        vector_index_name=svc.vector_index_name,
        documents_indexed=documents_indexed,
        chunks_indexed=chunks_indexed,
    )
