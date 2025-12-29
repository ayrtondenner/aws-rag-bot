from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Path
from starlette.concurrency import run_in_threadpool

from app.models.opensearch import IndexExistsResponse, IndexSageMakerDocsResponse
from app.services.dependencies import (
    get_document_text_service,
    get_opensearch_search_service,
    get_opensearch_vector_service,
    get_sagemaker_docs_service,
)
from app.services.document_text_service import DocumentTextService
from app.services.opensearch_service import OpenSearchService
from app.services.sagemaker_docs_service import SageMakerDocsService

router = APIRouter(prefix="/opensearch", tags=["opensearch"])


@router.get("/indexes/{index_name}/exists", response_model=IndexExistsResponse)
async def index_exists(
    index_name: str = Path(..., description="OpenSearch index name"),
    svc: OpenSearchService = Depends(get_opensearch_search_service),
) -> IndexExistsResponse:
    exists = await run_in_threadpool(svc.index_exists, index_name=index_name)
    return IndexExistsResponse(index_name=index_name, exists=exists)


@router.post("/sagemaker-docs/index", response_model=IndexSageMakerDocsResponse)
async def index_sagemaker_docs(
    docs: SageMakerDocsService = Depends(get_sagemaker_docs_service),
    text: DocumentTextService = Depends(get_document_text_service),
    search: OpenSearchService = Depends(get_opensearch_search_service),
    vector: OpenSearchService = Depends(get_opensearch_vector_service),
) -> IndexSageMakerDocsResponse:

    def _search_mapping() -> dict[str, object]:
        return {
            "properties": {
                "doc_id": {"type": "keyword"},
                "path": {"type": "keyword"},
                "title": {"type": "text"},
                "content": {"type": "text"},
                "source": {"type": "keyword"},
            }
        }

    def _vector_mapping(*, dimension: int) -> dict[str, object]:
        return {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "path": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "text": {"type": "text"},
                "embedding": {"type": "knn_vector", "dimension": dimension},
                "source": {"type": "keyword"},
            }
        }

    def _index_all() -> tuple[str, str, int, int]:
        docs_dir = docs.docs_dir
        if not docs_dir.exists() or not docs_dir.is_dir():
            raise ValueError(f"Docs directory not found: {docs_dir}")

        search_index = os.getenv("OPENSEARCH_SEARCH_INDEX_NAME", "sagemaker-docs")
        vector_index = os.getenv("OPENSEARCH_VECTOR_INDEX_NAME", "sagemaker-docs-vectors")

        if not search.index_exists(index_name=search_index):
            search.create_index_and_mapping(index_name=search_index, mapping=_search_mapping())

        dimension = int(os.getenv("BEDROCK_EMBEDDING_DIM", "1024"))
        if dimension <= 0:
            dimension = 1024
        if not vector.index_exists(index_name=vector_index):
            vector.create_index_and_mapping(
                index_name=vector_index,
                mapping=_vector_mapping(dimension=dimension),
                settings={"index.knn": True},
            )

        documents_indexed = 0
        chunks_indexed = 0

        for path in docs.list_markdown_files():
            rel_path = docs.relative_path(path=path)
            doc_id = docs.doc_id_from_rel_path(rel_path)
            title = path.stem
            content = docs.read_text_file(path)

            search.index_document(
                index_name=search_index,
                document_id=doc_id,
                document={
                    "doc_id": doc_id,
                    "path": rel_path,
                    "title": title,
                    "content": content,
                    "source": docs.source_name,
                },
            )
            documents_indexed += 1

            for i, chunk_text in enumerate(text.split_text_into_chunks(content)):
                embedding = text.text_to_embedding(chunk_text)
                chunk_id = f"{doc_id}_{i}"
                vector.index_document(
                    index_name=vector_index,
                    document_id=chunk_id,
                    document={
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "path": rel_path,
                        "chunk_index": i,
                        "text": chunk_text,
                        "embedding": embedding,
                        "source": docs.source_name,
                    },
                )
                chunks_indexed += 1

        return (search_index, vector_index, documents_indexed, chunks_indexed)

    search_index, vector_index, documents_indexed, chunks_indexed = await run_in_threadpool(_index_all)
    return IndexSageMakerDocsResponse(
        search_index_name=search_index,
        vector_index_name=vector_index,
        documents_indexed=documents_indexed,
        chunks_indexed=chunks_indexed,
    )
