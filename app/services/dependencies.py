from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import aiohttp
from fastapi import FastAPI, Request

from app.services.config import OpenSearchConfig, S3Config
from app.services.document_service import DocumentService, LocalDocsConfig
from app.services.opensearch_service import OpenSearchService
from app.services.s3_service import S3Service
from app.services.setup.opensearch_setup_service import OpenSearchSetupService
from app.services.setup.s3_setup_service import S3SetupService

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _docs_dir() -> Path:
    return _project_root() / "sagemaker-docs"


def get_s3_service() -> S3Service:
    """FastAPI dependency provider for an S3Service instance."""

    return S3Service(S3Config.from_env())


def get_document_service() -> DocumentService:
    """Dependency provider for document-only utilities (local docs + split + embeddings)."""

    source_name = os.getenv("OPENSEARCH_DOCS_SOURCE_NAME", "sagemaker-docs")
    return DocumentService(docs=LocalDocsConfig(docs_dir=_docs_dir(), source_name=source_name))


def get_http_session_from_app(app: FastAPI) -> aiohttp.ClientSession:
    session = getattr(app.state, "http_session", None)
    if session is None:
        raise RuntimeError("HTTP session not initialized (app.state.http_session)")
    if not isinstance(session, aiohttp.ClientSession):
        raise RuntimeError("Unexpected http_session type")
    return session


def get_http_session(request: Request) -> aiohttp.ClientSession:
    return get_http_session_from_app(request.app)


def get_opensearch_service(request: Request) -> OpenSearchService:
    return get_opensearch_service_from_app(request.app)


def get_opensearch_service_from_app(app: FastAPI) -> OpenSearchService:
    return OpenSearchService(
        search_config=OpenSearchConfig.from_env_search(),
        vector_config=OpenSearchConfig.from_env_vector(),
        session=get_http_session_from_app(app),
        documents=get_document_service(),
    )


def get_s3_setup_service() -> S3SetupService:
    return S3SetupService(s3=get_s3_service())


def get_opensearch_setup_service(request: Request) -> OpenSearchSetupService:
    return OpenSearchSetupService(opensearch=get_opensearch_service(request))


def get_opensearch_setup_service_from_app(app: FastAPI) -> OpenSearchSetupService:
    """Helper for non-request contexts (e.g. app lifespan startup)."""

    return OpenSearchSetupService(opensearch=get_opensearch_service_from_app(app))


async def startup_ingest_sagemaker_docs(*, app: FastAPI) -> dict[str, int]:
    """Idempotent startup ingestion of local SageMaker docs into S3 + OpenSearch."""

    docs = get_document_service()
    docs_dir = docs.docs_dir

    if not docs_dir.exists() or not docs_dir.is_dir():
        logger.warning("SageMaker docs dir not found, skipping ingestion: %s", docs_dir)
        return {"local_docs": 0, "s3_uploaded": 0, "search_indexed": 0, "vector_chunks_indexed": 0}

    s3 = get_s3_service()
    opensearch = get_opensearch_service_from_app(app)

    local_md_files = docs.list_markdown_files()
    logger.info("SageMaker docs ingestion: found %d markdown files", len(local_md_files))

    # 1) S3 sync
    prefix = (os.getenv("SAGEMAKER_DOCS_S3_PREFIX", "sagemaker-docs/") or "").strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    existing_items = await s3.list_files(prefix=prefix)
    existing_keys = {item.key for item in existing_items}

    planned_uploads: list[tuple[Path, str]] = []
    for path in local_md_files:
        rel_key = path.relative_to(docs_dir).as_posix()
        key = f"{prefix}{rel_key}" if prefix else rel_key
        if key not in existing_keys:
            planned_uploads.append((path, key))

    s3_uploaded = 0
    if planned_uploads:
        concurrency = int(os.getenv("SAGEMAKER_DOCS_S3_CONCURRENCY", "10"))
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _upload_one(p: Path, key: str) -> bool:
            async with sem:
                await s3.upload_local_file(path=p, key=key, content_type="text/markdown")
                return True

        results = await asyncio.gather(*[_upload_one(p, k) for p, k in planned_uploads], return_exceptions=True)
        for r in results:
            if r is True:
                s3_uploaded += 1
            elif isinstance(r, Exception):
                logger.error("S3 upload failed: %s", str(r))

    # 2) OpenSearch indexing (idempotent)
    search_indexed = 0
    vector_chunks_indexed = 0

    opensearch_conc = int(os.getenv("SAGEMAKER_DOCS_OPENSEARCH_CONCURRENCY", "15"))
    embed_conc = int(os.getenv("SAGEMAKER_DOCS_EMBEDDING_CONCURRENCY", "3"))
    os_sem = asyncio.Semaphore(max(1, opensearch_conc))
    embed_sem = asyncio.Semaphore(max(1, embed_conc))

    async def _index_one_doc(path: Path) -> tuple[int, int]:
        rel_path = docs.relative_path(path=path)
        doc_id = docs.doc_id_from_rel_path(rel_path)
        title = path.stem
        content = docs.read_text_file(path)

        indexed_text = 0
        indexed_chunks = 0

        async with os_sem:
            exists = await opensearch.text_document_exists(doc_id=doc_id)
        if not exists:
            async with os_sem:
                await opensearch.index_text_document(
                    doc_id=doc_id,
                    path=rel_path,
                    title=title,
                    content=content,
                    source=docs.source_name,
                )
            indexed_text = 1

        async with os_sem:
            has_embeddings = await opensearch.embeddings_exist_for_doc(doc_id=doc_id)
        if not has_embeddings:
            phrases = docs.split_text_into_phrases(content)

            async def _embed(text: str) -> list[float]:
                async with embed_sem:
                    return await asyncio.to_thread(docs.text_to_embedding, text)

            for i, phrase in enumerate(phrases):
                embedding = await _embed(phrase)
                chunk_id = f"{doc_id}_{i}"
                async with os_sem:
                    await opensearch.index_embedding_document(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        path=rel_path,
                        chunk_index=i,
                        text=phrase,
                        embedding=embedding,
                        source=docs.source_name,
                    )
                indexed_chunks += 1

        return (indexed_text, indexed_chunks)

    # Limit per-doc concurrency so we don't explode requests on huge doc sets.
    doc_conc = max(1, opensearch_conc // 2)
    doc_sem = asyncio.Semaphore(doc_conc)

    async def _guarded(path: Path) -> tuple[int, int]:
        async with doc_sem:
            return await _index_one_doc(path)

    results = await asyncio.gather(*[_guarded(p) for p in local_md_files], return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            logger.error("OpenSearch ingestion failed: %s", str(r))
            continue
        t, c = r
        search_indexed += t
        vector_chunks_indexed += c

    return {
        "local_docs": len(local_md_files),
        "s3_uploaded": s3_uploaded,
        "search_indexed": search_indexed,
        "vector_chunks_indexed": vector_chunks_indexed,
    }