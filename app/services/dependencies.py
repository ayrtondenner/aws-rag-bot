from __future__ import annotations

from pathlib import Path

import aiohttp
from fastapi import Request

from app.services.config import OpenSearchConfig, S3Config
from app.services.config.sagemaker_docs import (
    SageMakerDocsConfig,
    SageMakerDocsSyncConfig,
)
from app.services.document_text_service import DocumentTextService
from app.services.opensearch_service import OpenSearchService
from app.services.s3_service import S3Service
from app.services.sagemaker_docs_service import (
    SageMakerDocsService,
    SageMakerDocsSyncService,
)
from app.services.setup.opensearch_setup_service import OpenSearchSetupService
from app.services.setup.s3_setup_service import S3SetupService


def get_s3_service() -> S3Service:
    """FastAPI dependency provider for an S3Service instance."""

    return S3Service(S3Config.from_env())


def get_sagemaker_docs_sync_service() -> SageMakerDocsSyncService:
    """Dependency provider for syncing local SageMaker docs to S3."""

    project_root = Path(__file__).resolve().parents[2]
    docs_dir = project_root / "sagemaker-docs"

    return SageMakerDocsSyncService(
        s3=get_s3_service(),
        config=SageMakerDocsSyncConfig(docs_dir=docs_dir),
    )


def get_document_text_service() -> DocumentTextService:
    """Dependency provider for text/document processing helpers."""

    return DocumentTextService()


def get_sagemaker_docs_service() -> SageMakerDocsService:
    """Dependency provider for generic local SageMaker docs helpers."""

    project_root = Path(__file__).resolve().parents[2]
    docs_dir = project_root / "sagemaker-docs"
    return SageMakerDocsService(SageMakerDocsConfig.from_env(docs_dir=docs_dir))

def get_http_session(request: Request) -> aiohttp.ClientSession:
    session = getattr(request.app.state, "http_session", None)
    if session is None:
        raise RuntimeError("HTTP session not initialized (app.state.http_session)")
    if not isinstance(session, aiohttp.ClientSession):
        raise RuntimeError("Unexpected http_session type")
    return session


def get_opensearch_search_service(request: Request) -> OpenSearchService:
    return OpenSearchService(OpenSearchConfig.from_env_search(), session=get_http_session(request))


def get_opensearch_vector_service(request: Request) -> OpenSearchService:
    return OpenSearchService(OpenSearchConfig.from_env_vector(), session=get_http_session(request))


def get_s3_setup_service() -> S3SetupService:
    return S3SetupService(s3=get_s3_service())


def get_opensearch_setup_service(request: Request) -> OpenSearchSetupService:
    return OpenSearchSetupService(
        search=get_opensearch_search_service(request),
        vector=get_opensearch_vector_service(request),
    )