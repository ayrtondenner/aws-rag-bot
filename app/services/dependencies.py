from __future__ import annotations

from pathlib import Path

from app.services.document_text_service import DocumentTextService
from app.services.opensearch_service import OpenSearchConfig, OpenSearchService
from app.services.s3_service import S3Config, S3Service
from app.services.sagemaker_docs_service import (
    SageMakerDocsConfig,
    SageMakerDocsService,
    SageMakerDocsSyncConfig,
    SageMakerDocsSyncService,
)


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


def get_opensearch_search_service() -> OpenSearchService:
    return OpenSearchService(OpenSearchConfig.from_env_search())


def get_opensearch_vector_service() -> OpenSearchService:
    return OpenSearchService(OpenSearchConfig.from_env_vector())
