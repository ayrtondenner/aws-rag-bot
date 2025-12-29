from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.document_text_service import DocumentTextService
from app.services.opensearch_service import OpenSearchService


logger = logging.getLogger(__name__)


class SageMakerDocsOpenSearchIndexServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class SageMakerDocsOpenSearchIndexConfig:
    docs_dir: Path
    search_index_name: str
    vector_index_name: str
    source_name: str = "sagemaker-docs"

    @staticmethod
    def from_env(*, docs_dir: Path) -> "SageMakerDocsOpenSearchIndexConfig":
        search_index_name = os.getenv("OPENSEARCH_SEARCH_INDEX_NAME", "sagemaker-docs")
        vector_index_name = os.getenv("OPENSEARCH_VECTOR_INDEX_NAME", "sagemaker-docs-vectors")
        source_name = os.getenv("OPENSEARCH_DOCS_SOURCE_NAME", "sagemaker-docs")

        return SageMakerDocsOpenSearchIndexConfig(
            docs_dir=docs_dir,
            search_index_name=search_index_name,
            vector_index_name=vector_index_name,
            source_name=source_name,
        )


class SageMakerDocsOpenSearchIndexService:
    def __init__(
        self,
        *,
        search: OpenSearchService,
        vector: OpenSearchService,
        text: DocumentTextService,
        config: SageMakerDocsOpenSearchIndexConfig,
    ) -> None:
        self._search = search
        self._vector = vector
        self._text = text
        self._config = config

    @property
    def search_index_name(self) -> str:
        return self._config.search_index_name

    @property
    def vector_index_name(self) -> str:
        return self._config.vector_index_name

    @staticmethod
    def _doc_id_from_rel_path(rel_path: str) -> str:
        # Stable, URL-safe id (hex) derived from relative path.
        return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()

    def _search_mapping(self) -> dict[str, Any]:
        return {
            "properties": {
                "doc_id": {"type": "keyword"},
                "path": {"type": "keyword"},
                "title": {"type": "text"},
                "content": {"type": "text"},
                "source": {"type": "keyword"},
            }
        }

    def _vector_mapping(self, *, dimension: int) -> dict[str, Any]:
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

    def ensure_indexes(self) -> None:
        """Create indexes if they don't exist (search + vector)."""

        search_index = self._config.search_index_name
        vector_index = self._config.vector_index_name

        if not self._search.index_exists(index_name=search_index):
            self._search.create_index_and_mapping(index_name=search_index, mapping=self._search_mapping())

        dimension = int(os.getenv("BEDROCK_EMBEDDING_DIM", 1024))
        if dimension <= 0:
            dimension = 1024
        if not self._vector.index_exists(index_name=vector_index):
            # Many OpenSearch setups require kNN to be enabled via settings.
            # For Serverless vector collections this may be ignored or accepted.
            self._vector.create_index_and_mapping(
                index_name=vector_index,
                mapping=self._vector_mapping(dimension=dimension),
                settings={"index.knn": True},
            )

    def index_local_docs(self) -> tuple[int, int]:
        """Index all local SageMaker docs.

        Stores full documents in the search collection index, and chunk+embedding
        documents in the vector collection index.

        Returns:
            (documents_indexed, chunks_indexed)
        """

        docs_dir = self._config.docs_dir
        if not docs_dir.exists() or not docs_dir.is_dir():
            raise SageMakerDocsOpenSearchIndexServiceError(f"Docs directory not found: {docs_dir}")

        self.ensure_indexes()

        search_index = self._config.search_index_name
        vector_index = self._config.vector_index_name

        documents_indexed = 0
        chunks_indexed = 0

        md_files = sorted(p for p in docs_dir.rglob("*.md") if p.is_file())
        for path in md_files:
            rel_path = path.relative_to(docs_dir).as_posix()
            doc_id = self._doc_id_from_rel_path(rel_path)
            title = path.stem

            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="utf-8", errors="replace")

            self._search.index_document(
                index_name=search_index,
                document_id=doc_id,
                document={
                    "doc_id": doc_id,
                    "path": rel_path,
                    "title": title,
                    "content": content,
                    "source": self._config.source_name,
                },
            )
            documents_indexed += 1

            chunks = self._text.split_text_into_chunks(content)
            for i, chunk_text in enumerate(chunks):
                embedding = self._text.text_to_embedding(chunk_text)
                chunk_id = f"{doc_id}_{i}"

                self._vector.index_document(
                    index_name=vector_index,
                    document_id=chunk_id,
                    document={
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "path": rel_path,
                        "chunk_index": i,
                        "text": chunk_text,
                        "embedding": embedding,
                        "source": self._config.source_name,
                    },
                )
                chunks_indexed += 1

        logger.info(
            "OpenSearch indexing complete: docs=%d chunks=%d (search_index=%s vector_index=%s)",
            documents_indexed,
            chunks_indexed,
            search_index,
            vector_index,
        )

        return (documents_indexed, chunks_indexed)
