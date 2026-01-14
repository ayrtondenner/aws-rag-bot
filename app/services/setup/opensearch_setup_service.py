from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional, cast

import aioboto3
import aiohttp

from app.services.config import OpenSearchConfig
from app.services.document_service import DocumentService
from app.services.opensearch_service import OpenSearchService, OpenSearchServiceError


class OpenSearchCollectionError(OpenSearchServiceError):
    pass


class OpenSearchSetupService:
    """Provisioning helper for OpenSearch Serverless + OpenSearch data plane.

    This class is intentionally small and conservative. It uses:
    - OpenSearch Serverless *control-plane* API calls (via botocore) to check/create collections.
    - OpenSearch *data-plane* HTTP calls (via OpenSearchService) to check/create indexes and mappings.

    Notes:
    - Collection creation may require additional security/network/encryption policies in AWS.
    - Index/mapping setup is scoped to a collection endpoint (data plane).
    """

    _DEFAULT_COLLECTION_WAIT_SECONDS: float = 120.0
    _COLLECTION_POLL_INTERVAL_SECONDS: float = 2.0

    def __init__(self, *, opensearch: OpenSearchService) -> None:
        self._opensearch = opensearch

    @staticmethod
    def from_env(*, session: aiohttp.ClientSession) -> "OpenSearchSetupService":
        return OpenSearchSetupService(
            opensearch=OpenSearchService(
                search_config=OpenSearchConfig.from_env_search(),
                vector_config=OpenSearchConfig.from_env_vector(),
                session=session,
                documents=DocumentService(),
            )
        )

    @staticmethod
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

    @staticmethod
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

    async def setup_opensearch_environment(self) -> None:
        """Public entry point: ensure collections, indexes, and mappings exist.

        Uses the following environment variables:
        - OPENSEARCH_SEARCH_COLLECTION_NAME
        - OPENSEARCH_VECTOR_COLLECTION_NAME
        - OPENSEARCH_SEARCH_INDEX_NAME
        - OPENSEARCH_VECTOR_INDEX_NAME
        - BEDROCK_EMBEDDING_DIM
        """

        search_collection = os.getenv("OPENSEARCH_SEARCH_COLLECTION_NAME", "").strip()
        vector_collection = os.getenv("OPENSEARCH_VECTOR_COLLECTION_NAME", "").strip()

        search_index = os.getenv("OPENSEARCH_SEARCH_INDEX_NAME", "").strip() or "sagemaker-docs-search-index"
        vector_index = os.getenv("OPENSEARCH_VECTOR_INDEX_NAME", "").strip() or "sagemaker-docs-vectors-index"

        dim_raw = os.getenv("BEDROCK_EMBEDDING_DIM", "1024")
        try:
            dimension = int(dim_raw)
        except ValueError:
            dimension = 1024
        if dimension <= 0:
            dimension = 1024

        if search_collection or vector_collection:
            region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
            if not region_name:
                raise ValueError("Missing AWS_REGION/AWS_DEFAULT_REGION for collection lookup/creation")

            session: Any = aioboto3.Session()
            client_cm = session.client("opensearchserverless", region_name=region_name)
            async with cast(Any, client_cm) as client:
                if search_collection:
                    await self._setup_collection(
                        client=client,
                        collection_name=search_collection,
                        collection_type="SEARCH",
                    )
                if vector_collection:
                    await self._setup_collection(
                        client=client,
                        collection_name=vector_collection,
                        collection_type="VECTORSEARCH",
                    )

        await self._ensure_index_and_mapping(
            target="search",
            index_name=search_index,
            expected_mapping=self._search_mapping(),
            settings=None,
        )

        await self._ensure_index_and_mapping(
            target="vector",
            index_name=vector_index,
            expected_mapping=self._vector_mapping(dimension=dimension),
            settings={"index.knn": True},
        )

    # -----------------
    # Private helpers
    # -----------------

    async def _collection_exists(self, *, client: Any, collection_name: str) -> bool:
        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name must be provided")

        try:
            resp = await client.batch_get_collection(names=[collection_name])
            details = resp.get("collectionDetails") or []
            return len(details) > 0
        except Exception as exc:
            raise OpenSearchCollectionError(f"Failed checking collection exists: {collection_name}") from exc

    async def _setup_collection(self, *, client: Any, collection_name: str, collection_type: str) -> None:
        if await self._collection_exists(client=client, collection_name=collection_name):
            return

        try:
            await client.create_collection(name=collection_name, type=collection_type)
        except Exception as exc:
            raise OpenSearchCollectionError(
                "Failed creating OpenSearch Serverless collection. "
                "This often means required security/network/encryption policies are missing. "
                f"(collection={collection_name}, type={collection_type})"
            ) from exc

        deadline = time.monotonic() + self._DEFAULT_COLLECTION_WAIT_SECONDS
        while time.monotonic() < deadline:
            try:
                resp = await client.batch_get_collection(names=[collection_name])
                details = (resp.get("collectionDetails") or [])
                if not details:
                    await asyncio.sleep(self._COLLECTION_POLL_INTERVAL_SECONDS)
                    continue

                status = (details[0].get("status") or "").upper()
                if status in {"ACTIVE"}:
                    return
                if status in {"FAILED", "DELETING"}:
                    raise OpenSearchCollectionError(
                        f"Collection entered unexpected status after creation: {collection_name} (status={status})"
                    )
            except OpenSearchCollectionError:
                raise
            except Exception:
                # Be tolerant of eventual consistency / transient permission issues.
                pass

            await asyncio.sleep(self._COLLECTION_POLL_INTERVAL_SECONDS)

        raise OpenSearchCollectionError(f"Timed out waiting for collection to become ACTIVE: {collection_name}")

    async def _ensure_index_and_mapping(
        self,
        *,
        target: str,
        index_name: str,
        expected_mapping: dict[str, Any],
        settings: Optional[dict[str, Any]],
    ) -> None:
        target_typed = cast(Any, target)
        if not await self._opensearch.index_exists(index_name=index_name, target=target_typed):
            await self._opensearch.create_index(
                index_name=index_name,
                mapping=expected_mapping,
                settings=settings,
                target=target_typed,
            )
            return

        if await self._opensearch.mapping_exists(
            index_name=index_name,
            expected_mapping=expected_mapping,
            target=target_typed,
        ):
            return

        await self._opensearch.put_mapping(index_name=index_name, mapping=expected_mapping, target=target_typed)
