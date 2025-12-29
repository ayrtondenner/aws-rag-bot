from __future__ import annotations

import asyncio
import json
import os
import time
from http import HTTPStatus
from typing import Any, Optional, cast

import aioboto3
import aiohttp

from app.services.config import OpenSearchConfig
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

    def __init__(self, *, search: OpenSearchService, vector: OpenSearchService) -> None:
        self._search = search
        self._vector = vector

    @staticmethod
    def from_env(*, session: aiohttp.ClientSession) -> "OpenSearchSetupService":
        return OpenSearchSetupService(
            search=OpenSearchService(OpenSearchConfig.from_env_search(), session=session),
            vector=OpenSearchService(OpenSearchConfig.from_env_vector(), session=session),
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

        await self._setup_mapping(
            collection_name=search_collection or "__search__",
            index_name=search_index,
            expected_mapping=self._search_mapping(),
            index_settings=None,
        )

        await self._setup_mapping(
            collection_name=vector_collection or "__vector__",
            index_name=vector_index,
            expected_mapping=self._vector_mapping(dimension=dimension),
            index_settings={"index.knn": True},
        )

    # -----------------
    # Private helpers
    # -----------------

    def _service_for_collection_name(self, *, collection_name: str) -> OpenSearchService:
        configured_search = (os.getenv("OPENSEARCH_SEARCH_COLLECTION_NAME") or "").strip()
        configured_vector = (os.getenv("OPENSEARCH_VECTOR_COLLECTION_NAME") or "").strip()

        if collection_name in ("__search__", configured_search):
            return self._search
        if collection_name in ("__vector__", configured_vector):
            return self._vector

        raise ValueError(
            "Unknown collection_name. Expected OPENSEARCH_SEARCH_COLLECTION_NAME/OPENSEARCH_VECTOR_COLLECTION_NAME "
            f"(got {collection_name!r})."
        )

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

    async def _index_exists_in_collection(self, *, collection_name: str, index_name: str) -> bool:
        svc = self._service_for_collection_name(collection_name=collection_name)
        return await svc.index_exists(index_name=index_name)

    async def _setup_index(
        self,
        *,
        collection_name: str,
        index_name: str,
        index_settings: Optional[dict[str, Any]] = None,
    ) -> None:
        if await self._index_exists_in_collection(collection_name=collection_name, index_name=index_name):
            return

        svc = self._service_for_collection_name(collection_name=collection_name)
        # Create index with an empty mapping; mapping is applied separately via _setup_mapping.
        await svc.create_index_and_mapping(index_name=index_name, mapping={"properties": {}}, settings=index_settings)

    async def _mapping_exists_in_index(
        self,
        *,
        collection_name: str,
        index_name: str,
        expected_mapping: dict[str, object],
    ) -> bool:
        svc = self._service_for_collection_name(collection_name=collection_name)

        status, payload = await svc._signed_request(method="GET", path=f"/{index_name}/_mapping")
        if status == HTTPStatus.NOT_FOUND:
            return False
        if status != HTTPStatus.OK:
            try:
                details = payload.decode("utf-8") if payload else ""
            except Exception:
                details = ""
            raise OpenSearchServiceError(
                f"Unexpected OpenSearch response getting mapping (index={index_name}) HTTP {status} {details}".strip()
            )

        try:
            parsed = json.loads(payload.decode("utf-8")) if payload else {}
        except Exception:
            parsed = {}

        existing_props = self._extract_properties_from_mapping_response(index_name=index_name, mapping_resp=parsed)
        expected_props = (expected_mapping.get("properties") if isinstance(expected_mapping, dict) else None) or {}
        if not isinstance(expected_props, dict):
            expected_props = {}

        return self._properties_cover_expected(existing_props=existing_props, expected_props=expected_props)

    @staticmethod
    def _extract_properties_from_mapping_response(*, index_name: str, mapping_resp: dict[str, Any]) -> dict[str, Any]:
        # Typical response:
        # {"my-index": {"mappings": {"properties": {...}}}}
        node = mapping_resp.get(index_name)
        if not isinstance(node, dict):
            # Sometimes index name may not match (aliases). Fall back to first key.
            if mapping_resp and len(mapping_resp) == 1:
                node = next(iter(mapping_resp.values()))
            else:
                node = None

        if not isinstance(node, dict):
            return {}
        mappings = node.get("mappings")
        if not isinstance(mappings, dict):
            return {}
        props = mappings.get("properties")
        return props if isinstance(props, dict) else {}

    @staticmethod
    def _properties_cover_expected(*, existing_props: dict[str, Any], expected_props: dict[str, Any]) -> bool:
        for field, expected_spec in expected_props.items():
            if field not in existing_props:
                return False
            if not isinstance(expected_spec, dict):
                continue

            existing_spec = existing_props.get(field)
            if not isinstance(existing_spec, dict):
                return False

            expected_type = expected_spec.get("type")
            if expected_type and existing_spec.get("type") != expected_type:
                return False

            if expected_type == "knn_vector":
                expected_dim_raw = expected_spec.get("dimension")
                existing_dim_raw = existing_spec.get("dimension")
                if expected_dim_raw is None or existing_dim_raw is None:
                    return False

                try:
                    expected_dim = int(expected_dim_raw)
                    existing_dim = int(existing_dim_raw)
                except Exception:
                    return False
                if expected_dim != existing_dim:
                    return False

        return True

    async def _setup_mapping(
        self,
        *,
        collection_name: str,
        index_name: str,
        expected_mapping: dict[str, object],
        index_settings: Optional[dict[str, Any]] = None,
    ) -> None:
        await self._setup_index(collection_name=collection_name, index_name=index_name, index_settings=index_settings)

        if await self._mapping_exists_in_index(
            collection_name=collection_name,
            index_name=index_name,
            expected_mapping=expected_mapping,
        ):
            return

        svc = self._service_for_collection_name(collection_name=collection_name)
        status, payload = await svc._signed_request(
            method="PUT",
            path=f"/{index_name}/_mapping",
            body=json.dumps(expected_mapping).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        if status in (HTTPStatus.OK, HTTPStatus.CREATED):
            return

        try:
            details = payload.decode("utf-8") if payload else ""
        except Exception:
            details = ""

        raise OpenSearchServiceError(
            f"Failed to put OpenSearch mapping (index={index_name}) HTTP {status} {details}".strip()
        )
