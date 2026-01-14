from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from http import HTTPStatus
from typing import Any, Literal, Optional
from urllib.parse import quote

import aioboto3
import aiohttp
import botocore.session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from app.services.config import OpenSearchConfig
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)


class OpenSearchServiceError(RuntimeError):
    pass


class OpenSearchIndexAlreadyExistsError(OpenSearchServiceError):
    pass


class OpenSearchUnexpectedResponseError(OpenSearchServiceError):
    pass
class _OpenSearchClient:
    """Async OpenSearch data-plane service using aiohttp + AWS SigV4 signing.

    It avoids blocking the event loop by using an async HTTP client.

    Notes:
    - SigV4 signing still uses botocore.
    - AWS credential resolution is performed via botocore; in common setups this is
      cached/in-memory, but some credential providers may still perform I/O.
    """

    def __init__(self, config: OpenSearchConfig, *, session: aiohttp.ClientSession) -> None:
        self._config = config
        self._session = session

    async def _signed_request(
        self,
        *,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> tuple[int, bytes]:
        if not path.startswith("/"):
            path = "/" + path

        url = f"{self._config.endpoint}{path}"

        session = botocore.session.get_session()
        credentials = session.get_credentials()
        if credentials is None:
            raise OpenSearchServiceError("No AWS credentials available for OpenSearch request signing")

        frozen = credentials.get_frozen_credentials()

        effective_headers: dict[str, str] = {"Accept": "application/json"}
        if headers:
            effective_headers.update(headers)

        if body is not None and "Content-Type" not in {k.title(): v for k, v in effective_headers.items()}:
            effective_headers.setdefault("Content-Type", "application/json")

        aws_request = AWSRequest(method=method.upper(), url=url, data=body, headers=effective_headers)
        SigV4Auth(frozen, self._config.service_name, self._config.region_name).add_auth(aws_request)
        prepared = aws_request.prepare()

        try:
            timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
            async with self._session.request(
                method=method.upper(),
                url=url,
                data=body,
                headers=dict(prepared.headers),
                timeout=timeout,
            ) as resp:
                payload = await resp.read()
                return (resp.status, payload or b"")
        except aiohttp.ClientError as exc:
            logger.exception("OpenSearch async request failed (method=%s path=%s)", method, path)
            raise OpenSearchServiceError("OpenSearch request failed") from exc
        except Exception as exc:
            logger.exception("OpenSearch async request failed (method=%s path=%s)", method, path)
            raise OpenSearchServiceError("OpenSearch request failed") from exc

    @staticmethod
    def _validate_index_name(index_name: str) -> None:
        if not index_name or not index_name.strip():
            raise ValueError("index_name must be provided")

    async def index_exists(self, *, index_name: str) -> bool:
        """Return True if the index exists, otherwise False."""

        self._validate_index_name(index_name)

        status, _ = await self._signed_request(method="HEAD", path=f"/{index_name}")
        if status == HTTPStatus.OK:
            return True
        if status == HTTPStatus.NOT_FOUND:
            return False

        raise OpenSearchServiceError(f"Unexpected OpenSearch response checking index exists: HTTP {status}")

    # TODO: readd returns and raises to docstring of the methods
    # TODO: delete this method. Index and mapping should be created separately at setup time.
    async def create_index_and_mapping(
        self,
        *,
        index_name: str,
        mapping: dict[str, Any],
        settings: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Create an index with the provided mapping."""

        self._validate_index_name(index_name)
        if await self.index_exists(index_name=index_name):
            raise OpenSearchIndexAlreadyExistsError(f"Index already exists: {index_name}")

        body: dict[str, Any] = {"mappings": mapping}
        if settings:
            body["settings"] = settings

        status, payload = await self._signed_request(
            method="PUT",
            path=f"/{index_name}",
            body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        if status in (HTTPStatus.OK, HTTPStatus.CREATED):
            if not payload:
                return True
            try:
                parsed = json.loads(payload.decode("utf-8"))
                return bool(parsed.get("acknowledged", True))
            except Exception:
                return True

        try:
            details = payload.decode("utf-8") if payload else ""
        except Exception:
            details = ""

        raise OpenSearchServiceError(
            f"Failed to create OpenSearch index (index={index_name}) HTTP {status} {details}".strip()
        )

    async def index_document(
        self,
        *,
        index_name: str,
        document_id: str,
        document: dict[str, Any],
    ) -> bool:
        """Create or update a document by id (PUT /{index}/_doc/{id})."""

        self._validate_index_name(index_name)
        if not document_id or not document_id.strip():
            raise ValueError("document_id must be provided")

        safe_id = quote(document_id, safe="")
        status, payload = await self._signed_request(
            method="PUT",
            path=f"/{index_name}/_doc/{safe_id}",
            body=json.dumps(document).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        if status in (HTTPStatus.OK, HTTPStatus.CREATED):
            return True

        try:
            details = payload.decode("utf-8") if payload else ""
        except Exception:
            details = ""

        raise OpenSearchServiceError(
            f"Failed to index OpenSearch document (index={index_name}, id={document_id}) HTTP {status} {details}".strip()
        )

    async def document_exists(self, *, index_name: str, document_id: str) -> bool:
        """Return True if a document exists by id (HEAD /{index}/_doc/{id})."""

        self._validate_index_name(index_name)
        if not document_id or not document_id.strip():
            raise ValueError("document_id must be provided")

        safe_id = quote(document_id, safe="")
        status, _ = await self._signed_request(method="HEAD", path=f"/{index_name}/_doc/{safe_id}")
        if status == HTTPStatus.OK:
            return True
        if status == HTTPStatus.NOT_FOUND:
            return False

        raise OpenSearchUnexpectedResponseError(
            f"Unexpected OpenSearch response checking document exists: HTTP {status} (index={index_name})".strip()
        )

    async def search(self, *, index_name: str, query: dict[str, Any]) -> dict[str, Any]:
        """Execute a search query (POST /{index}/_search) and return parsed JSON."""

        self._validate_index_name(index_name)

        status, payload = await self._signed_request(
            method="POST",
            path=f"/{index_name}/_search",
            body=json.dumps(query).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        if status != HTTPStatus.OK:
            try:
                details = payload.decode("utf-8") if payload else ""
            except Exception:
                details = ""
            raise OpenSearchUnexpectedResponseError(
                f"Unexpected OpenSearch response searching (index={index_name}) HTTP {status} {details}".strip()
            )

        try:
            return json.loads(payload.decode("utf-8")) if payload else {}
        except Exception as exc:
            raise OpenSearchUnexpectedResponseError("Failed to parse OpenSearch search response") from exc

    async def count_by_term(self, *, index_name: str, field: str, value: str) -> int:
        """Return hit count for a simple term query (size=0)"""

        if not field or not field.strip():
            raise ValueError("field must be provided")
        if value is None:
            raise ValueError("value must be provided")

        resp = await self.search(
            index_name=index_name,
            query={
                "size": 0,
                "track_total_hits": True,
                "query": {"term": {field: {"value": value}}},
            },
        )

        hits = resp.get("hits")
        if not isinstance(hits, dict):
            return 0

        total = hits.get("total")
        # OpenSearch may return either {"value": n, "relation": "eq"} or a raw int depending on version.
        if isinstance(total, int):
            return total
        if isinstance(total, dict):
            value_raw = total.get("value")
            try:
                if value_raw is None:
                    return 0
                return int(value_raw)
            except Exception:
                return 0

        return 0


class OpenSearchService:
    """High-level OpenSearch service (control-plane checks + data-plane indexing/search).

    This wraps two signed data-plane clients:
    - one for the *text/search* collection
    - one for the *vector/embeddings* collection
    """

    _DEFAULT_SEARCH_INDEX = "sagemaker-docs-search-index"
    _DEFAULT_VECTOR_INDEX = "sagemaker-docs-vectors-index"

    def __init__(
        self,
        *,
        search_config: OpenSearchConfig,
        vector_config: OpenSearchConfig,
        session: aiohttp.ClientSession,
        documents: DocumentService,
        search_index: Optional[str] = None,
        vector_index: Optional[str] = None,
    ) -> None:
        self._search = _OpenSearchClient(search_config, session=session)
        self._vector = _OpenSearchClient(vector_config, session=session)
        self._documents = documents

        self._search_index = (search_index or os.getenv("OPENSEARCH_SEARCH_INDEX_NAME") or "").strip() or self._DEFAULT_SEARCH_INDEX
        self._vector_index = (vector_index or os.getenv("OPENSEARCH_VECTOR_INDEX_NAME") or "").strip() or self._DEFAULT_VECTOR_INDEX

    # -----------------
    # Control-plane
    # -----------------

    async def collection_exists(self, *, collection_name: str) -> bool:
        """Return True if an OpenSearch Serverless collection exists.

        This calls the `opensearchserverless` control-plane API.
        """

        name = (collection_name or "").strip()
        if not name:
            raise ValueError("collection_name must be provided")

        region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        if not region_name:
            raise ValueError("Missing AWS_REGION/AWS_DEFAULT_REGION")

        session = aioboto3.Session()
        async with session.client("opensearchserverless", region_name=region_name) as client:
            resp = await client.batch_get_collection(names=[name])
            details = resp.get("collectionDetails") or []
            return len(details) > 0

    # -----------------
    # Data-plane checks
    # -----------------

    def _client(self, *, target: Literal["search", "vector"]) -> _OpenSearchClient:
        return self._search if target == "search" else self._vector

    async def index_exists(self, *, index_name: str, target: Literal["search", "vector"] = "search") -> bool:
        return await self._client(target=target).index_exists(index_name=index_name)

    async def mapping_exists(
        self,
        *,
        index_name: str,
        expected_mapping: dict[str, object],
        target: Literal["search", "vector"] = "search",
    ) -> bool:
        """Return True if the index mapping covers the expected fields/types."""

        client = self._client(target=target)
        status, payload = await client._signed_request(method="GET", path=f"/{index_name}/_mapping")
        if status == HTTPStatus.NOT_FOUND:
            return False
        if status != HTTPStatus.OK:
            try:
                details = payload.decode("utf-8") if payload else ""
            except Exception:
                details = ""
            raise OpenSearchUnexpectedResponseError(
                f"Unexpected OpenSearch response getting mapping (index={index_name}) HTTP {status} {details}".strip()
            )

        try:
            parsed: dict[str, Any] = json.loads(payload.decode("utf-8")) if payload else {}
        except Exception:
            parsed = {}

        existing_props = self._extract_properties_from_mapping_response(index_name=index_name, mapping_resp=parsed)
        expected_props = (expected_mapping.get("properties") if isinstance(expected_mapping, dict) else None) or {}
        if not isinstance(expected_props, dict):
            expected_props = {}
        return self._properties_cover_expected(existing_props=existing_props, expected_props=expected_props)

    @staticmethod
    def _extract_properties_from_mapping_response(*, index_name: str, mapping_resp: dict[str, Any]) -> dict[str, Any]:
        node = mapping_resp.get(index_name)
        if not isinstance(node, dict):
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

    # -----------------
    # Data-plane writes
    # -----------------

    async def create_index(
        self,
        *,
        index_name: str,
        mapping: dict[str, Any],
        settings: Optional[dict[str, Any]] = None,
        target: Literal["search", "vector"] = "search",
    ) -> bool:
        """Create an index with mapping/settings in the selected collection."""

        return await self._client(target=target).create_index_and_mapping(
            index_name=index_name,
            mapping=mapping,
            settings=settings,
        )

    async def put_mapping(
        self,
        *,
        index_name: str,
        mapping: dict[str, Any],
        target: Literal["search", "vector"] = "search",
    ) -> None:
        """Apply (merge) a mapping onto an existing index."""

        client = self._client(target=target)
        status, payload = await client._signed_request(
            method="PUT",
            path=f"/{index_name}/_mapping",
            body=json.dumps(mapping).encode("utf-8"),
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

    async def index_text_document(
        self,
        *,
        doc_id: str,
        path: str,
        title: str,
        content: str,
        source: Optional[str] = None,
        index_name: Optional[str] = None,
    ) -> bool:
        idx = (index_name or self._search_index).strip()
        return await self._search.index_document(
            index_name=idx,
            document_id=doc_id,
            document={
                "doc_id": doc_id,
                "path": path,
                "title": title,
                "content": content,
                "source": source or "sagemaker-docs",
            },
        )

    async def text_document_exists(self, *, doc_id: str, index_name: Optional[str] = None) -> bool:
        idx = (index_name or self._search_index).strip()
        return await self._search.document_exists(index_name=idx, document_id=doc_id)

    async def index_embedding_document(
        self,
        *,
        chunk_id: str,
        doc_id: str,
        path: str,
        chunk_index: int,
        text: str,
        embedding: list[float],
        source: Optional[str] = None,
        index_name: Optional[str] = None,
    ) -> bool:
        idx = (index_name or self._vector_index).strip()
        return await self._vector.index_document(
            index_name=idx,
            document_id=chunk_id,
            document={
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "path": path,
                "chunk_index": int(chunk_index),
                "text": text,
                "embedding": embedding,
                "source": source or "sagemaker-docs",
            },
        )

    async def embeddings_exist_for_doc(self, *, doc_id: str, index_name: Optional[str] = None) -> bool:
        idx = (index_name or self._vector_index).strip()
        count = await self._vector.count_by_term(index_name=idx, field="doc_id", value=doc_id)
        return count > 0

    # -----------------
    # Hybrid search
    # -----------------

    @staticmethod
    def _strip_em_tags(text: str) -> str:
        return re.sub(r"</?em>", "", text or "")

    def _lexical_query(self, *, query: str, k: int) -> dict[str, Any]:
        return {
            "size": k,
            "query": {"multi_match": {"query": query, "fields": ["title^2", "content"]}},
            "highlight": {"fields": {"content": {"fragment_size": 180, "number_of_fragments": 2}}},
            "_source": ["path", "title"],
        }

    def _vector_query(self, *, embedding: list[float], k: int) -> dict[str, Any]:
        return {
            "size": k,
            "query": {"knn": {"embedding": {"vector": embedding, "k": k}}},
            "_source": ["path", "text"],
        }

    async def hybrid_search(
        self,
        *,
        query: str,
        k_text: int = 5,
        k_vector: int = 5,
        search_index: Optional[str] = None,
        vector_index: Optional[str] = None,
    ) -> tuple[list[str], list[str]]:
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return ([], [])

        k_text = max(1, min(50, int(k_text)))
        k_vector = max(1, min(50, int(k_vector)))

        embedding = await asyncio.to_thread(self._documents.text_to_embedding, cleaned_query)

        idx_search = (search_index or self._search_index).strip()
        idx_vector = (vector_index or self._vector_index).strip()

        lexical_resp, vector_resp = await asyncio.gather(
            self._search.search(index_name=idx_search, query=self._lexical_query(query=cleaned_query, k=k_text)),
            self._vector.search(index_name=idx_vector, query=self._vector_query(embedding=embedding, k=k_vector)),
        )

        phrases: list[str] = []
        documents: list[str] = []
        seen_phrases: set[str] = set()
        seen_docs: set[str] = set()

        def _add_phrase(*, phrase: str, doc: Optional[str]) -> None:
            p = (phrase or "").strip()
            if not p:
                return
            if p not in seen_phrases:
                seen_phrases.add(p)
                phrases.append(p)
            if doc:
                d = doc.strip()
                if d and d not in seen_docs:
                    seen_docs.add(d)
                    documents.append(d)

        for hit in (vector_resp.get("hits", {}) or {}).get("hits", []) or []:
            src = hit.get("_source") or {}
            _add_phrase(phrase=str(src.get("text") or ""), doc=str(src.get("path") or ""))

        for hit in (lexical_resp.get("hits", {}) or {}).get("hits", []) or []:
            src = hit.get("_source") or {}
            path = str(src.get("path") or "")
            highlights = (hit.get("highlight") or {}).get("content") or []
            if isinstance(highlights, list) and highlights:
                for frag in highlights:
                    _add_phrase(phrase=self._strip_em_tags(str(frag)), doc=path)
            else:
                title = str(src.get("title") or "")
                if title:
                    _add_phrase(phrase=title, doc=path)

        return (phrases, documents)
