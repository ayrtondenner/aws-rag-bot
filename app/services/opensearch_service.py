from __future__ import annotations

import json
import logging
from http import HTTPStatus
from typing import Any, Optional
from urllib.parse import quote

import aiohttp
import botocore.session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from app.services.config import OpenSearchConfig

logger = logging.getLogger(__name__)


class OpenSearchServiceError(RuntimeError):
    pass


class OpenSearchIndexAlreadyExistsError(OpenSearchServiceError):
    pass

class OpenSearchService:
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
