from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from http import HTTPStatus
from typing import Any, Optional
from urllib.parse import quote

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
    """Minimal OpenSearch data-plane service using AWS SigV4 signing.

    This avoids extra dependencies (like opensearch-py) and uses the AWS credentials
    already configured for the app (env vars, profiles/SSO, instance role, etc.).
    """

    def __init__(self, config: OpenSearchConfig) -> None:
        self._config = config

    def _signed_request(
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

        req = urllib.request.Request(
            url=url,
            data=body,
            method=method.upper(),
            headers=dict(prepared.headers),
        )

        try:
            with urllib.request.urlopen(req, timeout=self._config.timeout_seconds) as resp:
                return (resp.status, resp.read() or b"")
        except urllib.error.HTTPError as http_err:
            # HTTPError is also a valid response; read body for context.
            try:
                payload = http_err.read() or b""
            except Exception:
                payload = b""
            return (http_err.code, payload)
        except Exception as exc:
            logger.exception("OpenSearch request failed (method=%s path=%s)", method, path)
            raise OpenSearchServiceError("OpenSearch request failed") from exc

    @staticmethod
    def _validate_index_name(index_name: str) -> None:
        if not index_name or not index_name.strip():
            raise ValueError("index_name must be provided")

    def index_exists(self, *, index_name: str) -> bool:
        """Return True if the index exists, otherwise False."""

        self._validate_index_name(index_name)

        status, _ = self._signed_request(method="HEAD", path=f"/{index_name}")
        if status == HTTPStatus.OK:
            return True
        if status == HTTPStatus.NOT_FOUND:
            return False

        raise OpenSearchServiceError(f"Unexpected OpenSearch response checking index exists: HTTP {status}")

    def create_index_and_mapping(
        self,
        *,
        index_name: str,
        mapping: dict[str, Any],
        settings: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Create an index with the provided mapping.

        Returns:
            True if the index was created successfully.

        Raises:
            OpenSearchIndexAlreadyExistsError: if an index with the same name already exists.
            OpenSearchServiceError: for unexpected OpenSearch/AWS failures.
        """

        self._validate_index_name(index_name)
        if self.index_exists(index_name=index_name):
            raise OpenSearchIndexAlreadyExistsError(f"Index already exists: {index_name}")

        body: dict[str, Any] = {"mappings": mapping}
        if settings:
            body["settings"] = settings

        status, payload = self._signed_request(
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
                # Typically: {"acknowledged": true, "shards_acknowledged": true, "index": "..."}
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

    def index_document(
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
        status, payload = self._signed_request(
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
