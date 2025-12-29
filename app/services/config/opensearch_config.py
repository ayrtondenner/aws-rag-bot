from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class OpenSearchConfig:
    """Runtime configuration for OpenSearch (data plane) calls.

    `endpoint` should be the domain endpoint including scheme, e.g.
    "https://search-my-domain-abc123.eu-west-1.es.amazonaws.com".
    """

    endpoint: str
    region_name: str
    service_name: str
    _DEFAULT_TIMEOUT_SECONDS: ClassVar[float] = 30.0
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS

    @staticmethod
    def _infer_service_name_from_endpoint(endpoint: str) -> str:
        # OpenSearch Serverless collection endpoints commonly end with: .<region>.aoss.amazonaws.com
        if ".aoss.amazonaws.com" in endpoint:
            return "aoss"
        # Managed OpenSearch domains historically use the SigV4 service id "es"
        return "es"

    @staticmethod
    def from_env_named(
        *,
        endpoint_env: str,
        region_env: str = "OPENSEARCH_REGION",
        service_env: str = "OPENSEARCH_SERVICE_NAME",
        timeout_env: str = "OPENSEARCH_TIMEOUT_SECONDS",
    ) -> "OpenSearchConfig":
        endpoint = os.getenv(endpoint_env)
        if not endpoint:
            raise ValueError(f"Missing required environment variable: {endpoint_env}")

        region_name = os.getenv(region_env) or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        if not region_name:
            raise ValueError(
                f"Missing required environment variable: {region_env} (or AWS_REGION/AWS_DEFAULT_REGION)"
            )

        timeout_raw = os.getenv(timeout_env)
        timeout_seconds = OpenSearchConfig._DEFAULT_TIMEOUT_SECONDS
        if timeout_raw:
            try:
                timeout_seconds = float(timeout_raw)
            except ValueError as exc:
                raise ValueError(f"Invalid {timeout_env}; must be a number") from exc

        cleaned_endpoint = endpoint.rstrip("/")
        service_name = os.getenv(service_env) or OpenSearchConfig._infer_service_name_from_endpoint(cleaned_endpoint)

        return OpenSearchConfig(
            endpoint=cleaned_endpoint,
            region_name=region_name,
            service_name=service_name,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def from_env_search() -> "OpenSearchConfig":
        """Load config for the search collection endpoint."""

        return OpenSearchConfig.from_env_named(endpoint_env="OPENSEARCH_SEARCH_ENDPOINT")

    @staticmethod
    def from_env_vector() -> "OpenSearchConfig":
        """Load config for the vector collection endpoint."""

        return OpenSearchConfig.from_env_named(endpoint_env="OPENSEARCH_VECTOR_ENDPOINT")
