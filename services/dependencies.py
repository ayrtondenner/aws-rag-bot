from __future__ import annotations

from services.s3_service import S3Config, S3Service


def get_s3_service() -> S3Service:
    """FastAPI dependency provider for an S3Service instance."""

    return S3Service(S3Config.from_env())
