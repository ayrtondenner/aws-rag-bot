from __future__ import annotations

import os
import logging
import mimetypes
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import aioboto3

from typing import Optional

from app.models.s3 import FileItem


logger = logging.getLogger(__name__)


class S3ServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class S3Config:
    bucket_name: str
    region_name: Optional[str] = None
    endpoint_url: Optional[str] = None

    @staticmethod
    def from_env() -> "S3Config":
        bucket_name = os.getenv("S3_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("Missing required environment variable: S3_BUCKET_NAME")

        region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        endpoint_url = os.getenv("S3_ENDPOINT_URL")

        return S3Config(bucket_name=bucket_name, region_name=region_name, endpoint_url=endpoint_url)


class S3Service:
    def __init__(self, config: S3Config) -> None:
        self._config = config
        self._session = aioboto3.Session()

    def _client(self) -> Any:
        return self._session.client(
            "s3",
            region_name=self._config.region_name,
            endpoint_url=self._config.endpoint_url,
        )

    async def list_files(self, *, prefix: Optional[str] = None, max_keys: int = 1000) -> list[FileItem]:
        try:
            kwargs: dict[str, Any] = {"Bucket": self._config.bucket_name, "MaxKeys": max_keys}
            if prefix:
                kwargs["Prefix"] = prefix

            s3_client: Any = self._client()
            async with s3_client as s3:
                response = await s3.list_objects_v2(**kwargs)

            objects = response.get("Contents", [])
            return [FileItem.from_s3_object(o) for o in objects]
        except Exception as exc:
            logger.exception("S3 list_files failed")
            raise S3ServiceError("Failed to list files from S3") from exc

    async def upload_local_file(self, *, path: Path, key: str, content_type: Optional[str] = None) -> str:
        """Upload a local file to S3.

        Args:
            path: Local file path.
            key: Destination S3 object key.
            content_type: Optional content type override.

        Returns:
            The uploaded object key.
        """

        try:
            if not key:
                raise ValueError("'key' must be provided")
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(str(path))

            body = path.read_bytes()
            effective_content_type = content_type
            if effective_content_type is None:
                guessed, _ = mimetypes.guess_type(str(path))
                effective_content_type = guessed

            extra_args: dict[str, Any] = {}
            if effective_content_type:
                extra_args["ContentType"] = effective_content_type

            s3_client: Any = self._client()
            async with s3_client as s3:
                await s3.put_object(
                    Bucket=self._config.bucket_name,
                    Key=key,
                    Body=body,
                    **extra_args,
                )

            return key
        except Exception as exc:
            logger.exception("S3 upload_path failed")
            raise S3ServiceError(f"Failed to upload local file to S3 (key={key})") from exc

    async def delete_file(self, *, key: str) -> None:
        try:
            if not key:
                raise ValueError("'key' must be provided")

            s3_client: Any = self._client()
            async with s3_client as s3:
                await s3.delete_object(Bucket=self._config.bucket_name, Key=key)
        except Exception as exc:
            logger.exception("S3 delete_file failed")
            raise S3ServiceError("Failed to delete file from S3") from exc
