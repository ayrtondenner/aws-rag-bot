from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import aioboto3
from fastapi import UploadFile


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

    async def list_files(self, *, prefix: Optional[str] = None, max_keys: int = 1000) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"Bucket": self._config.bucket_name, "MaxKeys": max_keys}
        if prefix:
            kwargs["Prefix"] = prefix

        async with self._session.client(
            "s3",
            region_name=self._config.region_name,
            endpoint_url=self._config.endpoint_url,
        ) as s3:
            response = await s3.list_objects_v2(**kwargs)

        return response.get("Contents", [])

    async def upload_file(self, *, file: UploadFile, key: Optional[str] = None) -> str:
        object_key = key or file.filename
        if not object_key:
            raise ValueError("Either 'key' must be provided or the uploaded file must have a filename.")

        body = await file.read()
        extra_args: dict[str, Any] = {}
        if file.content_type:
            extra_args["ContentType"] = file.content_type

        async with self._session.client(
            "s3",
            region_name=self._config.region_name,
            endpoint_url=self._config.endpoint_url,
        ) as s3:
            await s3.put_object(
                Bucket=self._config.bucket_name,
                Key=object_key,
                Body=body,
                **extra_args,
            )

        return object_key

    async def delete_file(self, *, key: str) -> None:
        if not key:
            raise ValueError("'key' must be provided")

        async with self._session.client(
            "s3",
            region_name=self._config.region_name,
            endpoint_url=self._config.endpoint_url,
        ) as s3:
            await s3.delete_object(Bucket=self._config.bucket_name, Key=key)
