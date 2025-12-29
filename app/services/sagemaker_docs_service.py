from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from app.services.s3_service import S3Service, S3ServiceError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SageMakerDocsSyncConfig:
    """Internal configuration for SageMaker docs startup sync.

    This is service wiring/runtime configuration (paths, prefix, concurrency), not an API
    request/response schema. That's why it's a lightweight dataclass kept with the
    service instead of a Pydantic model under app/models.
    """

    docs_dir: Path
    s3_prefix: str = "sagemaker-docs/"
    concurrency: int = 10


class SageMakerDocsSyncService:
    def __init__(self, *, s3: S3Service, config: SageMakerDocsSyncConfig) -> None:
        self._s3 = s3
        self._config = config

    @staticmethod
    def _planned_upload_item(*, docs_dir: Path, prefix: str, existing_keys: set[str], path: Path) -> Optional[tuple[Path, str]]:
        rel_key = path.relative_to(docs_dir).as_posix()
        s3_key = f"{prefix}{rel_key}" if prefix else rel_key
        if s3_key in existing_keys:
            return None
        return (path, s3_key)

    async def startup_check_and_sync_docs(self) -> None:
        """Startup check: ensure all local SageMaker docs exist in S3.

        Steps:
        1) List existing objects in the bucket (under the configured prefix).
        2) Scan local `sagemaker-docs/`.
        3) Upload missing files in parallel, showing a tqdm progress bar.
        4) Log a short summary of planned uploads and outcomes.
        """

        docs_dir = self._config.docs_dir
        if not docs_dir.exists() or not docs_dir.is_dir():
            logger.warning("SageMaker docs dir not found, skipping sync: %s", docs_dir)
            return

        prefix = self._config.s3_prefix
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"

        logger.info("SageMaker docs startup sync: listing S3 objects (prefix=%r)", prefix)
        existing_items = await self._s3.list_files(prefix=prefix)
        existing_keys = {item.key for item in existing_items}

        local_files = sorted(p for p in docs_dir.rglob("*") if p.is_file())
        planned = [
            item
            for path in local_files
            if (item := self._planned_upload_item(docs_dir=docs_dir, prefix=prefix, existing_keys=existing_keys, path=path))
            is not None
        ]

        to_upload = len(planned)
        logger.info(
            "SageMaker docs startup sync: local=%d, s3(prefix)=%d, to_upload=%d",
            len(local_files),
            len(existing_items),
            to_upload,
        )

        if to_upload == 0:
            logger.info("SageMaker docs startup sync: nothing to upload")
            return

        semaphore = asyncio.Semaphore(self._config.concurrency)

        async def _upload_one(path: Path, key: str) -> tuple[str, bool, Optional[str]]:
            async with semaphore:
                try:
                    await self._s3.upload_local_file(path=path, key=key, content_type="text/markdown")
                    return (key, True, None)
                except S3ServiceError as exc:
                    return (key, False, str(exc))
                except Exception as exc:  # pragma: no cover
                    return (key, False, str(exc))

        tasks = [asyncio.create_task(_upload_one(path, key)) for path, key in planned]

        succeeded = 0
        failed = 0

        for fut in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Uploading SageMaker docs",
            unit="file",
        ):
            key, ok, err = await fut
            if ok:
                succeeded += 1
            else:
                failed += 1
                logger.error("SageMaker docs upload failed (key=%s): %s", key, err)

        logger.info(
            "SageMaker docs startup sync complete: to_upload=%d, succeeded=%d, failed=%d",
            to_upload,
            succeeded,
            failed,
        )


@dataclass(frozen=True)
class SageMakerDocsConfig:
    """Configuration for working with the local `sagemaker-docs/` folder."""

    docs_dir: Path
    source_name: str = "sagemaker-docs"

    @staticmethod
    def from_env(*, docs_dir: Path) -> "SageMakerDocsConfig":
        return SageMakerDocsConfig(
            docs_dir=docs_dir,
            source_name=os.getenv("OPENSEARCH_DOCS_SOURCE_NAME", "sagemaker-docs"),
        )


class SageMakerDocsService:
    """Generic helpers for working with SageMaker docs on disk.

    This service does not talk to OpenSearch; it only deals with local doc files.
    """

    def __init__(self, config: SageMakerDocsConfig) -> None:
        self._config = config

    @property
    def docs_dir(self) -> Path:
        return self._config.docs_dir

    @property
    def source_name(self) -> str:
        return self._config.source_name

    def list_markdown_files(self) -> list[Path]:
        docs_dir = self._config.docs_dir
        if not docs_dir.exists() or not docs_dir.is_dir():
            return []
        return sorted(p for p in docs_dir.rglob("*.md") if p.is_file())

    def relative_path(self, *, path: Path) -> str:
        return path.relative_to(self._config.docs_dir).as_posix()

    @staticmethod
    def doc_id_from_rel_path(rel_path: str) -> str:
        # Stable id (hex) derived from relative path.
        return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()

    @staticmethod
    def read_text_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="replace")
