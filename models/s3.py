from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class FileItem(BaseModel):
    key: str = Field(..., description="S3 object key")
    size: Optional[int] = None
    last_modified: Optional[datetime] = None
    etag: Optional[str] = None

    @staticmethod
    def from_s3_object(obj: dict[str, Any]) -> "FileItem":
        return FileItem(
            key=str(obj.get("Key")),
            size=obj.get("Size"),
            last_modified=obj.get("LastModified"),
            etag=obj.get("ETag"),
        )


class ListFilesResponse(BaseModel):
    files: list[FileItem]


class UploadResponse(BaseModel):
    key: str


class DeleteResponse(BaseModel):
    key: str
    deleted: bool
