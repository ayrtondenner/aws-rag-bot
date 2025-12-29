from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Path, Query

from app.models.s3 import DeleteResponse, FileListResponse
from app.services.dependencies import get_s3_service
from app.services.s3_service import S3Service

router = APIRouter(prefix="/s3", tags=["s3"])


@router.get("/files", response_model=FileListResponse)
async def list_files(
    prefix: Optional[str] = Query(default=None),
    s3: S3Service = Depends(get_s3_service),
) -> FileListResponse:
    files = await s3.list_files(prefix=prefix)
    return FileListResponse(count=len(files), files=files)

@router.delete("/files/{key:path}", response_model=DeleteResponse)
async def delete_file(
    key: str = Path(..., description="S3 object key"),
    s3: S3Service = Depends(get_s3_service),
) -> DeleteResponse:
    await s3.delete_file(key=key)
    return DeleteResponse(key=key, deleted=True)
