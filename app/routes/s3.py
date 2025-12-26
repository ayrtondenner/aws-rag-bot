from __future__ import annotations

from fastapi import APIRouter, Depends, File, Path, Query, UploadFile

from app.models.s3 import DeleteResponse, FileItem, ListFilesResponse, UploadResponse
from app.services.dependencies import get_s3_service
from app.services.s3_service import S3Service

router = APIRouter(prefix="/s3", tags=["s3"])


@router.get("/files", response_model=ListFilesResponse)
async def list_files(
    prefix: str | None = Query(default=None),
    s3: S3Service = Depends(get_s3_service),
) -> ListFilesResponse:
    objects = await s3.list_files(prefix=prefix)
    return ListFilesResponse(files=[FileItem.from_s3_object(o) for o in objects])


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    key: str | None = Query(default=None, description="Optional object key override"),
    s3: S3Service = Depends(get_s3_service),
) -> UploadResponse:
    uploaded_key = await s3.upload_file(file=file, key=key)
    return UploadResponse(key=uploaded_key)


@router.delete("/files/{key:path}", response_model=DeleteResponse)
async def delete_file(
    key: str = Path(..., description="S3 object key"),
    s3: S3Service = Depends(get_s3_service),
) -> DeleteResponse:
    await s3.delete_file(key=key)
    return DeleteResponse(key=key, deleted=True)
