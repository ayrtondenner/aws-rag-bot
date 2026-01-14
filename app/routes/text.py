from __future__ import annotations

from fastapi import APIRouter, Depends

from app.models.text import (
    EmbedTextRequest,
    EmbedTextResponse,
    SplitTextRequest,
    SplitTextResponse,
)
from app.services.dependencies import get_document_service
from app.services.document_service import DocumentService

router = APIRouter(prefix="/text", tags=["text"])


@router.post("/split", response_model=SplitTextResponse)
async def split_text(
    payload: SplitTextRequest,
    svc: DocumentService = Depends(get_document_service),
) -> SplitTextResponse:
    """Split text into overlapping chunks/phrases for downstream indexing."""

    chunks = svc.split_text_into_chunks(payload.text)
    return SplitTextResponse(count=len(chunks), chunks=chunks)


@router.post("/embed", response_model=EmbedTextResponse)
async def embed_text(
    payload: EmbedTextRequest,
    svc: DocumentService = Depends(get_document_service),
) -> EmbedTextResponse:
    """Generate an embedding vector for the provided text using Bedrock."""

    embedding = svc.text_to_embedding(payload.text)
    return EmbedTextResponse(dimensions=len(embedding), embedding=embedding)
