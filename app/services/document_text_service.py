from __future__ import annotations

import os
from typing import Optional

from langchain_aws.embeddings import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentTextServiceError(RuntimeError):
    pass


class DocumentTextService:

    _BEDROCK_EMBEDDING_DIM_DEFAULT = 1024

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self._embeddings: Optional[BedrockEmbeddings] = None

    def _get_embeddings_dimensions(self) -> int:
        dim = int(os.getenv("BEDROCK_EMBEDDING_DIM", self._BEDROCK_EMBEDDING_DIM_DEFAULT))
        if dim <= 0:
            dim = self._BEDROCK_EMBEDDING_DIM_DEFAULT
        return dim

    def _get_bedrock_embeddings(self) -> BedrockEmbeddings:
        if self._embeddings is not None:
            return self._embeddings

        region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        model_id = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
        target_dim = self._get_embeddings_dimensions()

        model_kwargs: Optional[dict[str, object]] = None
        if "titan-embed-text-v2" in model_id:
            model_kwargs = {"dimensions": target_dim}

        self._embeddings = BedrockEmbeddings(
            region_name=region_name,
            model_id=model_id,
            model_kwargs=model_kwargs,
        )
        return self._embeddings

    def split_text_into_chunks(self, text: str) -> list[str]:
        """Split input text into overlapping chunks for downstream processing.

        Uses LangChain's RecursiveCharacterTextSplitter with:
        - chunk_size=500
        - chunk_overlap=50
        """

        if not text:
            return []

        return self._splitter.split_text(text)

    def text_to_embedding(self, text: str) -> list[float]:
        """Convert a text string into an embedding vector using Amazon Bedrock.

        Uses LangChain `BedrockEmbeddings` with Amazon Titan Text Embeddings.
        Override the model via `BEDROCK_EMBEDDING_MODEL_ID` if needed.

        Configures Titan v2 to return native 1024-dimensional embeddings
        (or another dimension via `BEDROCK_EMBEDDING_DIM`).
        """

        if not text:
            return []

        try:
            embedding = self._get_bedrock_embeddings().embed_query(text)

            expected_dim = self._get_embeddings_dimensions()

            if len(embedding) != expected_dim:
                raise DocumentTextServiceError(
                    f"Unexpected embedding size {len(embedding)}; expected {expected_dim}. "
                    f"Check BEDROCK_EMBEDDING_MODEL_ID and BEDROCK_EMBEDDING_DIM."
                )

            return embedding
        except Exception as exc:
            raise DocumentTextServiceError("Failed to embed text using Bedrock") from exc
