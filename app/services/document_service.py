from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_aws.embeddings import BedrockEmbeddings
from llama_index.core.node_parser import SentenceSplitter


class DocumentServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalDocsConfig:
    """Configuration for working with the local `sagemaker-docs/` folder."""

    docs_dir: Path
    source_name: str = "sagemaker-docs"


class DocumentService:
    """Document-only utilities (no S3/OpenSearch calls).

    Responsibilities:
    - Work with local docs on disk (list/read/id).
    - Split text into overlapping chunks/phrases.
    - Generate embeddings for text via Amazon Bedrock.
    """

    _BEDROCK_EMBEDDING_DIM_DEFAULT = 1024

    def __init__(
        self,
        *,
        docs: Optional[LocalDocsConfig] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self._docs = docs
        # Requirement: embedding index uses overlapping *phrases*.
        # We use LlamaIndex's SentenceSplitter to prefer sentence/phrase boundaries.
        self._splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._embeddings: Optional[BedrockEmbeddings] = None

    # -----------------
    # Local docs helpers
    # -----------------

    @property
    def docs_dir(self) -> Path:
        if self._docs is None:
            raise DocumentServiceError("Local docs config not provided")
        return self._docs.docs_dir

    @property
    def source_name(self) -> str:
        if self._docs is None:
            return "sagemaker-docs"
        return self._docs.source_name

    def list_markdown_files(self) -> list[Path]:
        if self._docs is None:
            return []
        docs_dir = self._docs.docs_dir
        if not docs_dir.exists() or not docs_dir.is_dir():
            return []
        return sorted(p for p in docs_dir.rglob("*.md") if p.is_file())

    def relative_path(self, *, path: Path) -> str:
        if self._docs is None:
            raise DocumentServiceError("Local docs config not provided")
        return path.relative_to(self._docs.docs_dir).as_posix()

    @staticmethod
    def doc_id_from_rel_path(rel_path: str) -> str:
        return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()

    @staticmethod
    def read_text_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="replace")

    # -----------------
    # Text splitting
    # -----------------

    def split_text_into_chunks(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        For this project, chunks are treated as overlapping phrases (SentenceSplitter).
        """

        if not text:
            return []
        return list(self._splitter.split_text(text))

    def split_text_into_phrases(self, text: str) -> list[str]:
        """Alias for phrase-based splitting used for embedding ingestion."""

        return self.split_text_into_chunks(text)

    # -----------------
    # Embeddings
    # -----------------

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

    def text_to_embedding(self, text: str) -> list[float]:
        """Convert a text string into an embedding vector using Amazon Bedrock."""

        if not text:
            return []

        try:
            embedding = self._get_bedrock_embeddings().embed_query(text)
            expected_dim = self._get_embeddings_dimensions()

            if len(embedding) != expected_dim:
                raise DocumentServiceError(
                    f"Unexpected embedding size {len(embedding)}; expected {expected_dim}. "
                    "Check BEDROCK_EMBEDDING_MODEL_ID and BEDROCK_EMBEDDING_DIM."
                )

            return embedding
        except DocumentServiceError:
            raise
        except Exception as exc:
            raise DocumentServiceError("Failed to embed text using Bedrock") from exc
