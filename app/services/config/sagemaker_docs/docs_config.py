from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
