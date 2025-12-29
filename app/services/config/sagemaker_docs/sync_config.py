from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SageMakerDocsSyncConfig:
    """Configuration for SageMaker docs startup sync.

    This is service wiring/runtime configuration (paths, prefix, concurrency), not an API
    request/response schema.
    """

    docs_dir: Path
    s3_prefix: str = "sagemaker-docs/"
    concurrency: int = 10
