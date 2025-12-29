"""Configuration package (Facade).

This package acts as a small *Facade* over the underlying configuration modules.
Instead of requiring callers to know the exact module that defines each config
object (for example, ``opensearch_config.py``), we re-export the public config
types here so the rest of the codebase can import from a single, stable path:

	from app.services.config import OpenSearchConfig

Benefits:
- Keeps imports consistent and shorter.
- Allows internal module layout changes without touching all call sites.
- Clearly defines the public API of this package (via ``__all__``).
"""

from app.services.config.opensearch_config import OpenSearchConfig
from app.services.config.s3_config import S3Config
from app.services.config.sagemaker_docs import (
	SageMakerDocsConfig,
	SageMakerDocsSyncConfig,
)

__all__ = ["OpenSearchConfig", "S3Config", "SageMakerDocsConfig", "SageMakerDocsSyncConfig"]
