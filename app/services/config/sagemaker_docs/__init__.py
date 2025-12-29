"""SageMaker docs configuration (Facade subpackage).

This subpackage groups configuration types used by the SageMaker docs services.
Import from here to avoid depending on the internal module layout.

    from app.services.config.sagemaker_docs import SageMakerDocsConfig, SageMakerDocsSyncConfig
"""

# TODO: Check if its necessary to have two separate config files for sagemaker docs
# If possible, merge then, and move out of subpackage.

from app.services.config.sagemaker_docs.docs_config import SageMakerDocsConfig
from app.services.config.sagemaker_docs.sync_config import SageMakerDocsSyncConfig

__all__ = ["SageMakerDocsConfig", "SageMakerDocsSyncConfig"]
