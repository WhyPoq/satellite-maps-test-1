"""UI-independent image-processing pipeline."""
from .config import PipelineConfig
from .runner import process_image

__all__ = ["PipelineConfig", "process_image"]
