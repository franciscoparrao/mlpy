"""Model persistence and serialization for MLPY.

This module provides functionality to save and load trained models,
pipelines, and other MLPY objects.
"""

from .base import (
    ModelSerializer,
    ModelBundle,
    save_model,
    load_model,
    SERIALIZERS
)

from .serializers import (
    PickleSerializer,
    JoblibSerializer,
    JSONSerializer
)

from .utils import (
    ModelRegistry,
    export_model_package
)

# Optional serializers
try:
    from .onnx_serializer import ONNXSerializer
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

__all__ = [
    "ModelSerializer",
    "ModelBundle",
    "save_model",
    "load_model",
    "SERIALIZERS",
    "PickleSerializer",
    "JoblibSerializer", 
    "JSONSerializer",
    "ModelRegistry",
    "export_model_package"
]

if ONNX_AVAILABLE:
    __all__.append("ONNXSerializer")