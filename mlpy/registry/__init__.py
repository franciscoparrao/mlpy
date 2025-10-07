"""
Model Registry for MLPY.

This module provides model registry functionality for storing, versioning,
and managing trained models with metadata and tagging support.
"""

from .base import ModelRegistry, ModelVersion, ModelMetadata, ModelStage
from .filesystem import FileSystemRegistry
from .utils import generate_model_id, compare_models, validate_model_name, validate_version_string

__all__ = [
    'ModelRegistry',
    'ModelVersion',
    'ModelMetadata',
    'ModelStage',
    'FileSystemRegistry',
    'generate_model_id',
    'compare_models',
    'validate_model_name',
    'validate_version_string'
]