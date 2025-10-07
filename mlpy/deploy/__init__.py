"""
Módulo de deployment para MLPY.

Este módulo proporciona funcionalidad para desplegar modelos MLPY
como servicios API usando FastAPI o Flask.
"""

from .api import MLPYModelServer, create_app
from .client import MLPYClient
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    HealthCheck
)

__all__ = [
    'MLPYModelServer',
    'create_app',
    'MLPYClient',
    'PredictionRequest',
    'PredictionResponse',
    'ModelInfo',
    'HealthCheck'
]