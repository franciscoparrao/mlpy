"""
Sistema de Model Registry para MLPY.

Este módulo proporciona un sistema centralizado para registrar, buscar,
y gestionar todos los modelos disponibles en MLPY con metadata completa.
"""

from .registry import (
    ModelRegistry,
    ModelMetadata,
    ModelCategory,
    register_model,
    get_model,
    list_models,
    search_models
)

from .auto_selector import (
    AutoModelSelector,
    ModelRecommendation,
    select_best_model,
    recommend_models
)

from .factory import (
    ModelFactory,
    create_model,
    create_ensemble,
    get_model_info
)

__all__ = [
    # Registry Core
    'ModelRegistry',
    'ModelMetadata',
    'ModelCategory',
    'register_model',
    'get_model',
    'list_models',
    'search_models',
    
    # Auto Selection
    'AutoModelSelector',
    'ModelRecommendation', 
    'select_best_model',
    'recommend_models',
    
    # Factory
    'ModelFactory',
    'create_model',
    'create_ensemble',
    'get_model_info'
]