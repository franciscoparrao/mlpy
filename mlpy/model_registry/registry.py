"""
Core Model Registry para MLPY.

Sistema centralizado para registrar y gestionar todos los modelos disponibles
con metadata completa, categorización y búsqueda inteligente.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from enum import Enum
import importlib
import inspect
from pathlib import Path

from ..core.base import MLPYObject
from ..learners.base import Learner


class ModelCategory(Enum):
    """Categorías principales de modelos."""
    TRADITIONAL_ML = "traditional_ml"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    PROBABILISTIC = "probabilistic"
    SPECIALIZED = "specialized"


class TaskType(Enum):
    """Tipos de tareas soportadas."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ANOMALY_DETECTION = "anomaly_detection"
    TEXT_GENERATION = "text_generation"
    SEQUENCE_PREDICTION = "sequence_prediction"
    MULTI_LABEL = "multi_label"
    MULTI_OUTPUT = "multi_output"


class Complexity(Enum):
    """Niveles de complejidad computacional."""
    LOW = "low"          # < 1 min training on 10k samples
    MEDIUM = "medium"    # 1-10 min training on 10k samples  
    HIGH = "high"        # 10-60 min training on 10k samples
    VERY_HIGH = "very_high"  # > 60 min training on 10k samples


@dataclass
class ModelMetadata:
    """Metadata completa de un modelo."""
    
    # Basic Info
    name: str
    display_name: str
    description: str
    category: ModelCategory
    
    # Technical Details
    class_path: str  # e.g., "mlpy.learners.sklearn.LearnerRandomForest"
    task_types: List[TaskType]
    complexity: Complexity
    
    # Capabilities
    supports_probabilities: bool = False
    supports_feature_importance: bool = False
    supports_incremental_learning: bool = False
    supports_gpu: bool = False
    supports_parallel: bool = False
    supports_sparse_data: bool = False
    supports_categorical: bool = False
    supports_missing_values: bool = False
    
    # Data Requirements
    min_samples: int = 10
    max_samples: Optional[int] = None
    min_features: int = 1
    max_features: Optional[int] = None
    
    # Performance Characteristics
    training_speed: str = "medium"  # slow, medium, fast
    prediction_speed: str = "medium"
    memory_usage: str = "medium"    # low, medium, high
    
    # Dependencies
    required_packages: List[str] = None
    optional_packages: List[str] = None
    
    # Use Cases
    recommended_for: List[str] = None
    not_recommended_for: List[str] = None
    
    # Examples and Documentation
    example_code: Optional[str] = None
    documentation_url: Optional[str] = None
    paper_reference: Optional[str] = None
    
    # Version and Status
    version: str = "1.0.0"
    status: str = "stable"  # experimental, beta, stable, deprecated
    
    # Auto-generated
    import_path: Optional[str] = None
    available: bool = True
    
    def __post_init__(self):
        """Post-init processing."""
        if self.required_packages is None:
            self.required_packages = []
        if self.optional_packages is None:
            self.optional_packages = []
        if self.recommended_for is None:
            self.recommended_for = []
        if self.not_recommended_for is None:
            self.not_recommended_for = []


class ModelRegistry:
    """
    Registry centralizado para todos los modelos de MLPY.
    
    Proporciona funcionalidades para:
    - Registrar modelos con metadata completa
    - Buscar modelos por criterios
    - Obtener recomendaciones automáticas
    - Validar disponibilidad de dependencias
    """
    
    def __init__(self):
        self._models: Dict[str, ModelMetadata] = {}
        self._categories: Dict[ModelCategory, List[str]] = {}
        self._task_types: Dict[TaskType, List[str]] = {}
        self._initialized = False
    
    def initialize(self):
        """Inicializar registry con todos los modelos disponibles."""
        if self._initialized:
            return
        
        # Registrar modelos automáticamente
        self._register_builtin_models()
        self._initialized = True
    
    def register(self, metadata: ModelMetadata) -> bool:
        """
        Registrar un modelo en el registry.
        
        Parameters:
        -----------
        metadata : ModelMetadata
            Metadata completa del modelo
            
        Returns:
        --------
        bool : True si el registro fue exitoso
        """
        # Validar disponibilidad
        metadata.available = self._check_availability(metadata)
        
        # Registrar
        self._models[metadata.name] = metadata
        
        # Indexar por categoría
        if metadata.category not in self._categories:
            self._categories[metadata.category] = []
        self._categories[metadata.category].append(metadata.name)
        
        # Indexar por tipo de tarea
        for task_type in metadata.task_types:
            if task_type not in self._task_types:
                self._task_types[task_type] = []
            self._task_types[task_type].append(metadata.name)
        
        return True
    
    def get(self, name: str) -> Optional[ModelMetadata]:
        """Obtener metadata de un modelo."""
        self.initialize()
        return self._models.get(name)
    
    def list_all(self) -> List[ModelMetadata]:
        """Listar todos los modelos registrados."""
        self.initialize()
        return list(self._models.values())
    
    def list_by_category(self, category: ModelCategory) -> List[ModelMetadata]:
        """Listar modelos por categoría."""
        self.initialize()
        names = self._categories.get(category, [])
        return [self._models[name] for name in names]
    
    def list_by_task_type(self, task_type: TaskType) -> List[ModelMetadata]:
        """Listar modelos por tipo de tarea."""
        self.initialize()
        names = self._task_types.get(task_type, [])
        return [self._models[name] for name in names]
    
    def search(
        self,
        category: Optional[ModelCategory] = None,
        task_type: Optional[TaskType] = None,
        complexity: Optional[Complexity] = None,
        supports_gpu: Optional[bool] = None,
        supports_probabilities: Optional[bool] = None,
        min_samples: Optional[int] = None,
        max_samples: Optional[int] = None,
        available_only: bool = True,
        **kwargs
    ) -> List[ModelMetadata]:
        """
        Buscar modelos por criterios específicos.
        
        Parameters:
        -----------
        category : ModelCategory, optional
            Categoría del modelo
        task_type : TaskType, optional
            Tipo de tarea
        complexity : Complexity, optional
            Nivel de complejidad máximo
        supports_gpu : bool, optional
            Requiere soporte GPU
        supports_probabilities : bool, optional
            Requiere soporte de probabilidades
        min_samples : int, optional
            Número mínimo de muestras
        max_samples : int, optional
            Número máximo de muestras
        available_only : bool
            Solo modelos con dependencias disponibles
            
        Returns:
        --------
        List[ModelMetadata] : Modelos que cumplen los criterios
        """
        self.initialize()
        
        results = list(self._models.values())
        
        # Filtros
        if category is not None:
            results = [m for m in results if m.category == category]
        
        if task_type is not None:
            results = [m for m in results if task_type in m.task_types]
        
        if complexity is not None:
            complexity_order = {
                Complexity.LOW: 0,
                Complexity.MEDIUM: 1, 
                Complexity.HIGH: 2,
                Complexity.VERY_HIGH: 3
            }
            max_complexity = complexity_order[complexity]
            results = [
                m for m in results 
                if complexity_order[m.complexity] <= max_complexity
            ]
        
        if supports_gpu is not None:
            results = [m for m in results if m.supports_gpu == supports_gpu]
        
        if supports_probabilities is not None:
            results = [m for m in results if m.supports_probabilities == supports_probabilities]
        
        if min_samples is not None:
            results = [m for m in results if m.min_samples <= min_samples]
        
        if max_samples is not None:
            results = [
                m for m in results 
                if m.max_samples is None or m.max_samples >= max_samples
            ]
        
        if available_only:
            results = [m for m in results if m.available]
        
        # Filtros adicionales de kwargs
        for key, value in kwargs.items():
            if hasattr(ModelMetadata, key):
                results = [m for m in results if getattr(m, key) == value]
        
        return results
    
    def get_recommendations(
        self,
        task_type: TaskType,
        n_samples: int,
        n_features: int,
        complexity_preference: Complexity = Complexity.MEDIUM,
        **preferences
    ) -> List[Tuple[ModelMetadata, float]]:
        """
        Obtener recomendaciones de modelos con scoring.
        
        Returns:
        --------
        List[Tuple[ModelMetadata, float]] : (modelo, score) ordenado por score
        """
        candidates = self.search(
            task_type=task_type,
            min_samples=n_samples,
            available_only=True
        )
        
        scored_models = []
        
        for model in candidates:
            score = self._calculate_model_score(
                model, 
                task_type,
                n_samples,
                n_features,
                complexity_preference,
                **preferences
            )
            scored_models.append((model, score))
        
        # Ordenar por score descendente
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        return scored_models
    
    def _calculate_model_score(
        self,
        model: ModelMetadata,
        task_type: TaskType,
        n_samples: int,
        n_features: int,
        complexity_preference: Complexity,
        **preferences
    ) -> float:
        """Calcular score de recomendación para un modelo."""
        score = 0.0
        
        # Score base por tipo de tarea
        if task_type in model.task_types:
            score += 10.0
        
        # Penalizar/premiar por complejidad
        complexity_scores = {
            Complexity.LOW: {Complexity.LOW: 5, Complexity.MEDIUM: 3, Complexity.HIGH: 1, Complexity.VERY_HIGH: 0},
            Complexity.MEDIUM: {Complexity.LOW: 3, Complexity.MEDIUM: 5, Complexity.HIGH: 4, Complexity.VERY_HIGH: 2},
            Complexity.HIGH: {Complexity.LOW: 1, Complexity.MEDIUM: 4, Complexity.HIGH: 5, Complexity.VERY_HIGH: 4},
            Complexity.VERY_HIGH: {Complexity.LOW: 0, Complexity.MEDIUM: 2, Complexity.HIGH: 4, Complexity.VERY_HIGH: 5}
        }
        score += complexity_scores.get(complexity_preference, {}).get(model.complexity, 0)
        
        # Score por características específicas
        if preferences.get('needs_probabilities') and model.supports_probabilities:
            score += 3.0
        
        if preferences.get('needs_feature_importance') and model.supports_feature_importance:
            score += 2.0
        
        if preferences.get('large_dataset') and model.supports_parallel:
            score += 2.0
        
        if preferences.get('gpu_available') and model.supports_gpu:
            score += 3.0
        
        # Penalizar si no cumple requisitos de datos
        if n_samples < model.min_samples:
            score -= 5.0
        
        if model.max_samples and n_samples > model.max_samples:
            score -= 3.0
        
        # Score por categoría (algunas son más populares/confiables)
        category_scores = {
            ModelCategory.TRADITIONAL_ML: 3.0,
            ModelCategory.ENSEMBLE: 4.0,
            ModelCategory.DEEP_LEARNING: 2.0,
            ModelCategory.UNSUPERVISED: 2.0
        }
        score += category_scores.get(model.category, 1.0)
        
        return max(0.0, score)
    
    def _check_availability(self, metadata: ModelMetadata) -> bool:
        """Verificar si un modelo está disponible (dependencias instaladas)."""
        try:
            # Verificar paquetes requeridos
            for package in metadata.required_packages:
                importlib.import_module(package)
            
            # Intentar importar la clase del modelo
            if metadata.class_path:
                module_path, class_name = metadata.class_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                getattr(module, class_name)
            
            return True
        except ImportError:
            return False
    
    def _register_builtin_models(self):
        """Registrar todos los modelos built-in de MLPY."""
        
        # Traditional ML Models
        traditional_models = [
            {
                "name": "random_forest_classifier",
                "display_name": "Random Forest Classifier",
                "description": "Ensemble of decision trees with random feature selection",
                "category": ModelCategory.TRADITIONAL_ML,
                "class_path": "mlpy.learners.sklearn.LearnerRandomForestClassifier",
                "task_types": [TaskType.CLASSIFICATION],
                "complexity": Complexity.MEDIUM,
                "supports_probabilities": True,
                "supports_feature_importance": True,
                "supports_parallel": True,
                "training_speed": "fast",
                "prediction_speed": "fast",
                "memory_usage": "medium",
                "required_packages": ["sklearn"],
                "recommended_for": ["tabular_data", "medium_datasets", "feature_importance"],
                "min_samples": 50
            },
            {
                "name": "xgboost_classifier",
                "display_name": "XGBoost Classifier", 
                "description": "Gradient boosting framework optimized for speed and performance",
                "category": ModelCategory.TRADITIONAL_ML,
                "class_path": "mlpy.learners.xgboost_wrapper.LearnerXGBoostClassif",
                "task_types": [TaskType.CLASSIFICATION],
                "complexity": Complexity.MEDIUM,
                "supports_probabilities": True,
                "supports_feature_importance": True,
                "supports_gpu": True,
                "training_speed": "fast",
                "prediction_speed": "fast",
                "memory_usage": "medium",
                "required_packages": ["xgboost"],
                "recommended_for": ["tabular_data", "competitions", "performance"],
                "min_samples": 100
            }
        ]
        
        # Deep Learning Models
        deep_learning_models = [
            {
                "name": "lstm_classifier",
                "display_name": "LSTM Classifier",
                "description": "Long Short-Term Memory network for sequence classification",
                "category": ModelCategory.DEEP_LEARNING,
                "class_path": "mlpy.learners.deep_learning.LearnerLSTM",
                "task_types": [TaskType.CLASSIFICATION, TaskType.SEQUENCE_PREDICTION],
                "complexity": Complexity.HIGH,
                "supports_probabilities": True,
                "supports_gpu": True,
                "training_speed": "slow",
                "prediction_speed": "medium",
                "memory_usage": "high",
                "required_packages": ["torch"],
                "recommended_for": ["sequences", "time_series", "text"],
                "min_samples": 1000
            }
        ]
        
        # NLP Models
        nlp_models = [
            {
                "name": "bert_classifier",
                "display_name": "BERT Classifier",
                "description": "Bidirectional Encoder Representations from Transformers",
                "category": ModelCategory.NLP,
                "class_path": "mlpy.learners.nlp.LearnerBERTClassifier",
                "task_types": [TaskType.CLASSIFICATION],
                "complexity": Complexity.VERY_HIGH,
                "supports_probabilities": True,
                "supports_gpu": True,
                "training_speed": "slow",
                "prediction_speed": "medium",
                "memory_usage": "very_high",
                "required_packages": ["transformers", "torch"],
                "recommended_for": ["text_classification", "sentiment_analysis", "nlp"],
                "not_recommended_for": ["small_datasets", "non_text"],
                "min_samples": 100
            }
        ]
        
        # Ensemble Models
        ensemble_models = [
            {
                "name": "adaptive_ensemble",
                "display_name": "Adaptive Ensemble",
                "description": "Ensemble with automatic learner selection and weight optimization",
                "category": ModelCategory.ENSEMBLE,
                "class_path": "mlpy.learners.ensemble_advanced.LearnerAdaptiveEnsemble",
                "task_types": [TaskType.CLASSIFICATION, TaskType.REGRESSION],
                "complexity": Complexity.HIGH,
                "supports_probabilities": True,
                "supports_feature_importance": True,
                "training_speed": "slow",
                "prediction_speed": "medium",
                "memory_usage": "high",
                "recommended_for": ["high_performance", "robust_predictions", "competitions"],
                "min_samples": 500
            }
        ]
        
        # Unsupervised Models
        unsupervised_models = [
            {
                "name": "dbscan",
                "display_name": "DBSCAN Clustering",
                "description": "Density-based clustering with automatic parameter tuning",
                "category": ModelCategory.UNSUPERVISED,
                "class_path": "mlpy.learners.unsupervised.LearnerDBSCAN",
                "task_types": [TaskType.CLUSTERING],
                "complexity": Complexity.MEDIUM,
                "supports_parallel": True,
                "training_speed": "medium",
                "prediction_speed": "fast",
                "memory_usage": "medium",
                "required_packages": ["sklearn"],
                "recommended_for": ["outlier_detection", "irregular_clusters", "noise_handling"],
                "min_samples": 50
            }
        ]
        
        # Registrar todos los modelos
        all_models = (
            traditional_models + 
            deep_learning_models + 
            nlp_models + 
            ensemble_models + 
            unsupervised_models
        )
        
        for model_dict in all_models:
            metadata = ModelMetadata(**model_dict)
            self.register(metadata)
    
    def export_to_json(self, filepath: str):
        """Exportar registry a JSON."""
        self.initialize()
        
        data = {}
        for name, metadata in self._models.items():
            # Convertir enums a strings para JSON
            metadata_dict = asdict(metadata)
            metadata_dict['category'] = metadata.category.value
            metadata_dict['task_types'] = [tt.value for tt in metadata.task_types]
            metadata_dict['complexity'] = metadata.complexity.value
            data[name] = metadata_dict
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_json(self, filepath: str):
        """Cargar registry desde JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for name, metadata_dict in data.items():
            # Convertir strings a enums
            metadata_dict['category'] = ModelCategory(metadata_dict['category'])
            metadata_dict['task_types'] = [TaskType(tt) for tt in metadata_dict['task_types']]
            metadata_dict['complexity'] = Complexity(metadata_dict['complexity'])
            
            metadata = ModelMetadata(**metadata_dict)
            self.register(metadata)


# Instancia global del registry
_global_registry = ModelRegistry()


def register_model(metadata: ModelMetadata) -> bool:
    """Registrar un modelo en el registry global."""
    return _global_registry.register(metadata)


def get_model(name: str) -> Optional[ModelMetadata]:
    """Obtener metadata de un modelo del registry global."""
    return _global_registry.get(name)


def list_models(
    category: Optional[ModelCategory] = None,
    task_type: Optional[TaskType] = None
) -> List[ModelMetadata]:
    """Listar modelos del registry global."""
    if category:
        return _global_registry.list_by_category(category)
    elif task_type:
        return _global_registry.list_by_task_type(task_type)
    else:
        return _global_registry.list_all()


def search_models(**criteria) -> List[ModelMetadata]:
    """Buscar modelos en el registry global."""
    return _global_registry.search(**criteria)


__all__ = [
    'ModelRegistry',
    'ModelMetadata',
    'ModelCategory',
    'TaskType',
    'Complexity',
    'register_model',
    'get_model',
    'list_models',
    'search_models'
]