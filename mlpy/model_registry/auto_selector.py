"""
Auto Model Selector para MLPY.

Sistema inteligente que recomienda automáticamente los mejores modelos
basado en características de los datos y requerimientos del usuario.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum

from .registry import (
    ModelRegistry, ModelMetadata, ModelCategory, TaskType, Complexity,
    _global_registry
)
from ..tasks import Task, TaskClassif, TaskRegr
from ..validation.validators import validate_task_data


class DatasetSize(Enum):
    """Categorías de tamaño de dataset."""
    TINY = "tiny"        # < 1K samples
    SMALL = "small"      # 1K - 10K samples
    MEDIUM = "medium"    # 10K - 100K samples
    LARGE = "large"      # 100K - 1M samples
    VERY_LARGE = "very_large"  # > 1M samples


class DatasetComplexity(Enum):
    """Categorías de complejidad de datos."""
    SIMPLE = "simple"        # Linear/simple patterns
    MODERATE = "moderate"    # Some non-linearity
    COMPLEX = "complex"      # High non-linearity, interactions
    VERY_COMPLEX = "very_complex"  # Deep patterns, high dimensionality


@dataclass
class DataCharacteristics:
    """Características analizadas de un dataset."""
    
    # Basic stats
    n_samples: int
    n_features: int
    n_classes: Optional[int] = None
    
    # Data quality
    missing_ratio: float = 0.0
    duplicate_ratio: float = 0.0
    outlier_ratio: float = 0.0
    
    # Feature characteristics
    numerical_features: int = 0
    categorical_features: int = 0
    text_features: int = 0
    datetime_features: int = 0
    
    # Class balance (for classification)
    class_imbalance_ratio: Optional[float] = None
    minority_class_size: Optional[int] = None
    
    # Complexity indicators
    feature_correlation_max: float = 0.0
    target_correlation_max: float = 0.0
    noise_level: float = 0.0
    
    # Derived properties
    dataset_size: Optional[DatasetSize] = None
    dataset_complexity: Optional[DatasetComplexity] = None
    
    def __post_init__(self):
        """Calculate derived properties."""
        # Dataset size
        if self.n_samples < 1000:
            self.dataset_size = DatasetSize.TINY
        elif self.n_samples < 10000:
            self.dataset_size = DatasetSize.SMALL
        elif self.n_samples < 100000:
            self.dataset_size = DatasetSize.MEDIUM
        elif self.n_samples < 1000000:
            self.dataset_size = DatasetSize.LARGE
        else:
            self.dataset_size = DatasetSize.VERY_LARGE
        
        # Dataset complexity (simplified heuristic)
        complexity_score = 0
        
        # High dimensionality
        if self.n_features > 100:
            complexity_score += 1
        if self.n_features > 1000:
            complexity_score += 1
        
        # Feature interactions
        if self.feature_correlation_max > 0.7:
            complexity_score += 1
        
        # Target complexity
        if self.target_correlation_max < 0.3:
            complexity_score += 1
        
        # Noise
        if self.noise_level > 0.3:
            complexity_score += 1
        
        if complexity_score == 0:
            self.dataset_complexity = DatasetComplexity.SIMPLE
        elif complexity_score <= 2:
            self.dataset_complexity = DatasetComplexity.MODERATE
        elif complexity_score <= 3:
            self.dataset_complexity = DatasetComplexity.COMPLEX
        else:
            self.dataset_complexity = DatasetComplexity.VERY_COMPLEX


@dataclass
class ModelRecommendation:
    """Recomendación de modelo con justificación."""
    
    model_metadata: ModelMetadata
    confidence_score: float
    reasoning: List[str]
    warnings: List[str]
    estimated_training_time: str
    estimated_performance: str
    
    def __str__(self):
        return f"{self.model_metadata.display_name} (score: {self.confidence_score:.2f})"


class AutoModelSelector:
    """
    Selector automático de modelos basado en análisis de datos.
    
    Analiza características del dataset y recomienda los mejores modelos
    con justificación detallada.
    """
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or _global_registry
        self.analysis_cache = {}
    
    def analyze_data(self, task: Task) -> DataCharacteristics:
        """
        Analizar características del dataset.
        
        Parameters:
        -----------
        task : Task
            Tarea de MLPY a analizar
            
        Returns:
        --------
        DataCharacteristics : Características analizadas
        """
        # Check cache
        task_id = getattr(task, 'id', None)
        if task_id and task_id in self.analysis_cache:
            return self.analysis_cache[task_id]
        
        # Basic stats
        n_samples = task.nrow
        n_features = task.ncol - 1  # Exclude target
        
        # Get data
        data = task.data
        X = task.X
        y = task.y if hasattr(task, 'y') else None
        
        # Missing values
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        
        # Duplicates
        duplicate_ratio = data.duplicated().sum() / len(data)
        
        # Feature types
        numerical_features = X.select_dtypes(include=[np.number]).shape[1]
        categorical_features = X.select_dtypes(include=['object', 'category']).shape[1]
        
        # Text features (heuristic: object columns with long strings)
        text_features = 0
        for col in X.select_dtypes(include=['object']).columns:
            avg_length = X[col].astype(str).str.len().mean()
            if avg_length > 50:  # Likely text
                text_features += 1
                categorical_features -= 1
        
        # Datetime features
        datetime_features = X.select_dtypes(include=['datetime64']).shape[1]
        
        # Classification-specific analysis
        n_classes = None
        class_imbalance_ratio = None
        minority_class_size = None
        
        if isinstance(task, TaskClassif) and y is not None:
            n_classes = len(np.unique(y))
            class_counts = pd.Series(y).value_counts()
            minority_class_size = class_counts.min()
            class_imbalance_ratio = class_counts.max() / class_counts.min()
        
        # Correlation analysis (only for numerical features)
        feature_correlation_max = 0.0
        target_correlation_max = 0.0
        
        if numerical_features > 1:
            try:
                X_numerical = X.select_dtypes(include=[np.number])
                corr_matrix = X_numerical.corr().abs()
                
                # Max correlation between features (excluding diagonal)
                np.fill_diagonal(corr_matrix.values, 0)
                feature_correlation_max = corr_matrix.max().max()
                
                # Target correlation (for regression)
                if isinstance(task, TaskRegr) and y is not None:
                    y_series = pd.Series(y)
                    target_corrs = X_numerical.corrwith(y_series).abs()
                    target_correlation_max = target_corrs.max()
            except:
                pass  # Correlation calculation failed
        
        # Outlier detection (simplified Z-score method)
        outlier_ratio = 0.0
        if numerical_features > 0:
            try:
                X_numerical = X.select_dtypes(include=[np.number])
                z_scores = np.abs((X_numerical - X_numerical.mean()) / X_numerical.std())
                outliers = (z_scores > 3).any(axis=1)
                outlier_ratio = outliers.sum() / len(data)
            except:
                pass
        
        # Noise level estimation (simplified)
        noise_level = missing_ratio + min(outlier_ratio, 0.2)
        
        characteristics = DataCharacteristics(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            missing_ratio=missing_ratio,
            duplicate_ratio=duplicate_ratio,
            outlier_ratio=outlier_ratio,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            text_features=text_features,
            datetime_features=datetime_features,
            class_imbalance_ratio=class_imbalance_ratio,
            minority_class_size=minority_class_size,
            feature_correlation_max=feature_correlation_max,
            target_correlation_max=target_correlation_max,
            noise_level=noise_level
        )
        
        # Cache result
        if task_id:
            self.analysis_cache[task_id] = characteristics
        
        return characteristics
    
    def recommend_models(
        self,
        task: Task,
        top_k: int = 5,
        complexity_preference: Complexity = Complexity.MEDIUM,
        performance_preference: str = "balanced",  # "speed", "accuracy", "balanced"
        **preferences
    ) -> List[ModelRecommendation]:
        """
        Recomendar modelos para una tarea específica.
        
        Parameters:
        -----------
        task : Task
            Tarea de MLPY
        top_k : int
            Número de recomendaciones a retornar
        complexity_preference : Complexity
            Preferencia de complejidad computacional
        performance_preference : str
            Preferencia de performance ("speed", "accuracy", "balanced")
        **preferences
            Preferencias adicionales
            
        Returns:
        --------
        List[ModelRecommendation] : Lista de recomendaciones ordenadas
        """
        # Analizar datos
        data_chars = self.analyze_data(task)
        
        # Determinar tipo de tarea
        if isinstance(task, TaskClassif):
            task_type = TaskType.CLASSIFICATION
        elif isinstance(task, TaskRegr):
            task_type = TaskType.REGRESSION
        else:
            # Try to infer from data
            if data_chars.n_classes and data_chars.n_classes < 20:
                task_type = TaskType.CLASSIFICATION
            else:
                task_type = TaskType.REGRESSION
        
        # Buscar modelos candidatos
        candidates = self.registry.search(
            task_type=task_type,
            available_only=True,
            **preferences
        )
        
        # Score y filtrar candidatos
        recommendations = []
        
        for model in candidates:
            score, reasoning, warnings = self._score_model(
                model, 
                data_chars, 
                complexity_preference,
                performance_preference
            )
            
            if score > 0:  # Solo incluir modelos con score positivo
                recommendation = ModelRecommendation(
                    model_metadata=model,
                    confidence_score=score,
                    reasoning=reasoning,
                    warnings=warnings,
                    estimated_training_time=self._estimate_training_time(model, data_chars),
                    estimated_performance=self._estimate_performance(model, data_chars)
                )
                recommendations.append(recommendation)
        
        # Ordenar por score y retornar top-k
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        return recommendations[:top_k]
    
    def _score_model(
        self,
        model: ModelMetadata,
        data_chars: DataCharacteristics,
        complexity_preference: Complexity,
        performance_preference: str
    ) -> Tuple[float, List[str], List[str]]:
        """Score un modelo específico para los datos dados."""
        score = 0.0
        reasoning = []
        warnings = []
        
        # Base score
        score += 50.0
        
        # Dataset size compatibility
        if data_chars.n_samples < model.min_samples:
            score -= 30.0
            warnings.append(f"Dataset too small (need ≥{model.min_samples} samples)")
        elif data_chars.n_samples >= model.min_samples * 2:
            score += 10.0
            reasoning.append("Good sample size for this model")
        
        if model.max_samples and data_chars.n_samples > model.max_samples:
            score -= 20.0
            warnings.append(f"Dataset too large (max {model.max_samples} samples)")
        
        # Complexity matching
        complexity_scores = {
            (Complexity.LOW, Complexity.LOW): 20,
            (Complexity.LOW, Complexity.MEDIUM): 15,
            (Complexity.LOW, Complexity.HIGH): 5,
            (Complexity.LOW, Complexity.VERY_HIGH): -10,
            (Complexity.MEDIUM, Complexity.LOW): 10,
            (Complexity.MEDIUM, Complexity.MEDIUM): 20,
            (Complexity.MEDIUM, Complexity.HIGH): 15,
            (Complexity.MEDIUM, Complexity.VERY_HIGH): 5,
            (Complexity.HIGH, Complexity.LOW): 0,
            (Complexity.HIGH, Complexity.MEDIUM): 10,
            (Complexity.HIGH, Complexity.HIGH): 20,
            (Complexity.HIGH, Complexity.VERY_HIGH): 15,
            (Complexity.VERY_HIGH, Complexity.LOW): -20,
            (Complexity.VERY_HIGH, Complexity.MEDIUM): -10,
            (Complexity.VERY_HIGH, Complexity.HIGH): 10,
            (Complexity.VERY_HIGH, Complexity.VERY_HIGH): 20,
        }
        
        complexity_score = complexity_scores.get(
            (complexity_preference, model.complexity), 0
        )
        score += complexity_score
        
        if complexity_score > 15:
            reasoning.append(f"Good complexity match ({model.complexity.value})")
        elif complexity_score < 0:
            warnings.append(f"Complexity mismatch (model: {model.complexity.value})")
        
        # Data type compatibility
        if data_chars.text_features > 0:
            if model.category == ModelCategory.NLP:
                score += 25.0
                reasoning.append("Optimized for text data")
            elif model.category in [ModelCategory.TRADITIONAL_ML, ModelCategory.ENSEMBLE]:
                score -= 15.0
                warnings.append("Not optimized for text data")
        
        if data_chars.categorical_features > 0:
            if model.supports_categorical:
                score += 10.0
                reasoning.append("Handles categorical features well")
            else:
                score -= 5.0
                warnings.append("May need categorical encoding")
        
        if data_chars.missing_ratio > 0.1:
            if model.supports_missing_values:
                score += 10.0
                reasoning.append("Handles missing values natively")
            else:
                score -= 10.0
                warnings.append("Requires missing value imputation")
        
        # Performance preference
        if performance_preference == "speed":
            if model.training_speed == "fast":
                score += 15.0
                reasoning.append("Fast training speed")
            elif model.training_speed == "slow":
                score -= 10.0
                warnings.append("Slow training speed")
        
        elif performance_preference == "accuracy":
            if model.category in [ModelCategory.ENSEMBLE, ModelCategory.DEEP_LEARNING]:
                score += 15.0
                reasoning.append("Typically high accuracy")
            elif model.complexity in [Complexity.HIGH, Complexity.VERY_HIGH]:
                score += 10.0
                reasoning.append("Complex model, potentially high accuracy")
        
        # Dataset complexity matching
        if data_chars.dataset_complexity == DatasetComplexity.SIMPLE:
            if model.category == ModelCategory.TRADITIONAL_ML:
                score += 10.0
                reasoning.append("Good for simple patterns")
            elif model.category == ModelCategory.DEEP_LEARNING:
                score -= 5.0
                warnings.append("May be overkill for simple patterns")
        
        elif data_chars.dataset_complexity == DatasetComplexity.VERY_COMPLEX:
            if model.category in [ModelCategory.DEEP_LEARNING, ModelCategory.ENSEMBLE]:
                score += 15.0
                reasoning.append("Can capture complex patterns")
            elif model.category == ModelCategory.TRADITIONAL_ML:
                score -= 5.0
                warnings.append("May struggle with complex patterns")
        
        # Class imbalance considerations
        if data_chars.class_imbalance_ratio and data_chars.class_imbalance_ratio > 10:
            if "imbalanced" in model.recommended_for:
                score += 15.0
                reasoning.append("Handles imbalanced classes well")
            else:
                score -= 5.0
                warnings.append("May struggle with class imbalance")
        
        # Feature count considerations
        if data_chars.n_features > 1000:
            if model.supports_sparse_data or model.category == ModelCategory.DEEP_LEARNING:
                score += 10.0
                reasoning.append("Handles high-dimensional data")
            else:
                score -= 10.0
                warnings.append("May struggle with high dimensionality")
        
        # Memory considerations for large datasets
        if data_chars.dataset_size in [DatasetSize.LARGE, DatasetSize.VERY_LARGE]:
            if model.memory_usage == "low":
                score += 10.0
                reasoning.append("Memory efficient")
            elif model.memory_usage == "high":
                score -= 10.0
                warnings.append("High memory usage")
            
            if model.supports_parallel:
                score += 10.0
                reasoning.append("Supports parallel processing")
        
        return max(0.0, score), reasoning, warnings
    
    def _estimate_training_time(
        self, 
        model: ModelMetadata, 
        data_chars: DataCharacteristics
    ) -> str:
        """Estimate training time based on model and data characteristics."""
        
        # Base time estimates (very rough heuristics)
        base_times = {
            "fast": 1,
            "medium": 5,
            "slow": 20
        }
        
        base_time = base_times.get(model.training_speed, 5)
        
        # Adjust for dataset size
        size_multipliers = {
            DatasetSize.TINY: 0.1,
            DatasetSize.SMALL: 0.5,
            DatasetSize.MEDIUM: 1.0,
            DatasetSize.LARGE: 3.0,
            DatasetSize.VERY_LARGE: 10.0
        }
        
        time_estimate = base_time * size_multipliers.get(data_chars.dataset_size, 1.0)
        
        # Format time estimate
        if time_estimate < 1:
            return "< 1 minute"
        elif time_estimate < 60:
            return f"~{int(time_estimate)} minutes"
        else:
            hours = time_estimate / 60
            return f"~{hours:.1f} hours"
    
    def _estimate_performance(
        self, 
        model: ModelMetadata, 
        data_chars: DataCharacteristics
    ) -> str:
        """Estimate expected performance."""
        
        # This is a very simplified heuristic
        # In practice, this would be based on historical benchmarks
        
        if model.category == ModelCategory.ENSEMBLE:
            return "High (85-95%)"
        elif model.category == ModelCategory.DEEP_LEARNING:
            if data_chars.dataset_size in [DatasetSize.LARGE, DatasetSize.VERY_LARGE]:
                return "Very High (90-98%)"
            else:
                return "Medium (70-85%)"
        elif model.category == ModelCategory.TRADITIONAL_ML:
            return "Good (80-90%)"
        else:
            return "Variable"


def select_best_model(
    task: Task,
    complexity_preference: Complexity = Complexity.MEDIUM,
    **preferences
) -> Optional[ModelRecommendation]:
    """
    Seleccionar automáticamente el mejor modelo para una tarea.
    
    Parameters:
    -----------
    task : Task
        Tarea de MLPY
    complexity_preference : Complexity
        Preferencia de complejidad
    **preferences
        Preferencias adicionales
        
    Returns:
    --------
    ModelRecommendation : Mejor recomendación o None
    """
    selector = AutoModelSelector()
    recommendations = selector.recommend_models(
        task=task,
        top_k=1,
        complexity_preference=complexity_preference,
        **preferences
    )
    
    return recommendations[0] if recommendations else None


def recommend_models(
    task: Task,
    top_k: int = 3,
    **preferences
) -> List[ModelRecommendation]:
    """
    Recomendar múltiples modelos para una tarea.
    
    Parameters:
    -----------
    task : Task
        Tarea de MLPY
    top_k : int
        Número de recomendaciones
    **preferences
        Preferencias adicionales
        
    Returns:
    --------
    List[ModelRecommendation] : Lista de recomendaciones
    """
    selector = AutoModelSelector()
    return selector.recommend_models(task=task, top_k=top_k, **preferences)


__all__ = [
    'AutoModelSelector',
    'ModelRecommendation',
    'DataCharacteristics',
    'DatasetSize',
    'DatasetComplexity',
    'select_best_model',
    'recommend_models'
]