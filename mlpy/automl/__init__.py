"""AutoML components for MLPY.

This module provides automated machine learning functionality including:
- Hyperparameter tuning
- Feature engineering
- Model selection
"""

from .tuning import (
    TunerGrid,
    TunerRandom,
    ParamSet,
    ParamInt,
    ParamFloat,
    ParamCategorical,
    TuneResult
)
from .feature_engineering import (
    AutoFeaturesNumeric,
    AutoFeaturesCategorical,
    AutoFeaturesInteraction
)
from .simple_automl import (
    SimpleAutoML,
    AutoMLResult
)

__all__ = [
    # Tuning
    "TunerGrid",
    "TunerRandom",
    "ParamSet",
    "ParamInt",
    "ParamFloat",
    "ParamCategorical",
    "TuneResult",
    # Feature Engineering
    "AutoFeaturesNumeric",
    "AutoFeaturesCategorical",
    "AutoFeaturesInteraction",
    # Simple AutoML
    "SimpleAutoML",
    "AutoMLResult"
]