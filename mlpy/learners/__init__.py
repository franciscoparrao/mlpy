"""Learners module for MLPY.

This module provides base classes and implementations for machine learning
algorithms (learners) that can be trained and used for prediction.
"""

from .base import Learner
from .classification import LearnerClassif
from .regression import LearnerRegr
from .baseline import (
    LearnerClassifFeatureless,
    LearnerRegrFeatureless,
    LearnerClassifDebug,
    LearnerRegrDebug
)
from .ensemble import (
    LearnerEnsemble,
    LearnerVoting,
    LearnerStacking,
    LearnerBlending,
    create_ensemble
)

# Import sklearn wrappers if available
try:
    from .sklearn_wrapper import (
        LearnerSklearn,
        LearnerClassifSklearn,
        LearnerRegrSklearn,
        learner_sklearn
    )
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# Import time series learners if available
try:
    from .timeseries import (
        LearnerTimeSeriesBase,
        LearnerARIMA,
        LearnerProphet,
        LearnerExponentialSmoothing,
        learner_arima,
        learner_sarima,
        learner_prophet,
        learner_exponential_smoothing
    )
    _HAS_TIMESERIES = True
except ImportError:
    _HAS_TIMESERIES = False

# Import TGPY wrapper if available
try:
    from .tgpy_wrapper import LearnerTGPRegressor, LearnerTGPClassifier
    _HAS_TGPY = True
except ImportError:
    _HAS_TGPY = False

# Import XGBoost wrapper if available
try:
    from .xgboost_wrapper import (
        LearnerXGBoostClassif,
        LearnerXGBoostRegr,
        learner_xgboost
    )
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

# Import LightGBM wrapper if available
try:
    from .lightgbm_wrapper import (
        LearnerLightGBMClassif,
        LearnerLightGBMRegr,
        learner_lightgbm
    )
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

# Import CatBoost wrapper if available
try:
    from .catboost_wrapper import (
        LearnerCatBoostClassif,
        LearnerCatBoostRegr,
        learner_catboost
    )
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

__all__ = [
    "Learner",
    "LearnerClassif",
    "LearnerRegr",
    "LearnerClassifFeatureless",
    "LearnerRegrFeatureless", 
    "LearnerClassifDebug",
    "LearnerRegrDebug",
    "LearnerEnsemble",
    "LearnerVoting",
    "LearnerStacking",
    "LearnerBlending",
    "create_ensemble"
]

# Add sklearn exports if available
if _HAS_SKLEARN:
    __all__.extend([
        "LearnerSklearn",
        "LearnerClassifSklearn",
        "LearnerRegrSklearn",
        "learner_sklearn"
    ])

# Add time series exports if available
if _HAS_TIMESERIES:
    __all__.extend([
        "LearnerTimeSeriesBase",
        "LearnerARIMA",
        "LearnerProphet", 
        "LearnerExponentialSmoothing",
        "learner_arima",
        "learner_sarima",
        "learner_prophet",
        "learner_exponential_smoothing"
    ])

# Add TGPY exports if available
if _HAS_TGPY:
    __all__.extend([
        "LearnerTGPRegressor",
        "LearnerTGPClassifier"
    ])

# Add XGBoost exports if available
if _HAS_XGBOOST:
    __all__.extend([
        "LearnerXGBoostClassif",
        "LearnerXGBoostRegr",
        "learner_xgboost"
    ])

# Add LightGBM exports if available
if _HAS_LIGHTGBM:
    __all__.extend([
        "LearnerLightGBMClassif",
        "LearnerLightGBMRegr",
        "learner_lightgbm"
    ])

# Add CatBoost exports if available
if _HAS_CATBOOST:
    __all__.extend([
        "LearnerCatBoostClassif",
        "LearnerCatBoostRegr",
        "learner_catboost"
    ])