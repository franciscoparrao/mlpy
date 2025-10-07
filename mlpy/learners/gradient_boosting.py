"""Unified Gradient Boosting interface for MLPY.

This module provides a unified interface for gradient boosting models,
automatically selecting the best implementation based on data characteristics
and available libraries.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union, Literal
import warnings
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from .base import Learner
from .classification import LearnerClassif
from .regression import LearnerRegr
from ..tasks import Task, TaskClassif, TaskRegr
from ..utils.registry import mlpy_learners

# Check available backends
AVAILABLE_BACKENDS = []

try:
    from .xgboost_wrapper import LearnerXGBoost, LearnerXGBoostClassif, LearnerXGBoostRegr
    AVAILABLE_BACKENDS.append('xgboost')
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    from .lightgbm_wrapper import LearnerLightGBM, LearnerLightGBMClassif, LearnerLightGBMRegr
    AVAILABLE_BACKENDS.append('lightgbm')
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False

try:
    from .catboost_wrapper import LearnerCatBoost, LearnerCatBoostClassif, LearnerCatBoostRegr
    AVAILABLE_BACKENDS.append('catboost')
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False


@dataclass
class GBOptimizationProfile:
    """Optimization profile for gradient boosting."""
    use_gpu: bool = False
    gpu_device: Optional[int] = None
    handle_categorical: bool = True
    handle_missing: bool = True
    handle_text: bool = False
    distributed: bool = False
    n_workers: Optional[int] = None
    optimize_for: Literal['speed', 'accuracy', 'memory'] = 'accuracy'
    enable_uncertainty: bool = False


class GradientBoostingLearner(Learner):
    """Unified gradient boosting learner with automatic backend selection.
    
    This learner automatically selects the best gradient boosting implementation
    (XGBoost, LightGBM, or CatBoost) based on:
    - Data characteristics (size, categorical features, missing values)
    - Available hardware (GPU)
    - Optimization goals (speed vs accuracy)
    - Library availability
    
    Parameters
    ----------
    backend : str, default='auto'
        Which backend to use ('xgboost', 'lightgbm', 'catboost', 'auto').
        'auto' selects the best backend based on data and requirements.
    n_estimators : int, default=100
        Number of boosting rounds.
    max_depth : int, default=6
        Maximum tree depth.
    learning_rate : float, default=0.1
        Learning rate.
    objective : str, optional
        Objective function. If None, inferred from task.
    optimization_profile : GBOptimizationProfile, optional
        Optimization settings.
    auto_optimize : bool, default=True
        Whether to auto-optimize parameters based on data.
    verbose : bool, default=False
        Verbosity mode.
    **kwargs
        Additional parameters passed to the selected backend.
    """
    
    def __init__(
        self,
        backend: str = 'auto',
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        objective: Optional[str] = None,
        optimization_profile: Optional[GBOptimizationProfile] = None,
        auto_optimize: bool = True,
        verbose: bool = False,
        id: Optional[str] = None,
        **kwargs
    ):
        if not AVAILABLE_BACKENDS:
            raise ImportError(
                "No gradient boosting libraries available. "
                "Install at least one: pip install xgboost lightgbm catboost"
            )
            
        self.backend = backend
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.optimization_profile = optimization_profile or GBOptimizationProfile()
        self.auto_optimize = auto_optimize
        self.verbose = verbose
        self.gb_params = kwargs
        
        self._backend_learner = None
        self._selected_backend = None
        
        # Auto-generate ID
        if id is None:
            id = "gradient_boosting"
            
        super().__init__(
            id=id,
            properties={"importance", "prob", "shap"},
            packages={"xgboost", "lightgbm", "catboost"}
        )
        
    def _analyze_data(self, task: Task) -> Dict[str, Any]:
        """Analyze data characteristics to inform backend selection.
        
        Parameters
        ----------
        task : Task
            The task to analyze.
            
        Returns
        -------
        Dict[str, Any]
            Data characteristics.
        """
        # Get sample of data
        sample_size = min(1000, len(task.row_roles['use']))
        sample_rows = sorted(task.row_roles['use'])[:sample_size]
        X = task.data(rows=sample_rows, cols=task.feature_names, data_format='dataframe')
        
        analysis = {
            'n_samples': len(task.row_roles['use']),
            'n_features': len(task.feature_names),
            'has_categorical': False,
            'n_categorical': 0,
            'has_missing': False,
            'missing_ratio': 0.0,
            'has_text': False,
            'is_large': len(task.row_roles['use']) > 100000,
            'is_wide': len(task.feature_names) > 1000,
            'memory_estimate_mb': 0
        }
        
        if isinstance(X, pd.DataFrame):
            # Check for categorical features
            cat_cols = X.select_dtypes(include=['category', 'object']).columns
            analysis['has_categorical'] = len(cat_cols) > 0
            analysis['n_categorical'] = len(cat_cols)
            
            # Check for missing values
            analysis['has_missing'] = X.isnull().any().any()
            if analysis['has_missing']:
                analysis['missing_ratio'] = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
                
            # Check for text features (long strings)
            for col in X.select_dtypes(include=['object']).columns:
                avg_length = X[col].astype(str).str.len().mean()
                if avg_length > 50:  # Heuristic for text
                    analysis['has_text'] = True
                    break
                    
            # Estimate memory usage
            analysis['memory_estimate_mb'] = X.memory_usage(deep=True).sum() / 1024 / 1024
            
        return analysis
        
    def _select_backend(self, task: Task) -> str:
        """Select the best backend based on data and requirements.
        
        Parameters
        ----------
        task : Task
            The task to train on.
            
        Returns
        -------
        str
            Selected backend name.
        """
        if self.backend != 'auto':
            # User specified backend
            if self.backend not in AVAILABLE_BACKENDS:
                raise ValueError(
                    f"Backend '{self.backend}' not available. "
                    f"Available: {AVAILABLE_BACKENDS}"
                )
            return self.backend
            
        # Analyze data
        data_analysis = self._analyze_data(task)
        
        # Score each backend
        scores = {}
        
        # XGBoost scoring
        if 'xgboost' in AVAILABLE_BACKENDS:
            score = 100  # Base score
            
            # XGBoost strengths
            if self.optimization_profile.use_gpu:
                score += 20  # Good GPU support
            if data_analysis['is_large']:
                score += 10  # Good for large data
            if not data_analysis['has_categorical']:
                score += 10  # No native categorical support
                
            # XGBoost weaknesses
            if data_analysis['has_categorical']:
                score -= 20  # Needs encoding
            if data_analysis['has_text']:
                score -= 30  # No text support
                
            scores['xgboost'] = score
            
        # LightGBM scoring
        if 'lightgbm' in AVAILABLE_BACKENDS:
            score = 100  # Base score
            
            # LightGBM strengths
            if self.optimization_profile.optimize_for == 'speed':
                score += 30  # Fastest
            if data_analysis['is_large']:
                score += 20  # Very efficient with large data
            if data_analysis['has_categorical']:
                score += 15  # Good categorical support
            if self.optimization_profile.use_gpu:
                score += 15  # GPU support
            if data_analysis['is_wide']:
                score += 10  # Handles wide data well
                
            # LightGBM weaknesses
            if data_analysis['has_text']:
                score -= 30  # No text support
                
            scores['lightgbm'] = score
            
        # CatBoost scoring
        if 'catboost' in AVAILABLE_BACKENDS:
            score = 100  # Base score
            
            # CatBoost strengths
            if data_analysis['has_categorical']:
                score += 30  # Best categorical support
            if data_analysis['has_text']:
                score += 20  # Text feature support
            if self.optimization_profile.enable_uncertainty:
                score += 25  # Uncertainty quantification
            if self.optimization_profile.optimize_for == 'accuracy':
                score += 15  # Often most accurate
            if self.optimization_profile.use_gpu:
                score += 15  # Good GPU support
            if data_analysis['has_missing']:
                score += 10  # Handles missing values well
                
            # CatBoost weaknesses
            if self.optimization_profile.optimize_for == 'speed' and not self.optimization_profile.use_gpu:
                score -= 10  # Slower on CPU
                
            scores['catboost'] = score
            
        # Select backend with highest score
        selected = max(scores, key=scores.get)
        
        if self.verbose:
            logger.info(f"Backend selection scores: {scores}")
            logger.info(f"Selected backend: {selected}")
            logger.info(f"Data analysis: {data_analysis}")
            
        return selected
        
    def _create_backend_learner(self, backend: str, task: Task):
        """Create the appropriate backend learner.
        
        Parameters
        ----------
        backend : str
            Backend to use.
        task : Task
            The task to train on.
        """
        # Determine task type
        is_classification = isinstance(task, TaskClassif)
        
        # Prepare backend-specific parameters
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'objective': self.objective,
            **self.gb_params
        }
        
        # Add optimization-specific parameters
        if self.optimization_profile.use_gpu:
            if backend == 'xgboost':
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = self.optimization_profile.gpu_device or 0
            elif backend == 'lightgbm':
                params['device_type'] = 'gpu'
                params['gpu_device_id'] = self.optimization_profile.gpu_device or 0
            elif backend == 'catboost':
                params['task_type'] = 'GPU'
                if self.optimization_profile.gpu_device is not None:
                    params['devices'] = str(self.optimization_profile.gpu_device)
                    
        # Auto-optimization based on data analysis
        if self.auto_optimize:
            data_analysis = self._analyze_data(task)
            
            # Optimize for large datasets
            if data_analysis['is_large']:
                params['subsample'] = params.get('subsample', 0.8)
                if backend == 'lightgbm':
                    params['num_leaves'] = params.get('num_leaves', 63)
                    params['min_child_samples'] = params.get('min_child_samples', 100)
                    
            # Optimize for wide datasets
            if data_analysis['is_wide']:
                params['colsample_bytree'] = params.get('colsample_bytree', 0.8)
                if backend == 'lightgbm':
                    params['colsample_bylevel'] = params.get('colsample_bylevel', 0.8)
                    
            # Handle categorical features
            if data_analysis['has_categorical']:
                if backend == 'lightgbm':
                    params['categorical_features'] = 'auto'
                elif backend == 'catboost':
                    params['cat_features'] = 'auto'
                    
        # Create learner
        if backend == 'xgboost':
            if is_classification:
                self._backend_learner = LearnerXGBoostClassif(**params)
            else:
                self._backend_learner = LearnerXGBoostRegr(**params)
                
        elif backend == 'lightgbm':
            if is_classification:
                self._backend_learner = LearnerLightGBMClassif(**params)
            else:
                self._backend_learner = LearnerLightGBMRegr(**params)
                
        elif backend == 'catboost':
            # Add uncertainty if requested
            if self.optimization_profile.enable_uncertainty:
                params['posterior_sampling'] = True
                
            if is_classification:
                self._backend_learner = LearnerCatBoostClassif(**params)
            else:
                self._backend_learner = LearnerCatBoostRegr(**params)
                
        else:
            raise ValueError(f"Unknown backend: {backend}")
            
    def train(self, task: Task, row_ids: Optional[List[int]] = None):
        """Train the gradient boosting model.
        
        Parameters
        ----------
        task : Task
            The task to train on.
        row_ids : list of int, optional
            Subset of rows to use for training.
            
        Returns
        -------
        self
            The trained learner.
        """
        # Select backend if not already done
        if self._backend_learner is None:
            self._selected_backend = self._select_backend(task)
            self._create_backend_learner(self._selected_backend, task)
            
            if self.verbose:
                logger.info(f"Training with {self._selected_backend} backend")
                
        # Train the backend learner
        self._backend_learner.train(task, row_ids)
        
        return self
        
    def predict(self, task: Task, row_ids: Optional[List[int]] = None):
        """Make predictions.
        
        Parameters
        ----------
        task : Task
            The task to predict on.
        row_ids : list of int, optional
            Subset of rows to predict.
            
        Returns
        -------
        Prediction
            The predictions.
        """
        if self._backend_learner is None:
            raise RuntimeError("Model must be trained before prediction")
            
        return self._backend_learner.predict(task, row_ids)
        
    @property
    def model(self):
        """Access to the underlying model."""
        if self._backend_learner:
            return self._backend_learner.model
        return None
        
    @property
    def feature_importances(self) -> Optional[Dict[str, float]]:
        """Get feature importances."""
        if self._backend_learner:
            return self._backend_learner.feature_importances
        return None
        
    def get_shap_values(self, task: Task, row_ids: Optional[List[int]] = None):
        """Get SHAP values."""
        if self._backend_learner:
            return self._backend_learner.get_shap_values(task, row_ids)
        return None
        
    def plot_importance(self, **kwargs):
        """Plot feature importance."""
        if self._backend_learner:
            self._backend_learner.plot_importance(**kwargs)
            
    def clone(self, deep: bool = True):
        """Create a copy of the learner."""
        new_learner = self.__class__(
            backend=self.backend,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective=self.objective,
            optimization_profile=self.optimization_profile,
            auto_optimize=self.auto_optimize,
            verbose=self.verbose,
            id=self.id,
            **self.gb_params
        )
        
        if deep and self._backend_learner:
            new_learner._backend_learner = self._backend_learner.clone(deep=True)
            new_learner._selected_backend = self._selected_backend
            
        return new_learner
        
    def reset(self):
        """Reset the learner."""
        self._backend_learner = None
        self._selected_backend = None
        return self
        
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the selected backend.
        
        Returns
        -------
        Dict[str, Any]
            Backend information.
        """
        return {
            'available_backends': AVAILABLE_BACKENDS,
            'selected_backend': self._selected_backend,
            'optimization_profile': self.optimization_profile.__dict__ if self.optimization_profile else None,
            'is_trained': self.is_trained
        }


class GradientBoostingClassif(GradientBoostingLearner, LearnerClassif):
    """Unified gradient boosting classifier."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class GradientBoostingRegr(GradientBoostingLearner, LearnerRegr):
    """Unified gradient boosting regressor."""
    
    def __init__(self, **kwargs):
        super().__init__(predict_type="response", **kwargs)


# Register learners
mlpy_learners.register("gradient_boosting", GradientBoostingLearner)
mlpy_learners.register("gradient_boosting.classif", GradientBoostingClassif)
mlpy_learners.register("gradient_boosting.regr", GradientBoostingRegr)
mlpy_learners.register("gb", GradientBoostingLearner)  # Alias
mlpy_learners.register("gb.classif", GradientBoostingClassif)  # Alias
mlpy_learners.register("gb.regr", GradientBoostingRegr)  # Alias


def learner_gradient_boosting(**kwargs) -> Union[GradientBoostingClassif, GradientBoostingRegr]:
    """Create a gradient boosting learner with automatic backend selection.
    
    This function creates a unified gradient boosting learner that automatically
    selects the best backend (XGBoost, LightGBM, or CatBoost) based on your data
    and requirements.
    
    Parameters
    ----------
    **kwargs
        Parameters for gradient boosting.
        
    Returns
    -------
    GradientBoostingLearner
        A gradient boosting learner.
        
    Examples
    --------
    >>> from mlpy.learners import learner_gradient_boosting
    >>> 
    >>> # Auto-select best backend
    >>> gb = learner_gradient_boosting(n_estimators=100)
    >>> 
    >>> # Force specific backend
    >>> gb_xgb = learner_gradient_boosting(backend='xgboost', n_estimators=100)
    >>> 
    >>> # Optimize for speed
    >>> from mlpy.learners.gradient_boosting import GBOptimizationProfile
    >>> profile = GBOptimizationProfile(optimize_for='speed', use_gpu=True)
    >>> gb_fast = learner_gradient_boosting(optimization_profile=profile)
    >>> 
    >>> # Enable uncertainty quantification
    >>> profile_unc = GBOptimizationProfile(enable_uncertainty=True)
    >>> gb_unc = learner_gradient_boosting(optimization_profile=profile_unc)
    """
    # Determine type from objective if provided
    objective = kwargs.get('objective', None)
    
    if objective:
        # Classification objectives
        classif_objectives = ['binary', 'multiclass', 'logloss', 'classification']
        if any(obj in objective.lower() for obj in classif_objectives):
            return GradientBoostingClassif(**kwargs)
        else:
            return GradientBoostingRegr(**kwargs)
    else:
        # Return base class that will determine type during training
        return GradientBoostingLearner(**kwargs)