"""
Advanced AutoML Learner
========================

Main AutoML engine with intelligent search and optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
import time
import warnings
import logging
from datetime import datetime
import json
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)

logger = logging.getLogger(__name__)


@dataclass
class AutoMLConfig:
    """Configuration for AutoML."""
    
    task_type: str = "auto"  # "classification", "regression", "auto"
    time_budget: int = 3600  # seconds
    max_trials: int = 100
    optimization_metric: Optional[str] = None
    validation_strategy: str = "holdout"  # "holdout", "cv"
    test_size: float = 0.2
    n_folds: int = 5
    ensemble_size: int = 5
    
    # Feature engineering
    auto_feature_engineering: bool = True
    max_features_generated: int = 50
    feature_selection: bool = True
    feature_selection_threshold: float = 0.01
    
    # Model search
    include_models: Optional[List[str]] = None
    exclude_models: Optional[List[str]] = None
    include_neural: bool = False
    
    # Optimization
    optimizer: str = "bayesian"  # "random", "grid", "bayesian", "evolutionary"
    n_jobs: int = -1
    random_state: int = 42
    verbose: int = 1
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_improvement: float = 0.001
    
    # Memory and compute
    memory_limit: Optional[int] = None  # MB
    gpu_enabled: bool = False
    
    # Output
    save_models: bool = True
    save_path: str = "./automl_output"
    return_all_models: bool = False


@dataclass
class ModelPerformance:
    """Store model performance metrics."""
    
    model_name: str
    parameters: Dict[str, Any]
    train_score: float
    val_score: float
    test_score: Optional[float] = None
    training_time: float = 0.0
    prediction_time: float = 0.0
    memory_usage: float = 0.0
    feature_importance: Optional[Dict[str, float]] = None
    cross_val_scores: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'parameters': self.parameters,
            'train_score': self.train_score,
            'val_score': self.val_score,
            'test_score': self.test_score,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'memory_usage': self.memory_usage
        }


@dataclass
class AutoMLResults:
    """Results from AutoML run."""
    
    best_model: Any
    best_score: float
    best_parameters: Dict[str, Any]
    all_models: List[ModelPerformance]
    feature_importance: Dict[str, float]
    generated_features: Optional[List[str]] = None
    selected_features: Optional[List[str]] = None
    preprocessing_pipeline: Optional[Any] = None
    ensemble_models: Optional[List[Any]] = None
    total_time: float = 0.0
    n_trials_completed: int = 0
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def summary(self) -> pd.DataFrame:
        """Get summary of all models."""
        data = [m.to_dict() for m in self.all_models]
        df = pd.DataFrame(data)
        return df.sort_values('val_score', ascending=False)
    
    def save(self, path: str):
        """Save results to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save models
        with open(f"{path}/best_model.pkl", 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save results
        results_dict = {
            'best_score': self.best_score,
            'best_parameters': self.best_parameters,
            'feature_importance': self.feature_importance,
            'generated_features': self.generated_features,
            'selected_features': self.selected_features,
            'total_time': self.total_time,
            'n_trials_completed': self.n_trials_completed
        }
        
        with open(f"{path}/results.json", 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save summary
        self.summary().to_csv(f"{path}/model_summary.csv", index=False)


class AutoMLearner:
    """Advanced AutoML with intelligent search."""
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        """
        Initialize AutoML.
        
        Args:
            config: AutoML configuration
        """
        self.config = config or AutoMLConfig()
        self.results = None
        self.start_time = None
        self.trials_completed = 0
        self.best_score = -np.inf
        self.best_model = None
        self.optimization_history = []
        
        # Components (will be initialized in fit)
        self.optimizer = None
        self.feature_engineer = None
        self.model_selector = None
        self.ensemble = None
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        if self.config.verbose > 0:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - AutoML - %(message)s'
            )
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> 'AutoMLearner':
        """
        Fit AutoML on data.
        
        Args:
            X: Training features
            y: Training target
            X_test: Optional test features
            y_test: Optional test target
            
        Returns:
            Self
        """
        self.start_time = time.time()
        
        # Convert to numpy if needed
        X = self._to_numpy(X)
        y = self._to_numpy(y).ravel()
        
        if X_test is not None:
            X_test = self._to_numpy(X_test)
            y_test = self._to_numpy(y_test).ravel() if y_test is not None else None
        
        # Detect task type
        if self.config.task_type == "auto":
            self.config.task_type = self._detect_task_type(y)
            logger.info(f"Detected task type: {self.config.task_type}")
        
        # Set optimization metric
        if self.config.optimization_metric is None:
            self.config.optimization_metric = self._get_default_metric()
        
        # Split data
        if X_test is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y if self.config.task_type == "classification" else None
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = X_test, y_test
        
        # Initialize components
        self._initialize_components(X_train, y_train)
        
        # Preprocessing
        X_train_processed, X_val_processed = self._preprocess_data(
            X_train, X_val, y_train
        )
        
        # Feature engineering
        if self.config.auto_feature_engineering:
            X_train_processed, X_val_processed = self._engineer_features(
                X_train_processed, X_val_processed, y_train
            )
        
        # Model search
        all_models = []
        
        while self._should_continue():
            # Get next configuration
            model_config = self.optimizer.get_next_config()
            
            if model_config is None:
                break
            
            # Train and evaluate model
            performance = self._train_and_evaluate(
                model_config,
                X_train_processed, y_train,
                X_val_processed, y_val
            )
            
            all_models.append(performance)
            
            # Update best model
            if performance.val_score > self.best_score:
                self.best_score = performance.val_score
                self.best_model = model_config['model']
                logger.info(f"New best model: {performance.model_name} "
                          f"(score: {performance.val_score:.4f})")
            
            # Update optimizer
            self.optimizer.update(model_config, performance.val_score)
            
            self.trials_completed += 1
            
            # Early stopping
            if self._check_early_stopping():
                logger.info("Early stopping triggered")
                break
        
        # Build ensemble
        if self.config.ensemble_size > 1:
            self.ensemble = self._build_ensemble(
                all_models[:self.config.ensemble_size],
                X_train_processed, y_train
            )
        
        # Create results
        self.results = AutoMLResults(
            best_model=self.best_model,
            best_score=self.best_score,
            best_parameters=self.optimizer.best_params,
            all_models=all_models,
            feature_importance=self._get_feature_importance(),
            generated_features=getattr(self.feature_engineer, 'generated_features', None),
            selected_features=getattr(self.feature_engineer, 'selected_features', None),
            preprocessing_pipeline=self.preprocessor,
            ensemble_models=self.ensemble,
            total_time=time.time() - self.start_time,
            n_trials_completed=self.trials_completed,
            optimization_history=self.optimization_history
        )
        
        # Save results
        if self.config.save_models:
            self.results.save(self.config.save_path)
        
        logger.info(f"AutoML completed in {self.results.total_time:.2f} seconds")
        logger.info(f"Best score: {self.best_score:.4f}")
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if self.results is None:
            raise ValueError("Model not fitted yet")
        
        X = self._to_numpy(X)
        X = self.preprocessor.transform(X)
        
        if self.feature_engineer:
            X = self.feature_engineer.transform(X)
        
        if self.ensemble:
            predictions = []
            for model in self.ensemble:
                predictions.append(model.predict(X))
            return np.mean(predictions, axis=0)
        else:
            return self.best_model.predict(X)
    
    def _to_numpy(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        return data
    
    def _detect_task_type(self, y: np.ndarray) -> str:
        """Detect if task is classification or regression."""
        unique_values = len(np.unique(y))
        
        # Check if values are integers
        is_integer = np.all(y == y.astype(int))
        
        if is_integer and unique_values < 20:
            return "classification"
        else:
            return "regression"
    
    def _get_default_metric(self) -> str:
        """Get default optimization metric."""
        if self.config.task_type == "classification":
            return "accuracy"
        else:
            return "neg_mean_squared_error"
    
    def _initialize_components(self, X: np.ndarray, y: np.ndarray):
        """Initialize AutoML components."""
        # Import here to avoid circular imports
        from .optimizers import get_optimizer
        from .feature_engineering import AutoFeatureEngineer
        from .search_spaces import get_default_search_space
        
        # Initialize optimizer
        search_space = get_default_search_space(
            task_type=self.config.task_type,
            n_features=X.shape[1],
            n_samples=X.shape[0]
        )
        
        self.optimizer = get_optimizer(
            self.config.optimizer,
            search_space,
            n_trials=self.config.max_trials
        )
        
        # Initialize feature engineer
        if self.config.auto_feature_engineering:
            self.feature_engineer = AutoFeatureEngineer(
                max_features=self.config.max_features_generated,
                selection_threshold=self.config.feature_selection_threshold
            )
        
        # Initialize preprocessor
        self.preprocessor = StandardScaler()
    
    def _preprocess_data(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data."""
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        
        # Scale features
        X_train = self.preprocessor.fit_transform(X_train)
        X_val = self.preprocessor.transform(X_val)
        
        return X_train, X_val
    
    def _engineer_features(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Engineer features."""
        if self.feature_engineer:
            X_train = self.feature_engineer.fit_transform(X_train, y_train)
            X_val = self.feature_engineer.transform(X_val)
        
        return X_train, X_val
    
    def _train_and_evaluate(
        self,
        model_config: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> ModelPerformance:
        """Train and evaluate a model."""
        model = model_config['model']
        params = model_config['params']
        
        # Set parameters
        model.set_params(**params)
        
        # Train
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Calculate scores
        if self.config.task_type == "classification":
            train_score = accuracy_score(y_train, train_pred)
            val_score = accuracy_score(y_val, val_pred)
        else:
            train_score = -mean_squared_error(y_train, train_pred)
            val_score = -mean_squared_error(y_val, val_pred)
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(enumerate(model.feature_importances_))
        
        return ModelPerformance(
            model_name=model.__class__.__name__,
            parameters=params,
            train_score=train_score,
            val_score=val_score,
            training_time=train_time,
            feature_importance=feature_importance
        )
    
    def _should_continue(self) -> bool:
        """Check if should continue optimization."""
        # Check time budget
        if time.time() - self.start_time > self.config.time_budget:
            logger.info("Time budget exceeded")
            return False
        
        # Check max trials
        if self.trials_completed >= self.config.max_trials:
            logger.info("Max trials reached")
            return False
        
        return True
    
    def _check_early_stopping(self) -> bool:
        """Check early stopping criteria."""
        if not self.config.early_stopping:
            return False
        
        if len(self.optimization_history) < self.config.patience:
            return False
        
        recent_scores = [h['score'] for h in self.optimization_history[-self.config.patience:]]
        improvement = max(recent_scores) - min(recent_scores)
        
        return improvement < self.config.min_improvement
    
    def _build_ensemble(
        self,
        models: List[ModelPerformance],
        X: np.ndarray,
        y: np.ndarray
    ) -> List[Any]:
        """Build ensemble from top models."""
        ensemble = []
        for perf in models:
            model = perf.model_name
            # Retrain on full data
            model.fit(X, y)
            ensemble.append(model)
        return ensemble
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance."""
        if not self.best_model or not hasattr(self.best_model, 'feature_importances_'):
            return {}
        
        importance = self.best_model.feature_importances_
        return dict(enumerate(importance))