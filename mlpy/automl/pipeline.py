"""
AutoML Pipeline Management
==========================

Automated pipeline construction and optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class PipelineStep:
    """Represents a pipeline step with hyperparameters."""
    
    def __init__(
        self,
        name: str,
        estimator_class: type,
        param_space: Dict[str, Any],
        is_optional: bool = False
    ):
        """
        Initialize pipeline step.
        
        Args:
            name: Step name
            estimator_class: Class of the estimator
            param_space: Parameter search space
            is_optional: Whether step is optional
        """
        self.name = name
        self.estimator_class = estimator_class
        self.param_space = param_space
        self.is_optional = is_optional
    
    def create_estimator(self, params: Dict[str, Any]) -> Any:
        """Create estimator with parameters."""
        return self.estimator_class(**params)


class AutoPipeline:
    """Automated pipeline construction."""
    
    def __init__(
        self,
        task_type: str = "classification",
        include_preprocessing: bool = True,
        include_feature_selection: bool = True,
        include_dimensionality_reduction: bool = False
    ):
        """
        Initialize auto pipeline.
        
        Args:
            task_type: "classification" or "regression"
            include_preprocessing: Include preprocessing steps
            include_feature_selection: Include feature selection
            include_dimensionality_reduction: Include PCA/dimensionality reduction
        """
        self.task_type = task_type
        self.include_preprocessing = include_preprocessing
        self.include_feature_selection = include_feature_selection
        self.include_dimensionality_reduction = include_dimensionality_reduction
        
        self.pipeline_steps = []
        self.best_pipeline = None
    
    def get_pipeline_steps(self, n_features: int, n_samples: int) -> List[PipelineStep]:
        """
        Get pipeline steps based on data characteristics.
        
        Args:
            n_features: Number of features
            n_samples: Number of samples
            
        Returns:
            List of pipeline steps
        """
        steps = []
        
        # Imputation step
        if self.include_preprocessing:
            steps.append(PipelineStep(
                name="imputer",
                estimator_class=SimpleImputer,
                param_space={
                    "strategy": ["mean", "median", "most_frequent"]
                },
                is_optional=False
            ))
            
            # Scaling step
            steps.append(PipelineStep(
                name="scaler",
                estimator_class=None,  # Will be selected
                param_space={
                    "method": ["standard", "minmax", "robust", "none"]
                },
                is_optional=True
            ))
        
        # Feature selection
        if self.include_feature_selection:
            if n_features > 50:
                steps.append(PipelineStep(
                    name="feature_selection",
                    estimator_class=SelectKBest,
                    param_space={
                        "k": [10, 20, 30, 40, 50, "all"]
                    },
                    is_optional=True
                ))
        
        # Dimensionality reduction
        if self.include_dimensionality_reduction:
            if n_features > 100:
                steps.append(PipelineStep(
                    name="dim_reduction",
                    estimator_class=PCA,
                    param_space={
                        "n_components": [0.8, 0.9, 0.95, 0.99]
                    },
                    is_optional=True
                ))
        
        return steps
    
    def create_pipeline(
        self,
        steps: List[Tuple[str, Any]],
        model: Any
    ) -> Pipeline:
        """
        Create sklearn pipeline.
        
        Args:
            steps: List of (name, estimator) tuples
            model: Final model
            
        Returns:
            Sklearn Pipeline
        """
        all_steps = steps + [("model", model)]
        return Pipeline(all_steps)
    
    def optimize_pipeline(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        n_trials: int = 20
    ) -> Pipeline:
        """
        Optimize pipeline configuration.
        
        Args:
            X: Features
            y: Target
            model: Model to use
            n_trials: Number of configurations to try
            
        Returns:
            Best pipeline
        """
        from sklearn.model_selection import cross_val_score
        
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        # Get possible steps
        possible_steps = self.get_pipeline_steps(n_features, n_samples)
        
        best_score = -np.inf
        best_pipeline = None
        
        # Try different pipeline configurations
        for _ in range(n_trials):
            steps = []
            
            for step in possible_steps:
                if step.is_optional and np.random.random() < 0.5:
                    continue
                
                if step.name == "scaler":
                    # Select scaler type
                    method = np.random.choice(step.param_space["method"])
                    if method == "standard":
                        steps.append(("scaler", StandardScaler()))
                    elif method == "minmax":
                        steps.append(("scaler", MinMaxScaler()))
                    elif method == "robust":
                        steps.append(("scaler", RobustScaler()))
                    # else "none" - don't add scaler
                    
                elif step.name == "feature_selection":
                    k = np.random.choice(step.param_space["k"])
                    if k != "all":
                        steps.append(("feature_selection", SelectKBest(k=k)))
                        
                elif step.name == "dim_reduction":
                    n_components = np.random.choice(step.param_space["n_components"])
                    steps.append(("dim_reduction", PCA(n_components=n_components)))
                    
                else:
                    # Simple step with random params
                    params = {}
                    for param_name, param_values in step.param_space.items():
                        params[param_name] = np.random.choice(param_values)
                    
                    estimator = step.create_estimator(params)
                    steps.append((step.name, estimator))
            
            # Create and evaluate pipeline
            pipeline = self.create_pipeline(steps, model)
            
            try:
                # Quick cross-validation
                scores = cross_val_score(
                    pipeline, X, y,
                    cv=3,
                    scoring='accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'
                )
                score = scores.mean()
                
                if score > best_score:
                    best_score = score
                    best_pipeline = pipeline
                    
            except Exception as e:
                logger.warning(f"Pipeline evaluation failed: {e}")
                continue
        
        self.best_pipeline = best_pipeline
        return best_pipeline


class PipelineOptimizer:
    """Optimize entire ML pipelines."""
    
    def __init__(
        self,
        task_type: str = "classification",
        time_budget: int = 3600,
        n_jobs: int = -1
    ):
        """
        Initialize pipeline optimizer.
        
        Args:
            task_type: Task type
            time_budget: Time budget in seconds
            n_jobs: Number of parallel jobs
        """
        self.task_type = task_type
        self.time_budget = time_budget
        self.n_jobs = n_jobs
        
        self.pipelines_evaluated = []
        self.best_pipeline = None
        self.best_score = -np.inf
    
    def create_pipeline_space(self) -> Dict[str, Any]:
        """Create search space for pipelines."""
        space = {
            # Preprocessing
            "imputation": ["mean", "median", "most_frequent"],
            "scaling": ["standard", "minmax", "robust", "none"],
            
            # Feature engineering
            "polynomial_features": [False, True],
            "polynomial_degree": [2, 3],
            
            # Feature selection
            "feature_selection": ["none", "kbest", "variance", "model_based"],
            "n_features_to_select": [10, 20, 30, 50, 100, "all"],
            
            # Dimensionality reduction
            "dim_reduction": ["none", "pca", "svd"],
            "n_components": [0.8, 0.9, 0.95],
            
            # Model selection
            "model": ["rf", "gb", "xgb", "svm", "lr"]
        }
        
        return space
    
    def build_pipeline(self, config: Dict[str, Any]) -> Pipeline:
        """Build pipeline from configuration."""
        steps = []
        
        # Imputation
        if config.get("imputation"):
            steps.append(("imputer", SimpleImputer(strategy=config["imputation"])))
        
        # Scaling
        if config.get("scaling") and config["scaling"] != "none":
            if config["scaling"] == "standard":
                steps.append(("scaler", StandardScaler()))
            elif config["scaling"] == "minmax":
                steps.append(("scaler", MinMaxScaler()))
            elif config["scaling"] == "robust":
                steps.append(("scaler", RobustScaler()))
        
        # Polynomial features
        if config.get("polynomial_features"):
            from sklearn.preprocessing import PolynomialFeatures
            degree = config.get("polynomial_degree", 2)
            steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
        
        # Feature selection
        if config.get("feature_selection") and config["feature_selection"] != "none":
            n_features = config.get("n_features_to_select", "all")
            
            if config["feature_selection"] == "kbest":
                if n_features != "all":
                    steps.append(("feature_selection", SelectKBest(k=n_features)))
            elif config["feature_selection"] == "variance":
                steps.append(("feature_selection", VarianceThreshold()))
        
        # Dimensionality reduction
        if config.get("dim_reduction") and config["dim_reduction"] != "none":
            n_components = config.get("n_components", 0.95)
            
            if config["dim_reduction"] == "pca":
                steps.append(("dim_reduction", PCA(n_components=n_components)))
            elif config["dim_reduction"] == "svd":
                from sklearn.decomposition import TruncatedSVD
                n_comp = int(n_components * 100) if n_components < 1 else int(n_components)
                steps.append(("dim_reduction", TruncatedSVD(n_components=n_comp)))
        
        # Model
        model = self._get_model(config.get("model", "rf"))
        steps.append(("model", model))
        
        return Pipeline(steps)
    
    def _get_model(self, model_name: str) -> Any:
        """Get model by name."""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        if self.task_type == "classification":
            if model_name == "rf":
                return RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_name == "gb":
                return GradientBoostingClassifier(n_estimators=100, random_state=42)
            elif model_name == "svm":
                return SVC(probability=True, random_state=42)
            elif model_name == "lr":
                return LogisticRegression(max_iter=1000, random_state=42)
            else:
                return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import Ridge
            from sklearn.svm import SVR
            
            if model_name == "rf":
                return RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_name == "gb":
                return GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif model_name == "svm":
                return SVR()
            elif model_name == "lr":
                return Ridge()
            else:
                return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50
    ) -> Pipeline:
        """
        Optimize pipeline.
        
        Args:
            X: Features
            y: Target
            n_trials: Number of trials
            
        Returns:
            Best pipeline
        """
        from sklearn.model_selection import cross_val_score
        import time
        
        start_time = time.time()
        space = self.create_pipeline_space()
        
        for trial in range(n_trials):
            if time.time() - start_time > self.time_budget:
                break
            
            # Sample configuration
            config = {}
            for param, values in space.items():
                config[param] = np.random.choice(values)
            
            # Build pipeline
            try:
                pipeline = self.build_pipeline(config)
                
                # Evaluate
                scores = cross_val_score(
                    pipeline, X, y,
                    cv=3,
                    scoring='accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error',
                    n_jobs=self.n_jobs
                )
                score = scores.mean()
                
                self.pipelines_evaluated.append({
                    'config': config,
                    'score': score,
                    'pipeline': pipeline
                })
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_pipeline = pipeline
                    logger.info(f"New best pipeline found: score={score:.4f}")
                    
            except Exception as e:
                logger.warning(f"Pipeline failed: {e}")
                continue
        
        return self.best_pipeline