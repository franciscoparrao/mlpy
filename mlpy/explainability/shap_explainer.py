"""
SHAP (SHapley Additive exPlanations) Integration
================================================

Unified framework for interpreting predictions using game theory.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Install with: pip install shap")


@dataclass
class SHAPResults:
    """Container for SHAP analysis results."""
    
    values: np.ndarray  # SHAP values
    expected_value: Union[float, np.ndarray]  # Expected value(s)
    feature_names: List[str]
    data: Optional[np.ndarray] = None
    interaction_values: Optional[np.ndarray] = None
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from SHAP values."""
        importance = np.abs(self.values).mean(axis=0)
        
        if len(importance.shape) > 1:  # Multi-class
            importance = importance.mean(axis=1)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        return df.sort_values('importance', ascending=False)
    
    def get_top_features(self, n: int = 10) -> List[str]:
        """Get top n most important features."""
        importance_df = self.get_feature_importance()
        return importance_df.head(n)['feature'].tolist()


class SHAPExplainer:
    """SHAP-based model explainer."""
    
    def __init__(
        self,
        model: Any,
        data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        task_type: str = "classification",
        explainer_type: str = "auto"
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model to explain
            data: Background data for SHAP
            feature_names: Names of features
            task_type: "classification" or "regression"
            explainer_type: "tree", "kernel", "linear", "deep", or "auto"
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
        
        self.model = model
        self.data = data
        self.feature_names = feature_names or self._infer_feature_names(data)
        self.task_type = task_type
        self.explainer_type = explainer_type
        self.explainer = None
        self._setup_explainer()
    
    def _infer_feature_names(self, data: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        """Infer feature names from data."""
        if isinstance(data, pd.DataFrame):
            return data.columns.tolist()
        elif isinstance(data, np.ndarray):
            return [f"feature_{i}" for i in range(data.shape[1])]
        else:
            return []
    
    def _setup_explainer(self):
        """Setup appropriate SHAP explainer based on model type."""
        if self.explainer_type == "auto":
            self.explainer_type = self._detect_explainer_type()
        
        if self.data is None and self.explainer_type != "tree":
            raise ValueError("Background data required for non-tree explainers")
        
        # Convert data to numpy if needed
        background_data = self.data
        if isinstance(background_data, pd.DataFrame):
            background_data = background_data.values
        
        # Create appropriate explainer
        if self.explainer_type == "tree":
            # For tree-based models (RF, XGBoost, LightGBM)
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except:
                logger.warning("TreeExplainer failed, falling back to KernelExplainer")
                self.explainer_type = "kernel"
                
        if self.explainer_type == "kernel":
            # Model-agnostic but slower
            if background_data is not None:
                # Sample background data if too large
                if len(background_data) > 100:
                    indices = np.random.choice(len(background_data), 100, replace=False)
                    background_data = background_data[indices]
                
                # Create prediction function
                if hasattr(self.model, 'predict_proba'):
                    predict_fn = self.model.predict_proba
                else:
                    predict_fn = self.model.predict
                
                self.explainer = shap.KernelExplainer(predict_fn, background_data)
                
        elif self.explainer_type == "linear":
            # For linear models
            self.explainer = shap.LinearExplainer(self.model, background_data)
            
        elif self.explainer_type == "deep":
            # For deep learning models
            self.explainer = shap.DeepExplainer(self.model, background_data)
    
    def _detect_explainer_type(self) -> str:
        """Detect appropriate explainer type based on model."""
        model_class = self.model.__class__.__name__.lower()
        
        # Tree-based models
        tree_models = ['forest', 'tree', 'xgb', 'lgb', 'catboost', 'gradient']
        if any(tree in model_class for tree in tree_models):
            return "tree"
        
        # Linear models
        linear_models = ['linear', 'logistic', 'ridge', 'lasso', 'elastic']
        if any(linear in model_class for linear in linear_models):
            return "linear"
        
        # Deep learning models
        deep_models = ['neural', 'mlp', 'dense', 'sequential', 'torch', 'tensorflow']
        if any(deep in model_class for deep in deep_models):
            return "deep"
        
        # Default to kernel (works with any model)
        return "kernel"
    
    def explain(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        check_additivity: bool = False
    ) -> SHAPResults:
        """
        Generate SHAP explanations for samples.
        
        Args:
            X: Data to explain
            check_additivity: Whether to check SHAP additivity
            
        Returns:
            SHAPResults object containing SHAP values and metadata
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_array)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            # Multi-class: stack along last dimension
            shap_values = np.stack(shap_values, axis=-1)
        
        # Get expected value
        expected_value = self.explainer.expected_value
        
        # Check additivity if requested
        if check_additivity:
            self._check_additivity(X_array, shap_values, expected_value)
        
        return SHAPResults(
            values=shap_values,
            expected_value=expected_value,
            feature_names=self.feature_names,
            data=X_array
        )
    
    def explain_instance(
        self,
        instance: Union[pd.Series, np.ndarray],
        show_plot: bool = False
    ) -> Dict[str, float]:
        """
        Explain a single instance.
        
        Args:
            instance: Single sample to explain
            show_plot: Whether to display waterfall plot
            
        Returns:
            Dictionary of feature contributions
        """
        if isinstance(instance, pd.Series):
            instance = instance.values
        
        instance = instance.reshape(1, -1)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(instance)[0]
        
        # Create contribution dictionary
        contributions = {
            feature: value 
            for feature, value in zip(self.feature_names, shap_values)
        }
        
        # Add baseline
        contributions['__baseline__'] = float(self.explainer.expected_value)
        
        if show_plot and SHAP_AVAILABLE:
            shap.waterfall_plot(shap.Explanation(
                values=shap_values,
                base_values=self.explainer.expected_value,
                data=instance[0],
                feature_names=self.feature_names
            ))
        
        return contributions
    
    def get_interaction_values(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Calculate SHAP interaction values.
        
        Args:
            X: Data to calculate interactions for
            
        Returns:
            Interaction values array
        """
        if self.explainer_type != "tree":
            raise ValueError("Interaction values only available for tree explainers")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        interaction_values = self.explainer.shap_interaction_values(X)
        return interaction_values
    
    def _check_additivity(
        self,
        X: np.ndarray,
        shap_values: np.ndarray,
        expected_value: Union[float, np.ndarray]
    ):
        """Check SHAP additivity property."""
        # Predict using model
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(X)
            if len(predictions.shape) > 1:
                predictions = predictions[:, 1]  # Binary classification
        else:
            predictions = self.model.predict(X)
        
        # Calculate SHAP predictions
        if len(shap_values.shape) == 3:  # Multi-class
            shap_predictions = expected_value[1] + shap_values[:, :, 1].sum(axis=1)
        else:
            shap_predictions = expected_value + shap_values.sum(axis=1)
        
        # Check difference
        diff = np.abs(predictions - shap_predictions).mean()
        if diff > 0.01:
            warnings.warn(f"SHAP additivity check failed. Mean difference: {diff:.4f}")
    
    def plot_summary(
        self,
        shap_results: SHAPResults,
        plot_type: str = "dot",
        max_display: int = 20
    ):
        """
        Create SHAP summary plot.
        
        Args:
            shap_results: SHAP results to plot
            plot_type: "dot", "bar", or "violin"
            max_display: Maximum features to display
        """
        if not SHAP_AVAILABLE:
            logger.warning("Cannot create plot: SHAP not installed")
            return
        
        shap.summary_plot(
            shap_results.values,
            features=shap_results.data,
            feature_names=shap_results.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=True
        )
    
    def plot_waterfall(
        self,
        shap_results: SHAPResults,
        instance_idx: int = 0
    ):
        """
        Create waterfall plot for single instance.
        
        Args:
            shap_results: SHAP results
            instance_idx: Index of instance to plot
        """
        if not SHAP_AVAILABLE:
            logger.warning("Cannot create plot: SHAP not installed")
            return
        
        # Get values for instance
        if len(shap_results.values.shape) == 3:  # Multi-class
            values = shap_results.values[instance_idx, :, 1]
            base = shap_results.expected_value[1]
        else:
            values = shap_results.values[instance_idx]
            base = shap_results.expected_value
        
        shap.waterfall_plot(shap.Explanation(
            values=values,
            base_values=base,
            data=shap_results.data[instance_idx] if shap_results.data is not None else None,
            feature_names=shap_results.feature_names
        ))
    
    def plot_force(
        self,
        shap_results: SHAPResults,
        instance_idx: int = 0
    ):
        """
        Create force plot for single instance.
        
        Args:
            shap_results: SHAP results
            instance_idx: Index of instance to plot
        """
        if not SHAP_AVAILABLE:
            logger.warning("Cannot create plot: SHAP not installed")
            return
        
        # Get values for instance
        if len(shap_results.values.shape) == 3:  # Multi-class
            values = shap_results.values[instance_idx, :, 1]
            base = shap_results.expected_value[1]
        else:
            values = shap_results.values[instance_idx]
            base = shap_results.expected_value
        
        shap.force_plot(
            base,
            values,
            shap_results.data[instance_idx] if shap_results.data is not None else None,
            feature_names=shap_results.feature_names
        )
    
    def plot_dependence(
        self,
        shap_results: SHAPResults,
        feature: Union[str, int],
        interaction_feature: Optional[Union[str, int]] = "auto"
    ):
        """
        Create dependence plot for a feature.
        
        Args:
            shap_results: SHAP results
            feature: Feature to plot
            interaction_feature: Feature to color by
        """
        if not SHAP_AVAILABLE:
            logger.warning("Cannot create plot: SHAP not installed")
            return
        
        shap.dependence_plot(
            feature,
            shap_results.values,
            shap_results.data,
            feature_names=shap_results.feature_names,
            interaction_index=interaction_feature,
            show=True
        )