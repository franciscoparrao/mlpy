"""SHAP (SHapley Additive exPlanations) integration for MLPY."""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass

from .base import Interpreter, InterpretationResult, FeatureImportance
from ..learners import Learner
from ..tasks import Task, TaskClassif, TaskRegr

# Try to import SHAP
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    shap = None


@dataclass 
class SHAPExplanation:
    """Container for SHAP explanation values.
    
    Parameters
    ----------
    values : array-like
        SHAP values for each feature and instance.
    base_values : array-like
        Base values (expected values).
    data : array-like, optional
        Original feature values.
    feature_names : list of str, optional
        Feature names.
    """
    values: np.ndarray
    base_values: Union[float, np.ndarray]
    data: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate and process inputs."""
        self.values = np.asarray(self.values)
        if isinstance(self.base_values, (list, tuple)):
            self.base_values = np.asarray(self.base_values)
        if self.data is not None:
            self.data = np.asarray(self.data)
            
    def get_feature_importance(self, method: str = "mean_abs") -> FeatureImportance:
        """Calculate feature importance from SHAP values.
        
        Parameters
        ----------
        method : str
            Method to aggregate SHAP values:
            - "mean_abs": Mean absolute SHAP value
            - "mean": Mean SHAP value  
            - "max_abs": Maximum absolute SHAP value
            
        Returns
        -------
        FeatureImportance
            Feature importance scores.
        """
        if method == "mean_abs":
            importances = np.abs(self.values).mean(axis=0)
        elif method == "mean":
            importances = self.values.mean(axis=0)
        elif method == "max_abs":
            importances = np.abs(self.values).max(axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
            
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importances))]
        return FeatureImportance(
            features=feature_names,
            importances=importances,
            method=f"shap_{method}"
        )


class SHAPInterpreter(Interpreter):
    """SHAP-based model interpreter.
    
    Parameters
    ----------
    explainer_type : str, optional
        Type of SHAP explainer to use:
        - "auto": Automatically select based on model
        - "tree": TreeExplainer for tree-based models
        - "linear": LinearExplainer for linear models
        - "kernel": KernelExplainer (model-agnostic, slower)
        - "deep": DeepExplainer for neural networks
    background_data : array-like, optional
        Background dataset for KernelExplainer.
    nsamples : int, optional
        Number of samples for KernelExplainer.
    link : str, optional
        Link function for TreeExplainer ("identity" or "logit").
    """
    
    def __init__(
        self,
        explainer_type: str = "auto",
        background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        nsamples: Optional[int] = None,
        link: Optional[str] = None,
        **kwargs
    ):
        """Initialize SHAP interpreter."""
        super().__init__(id="shap_interpreter")
        
        if not _HAS_SHAP:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
            
        self.explainer_type = explainer_type
        self.background_data = background_data
        self.nsamples = nsamples
        self.link = link
        self.kwargs = kwargs
        
    def _create_explainer(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        task: Task
    ) -> Any:
        """Create appropriate SHAP explainer.
        
        Parameters
        ----------
        model : Any
            Trained model.
        X : array-like
            Training data.
        task : Task
            Task object.
            
        Returns
        -------
        shap.Explainer
            SHAP explainer object.
        """
        if self.explainer_type == "auto":
            # Try to auto-detect best explainer
            try:
                # This will automatically choose TreeExplainer for tree models,
                # LinearExplainer for linear models, etc.
                explainer = shap.Explainer(model, X, **self.kwargs)
            except Exception:
                # Fallback to KernelExplainer
                warnings.warn("Auto-detection failed, using KernelExplainer")
                explainer = self._create_kernel_explainer(model, X)
                
        elif self.explainer_type == "tree":
            explainer = shap.TreeExplainer(model, link=self.link, **self.kwargs)
            
        elif self.explainer_type == "linear":
            explainer = shap.LinearExplainer(model, X, **self.kwargs)
            
        elif self.explainer_type == "kernel":
            explainer = self._create_kernel_explainer(model, X)
            
        elif self.explainer_type == "deep":
            explainer = shap.DeepExplainer(model, X, **self.kwargs)
            
        else:
            raise ValueError(f"Unknown explainer type: {self.explainer_type}")
            
        return explainer
    
    def _create_kernel_explainer(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> Any:
        """Create KernelExplainer with appropriate background data.
        
        Parameters
        ----------
        model : Any
            Model to explain.
        X : array-like
            Training data.
            
        Returns
        -------
        shap.KernelExplainer
            Kernel explainer.
        """
        # Use provided background data or sample from X
        if self.background_data is not None:
            background = self.background_data
        else:
            # Sample subset of data as background
            n_background = min(100, len(X))
            if isinstance(X, pd.DataFrame):
                background = shap.sample(X, n_background)
            else:
                indices = np.random.choice(len(X), n_background, replace=False)
                background = X[indices]
                
        # Create prediction function
        if hasattr(model, 'predict_proba'):
            predict_fn = model.predict_proba
        else:
            predict_fn = model.predict
            
        # Create explainer
        explainer = shap.KernelExplainer(
            predict_fn,
            background,
            link="identity" if self.link is None else self.link,
            **self.kwargs
        )
        
        return explainer
    
    def interpret(
        self,
        learner: Learner,
        task: Task,
        indices: Optional[List[int]] = None,
        compute_global: bool = True,
        **kwargs
    ) -> InterpretationResult:
        """Interpret model using SHAP.
        
        Parameters
        ----------
        learner : Learner
            Trained learner to interpret.
        task : Task
            Task with data.
        indices : list of int, optional
            Indices to explain locally.
        compute_global : bool
            Whether to compute global importance.
        **kwargs
            Additional arguments for SHAP.
            
        Returns
        -------
        InterpretationResult
            SHAP interpretation results.
        """
        # Ensure learner is trained
        if not learner.is_trained:
            raise ValueError("Learner must be trained before interpretation")
            
        # Get data
        X = task.X
        feature_names = self.get_feature_names(task)
        
        # Get underlying model
        model = learner.model
        if model is None:
            raise ValueError("Learner has no trained model")
            
        # Create explainer
        explainer = self._create_explainer(model, X, task)
        
        # Compute SHAP values for all data
        if self.nsamples is not None and hasattr(explainer, "shap_values"):
            # For KernelExplainer with sampling
            shap_values = explainer.shap_values(X, nsamples=self.nsamples)
        else:
            shap_values = explainer(X)
            
        # Handle different SHAP value formats
        if hasattr(shap_values, "values"):
            # New SHAP Explanation object
            values = shap_values.values
            base_values = shap_values.base_values
            data_values = shap_values.data
        else:
            # Legacy format
            values = shap_values
            base_values = explainer.expected_value
            data_values = X
            
        # Handle multi-output (e.g., multiclass)
        if isinstance(values, list):
            # For multiclass, use values for positive class or first non-base class
            if isinstance(task, TaskClassif) and len(values) == 2:
                # Binary classification - use positive class
                values = values[1]
            else:
                # Use first output
                values = values[0]
                warnings.warn("Multiple outputs detected, using first output")
                
        if isinstance(base_values, (list, np.ndarray)) and len(base_values) > 1:
            base_values = base_values[0]
            
        # Create SHAP explanation object
        shap_explanation = SHAPExplanation(
            values=values,
            base_values=base_values,
            data=data_values,
            feature_names=feature_names
        )
        
        # Compute global importance if requested
        global_importance = None
        if compute_global:
            global_importance = shap_explanation.get_feature_importance()
            
        # Compute local explanations for specific indices
        local_explanations = None
        if indices is not None:
            validated_indices = self.validate_indices(indices, len(X))
            local_explanations = {}
            
            for idx in validated_indices:
                local_exp = SHAPExplanation(
                    values=values[idx:idx+1],
                    base_values=base_values,
                    data=data_values[idx:idx+1] if data_values is not None else None,
                    feature_names=feature_names
                )
                local_explanations[idx] = local_exp
                
        # Create result
        result = InterpretationResult(
            learner=learner,
            task=task,
            method="shap",
            global_importance=global_importance,
            local_explanations=local_explanations,
            metadata={
                "explainer_type": self.explainer_type,
                "shap_explanation": shap_explanation,
                "explainer": explainer
            }
        )
        
        return result
    
    def check_learner_compatibility(self, learner: Learner) -> bool:
        """Check if learner is compatible with SHAP.
        
        Parameters
        ----------
        learner : Learner
            Learner to check.
            
        Returns
        -------
        bool
            True if compatible.
        """
        # SHAP works with most models that have predict method
        if not hasattr(learner, 'model') or learner.model is None:
            return False
            
        model = learner.model
        
        # Check for required methods
        has_predict = hasattr(model, 'predict')
        has_predict_proba = hasattr(model, 'predict_proba')
        
        return has_predict or has_predict_proba