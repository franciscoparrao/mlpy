"""LIME (Local Interpretable Model-agnostic Explanations) integration for MLPY."""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass

from .base import Interpreter, InterpretationResult, FeatureImportance
from ..learners import Learner
from ..tasks import Task, TaskClassif, TaskRegr

# Try to import LIME
try:
    import lime
    import lime.lime_tabular
    import lime.lime_text
    _HAS_LIME = True
except ImportError:
    _HAS_LIME = False
    lime = None


@dataclass
class LIMEExplanation:
    """Container for LIME explanation.
    
    Parameters
    ----------
    instance_idx : int
        Index of explained instance.
    explanation : Any
        LIME explanation object.
    feature_importance : dict
        Feature importance scores for this instance.
    prediction : array-like
        Model prediction for this instance.
    true_label : Any, optional
        True label if available.
    """
    instance_idx: int
    explanation: Any
    feature_importance: Dict[str, float]
    prediction: np.ndarray
    true_label: Optional[Any] = None
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top n most important features.
        
        Parameters
        ----------
        n : int
            Number of features to return.
            
        Returns
        -------
        list of tuple
            List of (feature, importance) tuples.
        """
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:n]
    
    def as_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with features and importances.
        """
        return pd.DataFrame(
            list(self.feature_importance.items()),
            columns=['feature', 'importance']
        ).sort_values('importance', key=abs, ascending=False)


class LIMEInterpreter(Interpreter):
    """LIME-based model interpreter.
    
    Parameters
    ----------
    mode : str, optional
        Type of data:
        - "tabular": For tabular data (default)
        - "text": For text data
    num_features : int, optional
        Number of features to include in explanation.
    num_samples : int, optional
        Number of samples to generate for local approximation.
    kernel_width : float, optional
        Kernel width for exponential kernel.
    feature_selection : str, optional
        Feature selection method ("forward_selection", "lasso_path", "auto").
    discretize_continuous : bool, optional
        Whether to discretize continuous features.
    discretizer : str, optional
        Discretization method ("quartile", "decile", "entropy").
    random_state : int, optional
        Random seed.
    """
    
    def __init__(
        self,
        mode: str = "tabular",
        num_features: int = 10,
        num_samples: int = 5000,
        kernel_width: Optional[float] = None,
        feature_selection: str = "auto",
        discretize_continuous: bool = True,
        discretizer: str = "quartile",
        random_state: Optional[int] = None,
        **kwargs
    ):
        """Initialize LIME interpreter."""
        super().__init__(id="lime_interpreter")
        
        if not _HAS_LIME:
            raise ImportError("LIME is not installed. Install with: pip install lime")
            
        self.mode = mode
        self.num_features = num_features
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.feature_selection = feature_selection
        self.discretize_continuous = discretize_continuous
        self.discretizer = discretizer
        self.random_state = random_state
        self.kwargs = kwargs
        
    def _create_explainer(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: List[str],
        task: Task
    ) -> Any:
        """Create LIME explainer.
        
        Parameters
        ----------
        X : array-like
            Training data.
        feature_names : list of str
            Feature names.
        task : Task
            Task object.
            
        Returns
        -------
        lime.lime_tabular.LimeTabularExplainer or similar
            LIME explainer.
        """
        if self.mode == "tabular":
            # Determine mode based on task type
            if isinstance(task, TaskClassif):
                mode = "classification"
                class_names = task.class_names if hasattr(task, 'class_names') else None
            else:
                mode = "regression"
                class_names = None
                
            # Convert to numpy if needed
            if isinstance(X, pd.DataFrame):
                training_data = X.values
            else:
                training_data = X
                
            # Create explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                feature_names=feature_names,
                class_names=class_names,
                mode=mode,
                discretize_continuous=self.discretize_continuous,
                discretizer=self.discretizer,
                kernel_width=self.kernel_width,
                feature_selection=self.feature_selection,
                random_state=self.random_state,
                **self.kwargs
            )
            
        elif self.mode == "text":
            # Text explainer
            explainer = lime.lime_text.LimeTextExplainer(
                class_names=task.class_names if hasattr(task, 'class_names') else None,
                feature_selection=self.feature_selection,
                random_state=self.random_state,
                **self.kwargs
            )
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        return explainer
    
    def _get_predict_fn(self, model: Any, task: Task):
        """Get appropriate prediction function.
        
        Parameters
        ----------
        model : Any
            Trained model.
        task : Task
            Task object.
            
        Returns
        -------
        callable
            Prediction function.
        """
        if isinstance(task, TaskClassif):
            # For classification, use predict_proba if available
            if hasattr(model, 'predict_proba'):
                return model.predict_proba
            else:
                # Fallback to predict with dummy probabilities
                def predict_fn(X):
                    preds = model.predict(X)
                    # Convert to dummy probabilities
                    n_classes = len(np.unique(preds))
                    probs = np.zeros((len(X), n_classes))
                    for i, pred in enumerate(preds):
                        probs[i, int(pred)] = 1.0
                    return probs
                return predict_fn
        else:
            # For regression, use predict
            return model.predict
    
    def interpret(
        self,
        learner: Learner,
        task: Task,
        indices: Optional[List[int]] = None,
        compute_global: bool = True,
        **kwargs
    ) -> InterpretationResult:
        """Interpret model using LIME.
        
        Parameters
        ----------
        learner : Learner
            Trained learner to interpret.
        task : Task
            Task with data.
        indices : list of int, optional
            Indices to explain. If None, explains first instance.
        compute_global : bool
            Whether to compute global importance by aggregating local.
        **kwargs
            Additional arguments for LIME.
            
        Returns
        -------
        InterpretationResult
            LIME interpretation results.
        """
        # Ensure learner is trained
        if not learner.is_trained:
            raise ValueError("Learner must be trained before interpretation")
            
        # Get data and model
        X = task.X
        y = task.y
        feature_names = self.get_feature_names(task)
        model = learner.model
        
        if model is None:
            raise ValueError("Learner has no trained model")
            
        # Create explainer
        explainer = self._create_explainer(X, feature_names, task)
        
        # Get prediction function
        predict_fn = self._get_predict_fn(model, task)
        
        # Default to explaining first instance if no indices provided
        if indices is None:
            indices = [0]
        else:
            indices = self.validate_indices(indices, len(X))
            
        # Generate local explanations
        local_explanations = {}
        all_feature_importances = []
        
        for idx in indices:
            # Get instance
            if isinstance(X, pd.DataFrame):
                instance = X.iloc[idx].values
            else:
                instance = X[idx]
                
            # Generate explanation
            if self.mode == "tabular":
                exp = explainer.explain_instance(
                    instance,
                    predict_fn,
                    num_features=self.num_features,
                    num_samples=self.num_samples
                )
            else:
                # For text mode, instance should be text
                exp = explainer.explain_instance(
                    instance,
                    predict_fn,
                    num_features=self.num_features,
                    num_samples=self.num_samples
                )
                
            # Extract feature importance
            feature_importance = {}
            for feat_idx, importance in exp.as_list():
                if self.mode == "tabular":
                    # For tabular, feat_idx might be feature condition like "feature > value"
                    # Extract just the feature name
                    feat_name = str(feat_idx).split()[0]
                    if feat_name in feature_importance:
                        # Aggregate if multiple conditions for same feature
                        feature_importance[feat_name] += importance
                    else:
                        feature_importance[feat_name] = importance
                else:
                    feature_importance[feat_idx] = importance
                    
            # Get prediction for this instance
            prediction = predict_fn(instance.reshape(1, -1))
            true_label = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
            
            # Create LIME explanation object
            lime_exp = LIMEExplanation(
                instance_idx=idx,
                explanation=exp,
                feature_importance=feature_importance,
                prediction=prediction,
                true_label=true_label
            )
            
            local_explanations[idx] = lime_exp
            all_feature_importances.append(feature_importance)
            
        # Compute global importance if requested
        global_importance = None
        if compute_global and all_feature_importances:
            # Aggregate local importances
            global_scores = {}
            
            for feat_imp in all_feature_importances:
                for feat, imp in feat_imp.items():
                    if feat not in global_scores:
                        global_scores[feat] = []
                    global_scores[feat].append(abs(imp))
                    
            # Average importance across instances
            avg_importances = {
                feat: np.mean(scores) for feat, scores in global_scores.items()
            }
            
            # Create FeatureImportance object
            features = list(avg_importances.keys())
            importances = [avg_importances[f] for f in features]
            
            global_importance = FeatureImportance(
                features=features,
                importances=np.array(importances),
                method="lime_mean_local"
            )
            
        # Create result
        result = InterpretationResult(
            learner=learner,
            task=task,
            method="lime",
            global_importance=global_importance,
            local_explanations=local_explanations,
            metadata={
                "mode": self.mode,
                "num_features": self.num_features,
                "num_samples": self.num_samples,
                "explainer": explainer
            }
        )
        
        return result
    
    def check_learner_compatibility(self, learner: Learner) -> bool:
        """Check if learner is compatible with LIME.
        
        Parameters
        ----------
        learner : Learner
            Learner to check.
            
        Returns
        -------
        bool
            True if compatible.
        """
        # LIME works with any model that has predict method
        if not hasattr(learner, 'model') or learner.model is None:
            return False
            
        model = learner.model
        return hasattr(model, 'predict')