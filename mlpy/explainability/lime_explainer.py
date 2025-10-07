"""
LIME (Local Interpretable Model-agnostic Explanations)
======================================================

Local surrogate models for explaining individual predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
import warnings
import logging

logger = logging.getLogger(__name__)

try:
    import lime
    import lime.lime_tabular
    import lime.lime_text
    import lime.lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not installed. Install with: pip install lime")


@dataclass
class LIMEExplanation:
    """Container for LIME explanation results."""
    
    instance: np.ndarray
    prediction: Union[float, np.ndarray]
    local_exp: List[Tuple[int, float]]  # Feature index and weight
    score: float  # R^2 of local model
    local_pred: float
    feature_names: List[str]
    class_names: Optional[List[str]] = None
    
    def as_dict(self) -> Dict[str, float]:
        """Convert explanation to dictionary."""
        return {
            self.feature_names[idx]: weight
            for idx, weight in self.local_exp
        }
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top n most important features."""
        sorted_exp = sorted(self.local_exp, key=lambda x: abs(x[1]), reverse=True)
        return [
            (self.feature_names[idx], weight)
            for idx, weight in sorted_exp[:n]
        ]


class LIMEExplainer:
    """LIME-based local model explainer."""
    
    def __init__(
        self,
        model: Any,
        training_data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        mode: str = "tabular",
        kernel_width: Optional[float] = None,
        discretize_continuous: bool = True,
        random_state: int = 42
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: Model to explain
            training_data: Training data for statistics
            feature_names: Names of features
            class_names: Names of classes (for classification)
            mode: "tabular", "text", or "image"
            kernel_width: Width of exponential kernel
            discretize_continuous: Whether to discretize continuous features
            random_state: Random seed
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not installed. Install with: pip install lime")
        
        self.model = model
        self.mode = mode
        self.feature_names = feature_names or self._infer_feature_names(training_data)
        self.class_names = class_names
        self.random_state = random_state
        
        # Setup prediction function
        self.predict_fn = self._setup_predict_function()
        
        # Initialize appropriate LIME explainer
        if mode == "tabular":
            if isinstance(training_data, pd.DataFrame):
                training_data = training_data.values
                
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=self.feature_names,
                class_names=class_names,
                mode=self._get_mode(),
                discretize_continuous=discretize_continuous,
                kernel_width=kernel_width,
                random_state=random_state
            )
        elif mode == "text":
            self.explainer = lime.lime_text.LimeTextExplainer(
                class_names=class_names,
                random_state=random_state
            )
        elif mode == "image":
            self.explainer = lime.lime_image.LimeImageExplainer(
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _infer_feature_names(self, data: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        """Infer feature names from data."""
        if isinstance(data, pd.DataFrame):
            return data.columns.tolist()
        elif isinstance(data, np.ndarray):
            return [f"feature_{i}" for i in range(data.shape[1])]
        else:
            return []
    
    def _setup_predict_function(self) -> Callable:
        """Setup prediction function for LIME."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba
        else:
            # For regression or models without predict_proba
            def predict_fn(X):
                preds = self.model.predict(X)
                if len(preds.shape) == 1:
                    # Regression: return as 2D array
                    return preds.reshape(-1, 1)
                return preds
            return predict_fn
    
    def _get_mode(self) -> str:
        """Get LIME mode based on model type."""
        if hasattr(self.model, 'predict_proba'):
            return "classification"
        else:
            return "regression"
    
    def explain_instance(
        self,
        instance: Union[pd.Series, np.ndarray, str],
        num_features: int = 10,
        num_samples: int = 5000,
        distance_metric: str = "euclidean",
        model_regressor: Optional[Any] = None
    ) -> LIMEExplanation:
        """
        Explain a single instance.
        
        Args:
            instance: Instance to explain
            num_features: Number of features in explanation
            num_samples: Number of samples for local model
            distance_metric: Distance metric for weights
            model_regressor: Local model to use
            
        Returns:
            LIMEExplanation object
        """
        if self.mode == "tabular":
            return self._explain_tabular(
                instance, num_features, num_samples, 
                distance_metric, model_regressor
            )
        elif self.mode == "text":
            return self._explain_text(
                instance, num_features, num_samples
            )
        elif self.mode == "image":
            return self._explain_image(
                instance, num_features, num_samples
            )
    
    def _explain_tabular(
        self,
        instance: Union[pd.Series, np.ndarray],
        num_features: int,
        num_samples: int,
        distance_metric: str,
        model_regressor: Optional[Any]
    ) -> LIMEExplanation:
        """Explain tabular data instance."""
        if isinstance(instance, pd.Series):
            instance = instance.values
        
        # Get explanation
        exp = self.explainer.explain_instance(
            instance,
            self.predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            distance_metric=distance_metric,
            model_regressor=model_regressor
        )
        
        # Get prediction
        prediction = self.predict_fn(instance.reshape(1, -1))[0]
        
        # Extract explanation
        local_exp = exp.as_list()
        local_exp_indexed = []
        
        for feature_desc, weight in local_exp:
            # Parse feature description to get index
            for idx, name in enumerate(self.feature_names):
                if name in feature_desc:
                    local_exp_indexed.append((idx, weight))
                    break
        
        return LIMEExplanation(
            instance=instance,
            prediction=prediction,
            local_exp=local_exp_indexed,
            score=exp.score,
            local_pred=exp.local_pred[0] if hasattr(exp, 'local_pred') else 0.0,
            feature_names=self.feature_names,
            class_names=self.class_names
        )
    
    def _explain_text(
        self,
        text: str,
        num_features: int,
        num_samples: int
    ) -> LIMEExplanation:
        """Explain text instance."""
        exp = self.explainer.explain_instance(
            text,
            self.predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Get prediction
        prediction = self.predict_fn([text])[0]
        
        # Extract explanation (word indices and weights)
        local_exp = exp.as_list()
        
        return LIMEExplanation(
            instance=np.array([text]),
            prediction=prediction,
            local_exp=local_exp,
            score=exp.score if hasattr(exp, 'score') else 0.0,
            local_pred=0.0,
            feature_names=["text"],
            class_names=self.class_names
        )
    
    def _explain_image(
        self,
        image: np.ndarray,
        num_features: int,
        num_samples: int
    ) -> LIMEExplanation:
        """Explain image instance."""
        exp = self.explainer.explain_instance(
            image,
            self.predict_fn,
            top_labels=1,
            hide_color=0,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Get prediction
        prediction = self.predict_fn(image.reshape(1, *image.shape))[0]
        
        # Extract explanation (superpixel indices and weights)
        local_exp = exp.local_exp[0] if exp.local_exp else []
        
        return LIMEExplanation(
            instance=image,
            prediction=prediction,
            local_exp=local_exp,
            score=0.0,  # Not available for images
            local_pred=0.0,
            feature_names=["image"],
            class_names=self.class_names
        )
    
    def explain_multiple(
        self,
        instances: Union[pd.DataFrame, np.ndarray, List],
        num_features: int = 10,
        num_samples: int = 5000
    ) -> List[LIMEExplanation]:
        """
        Explain multiple instances.
        
        Args:
            instances: Instances to explain
            num_features: Number of features per explanation
            num_samples: Samples for each local model
            
        Returns:
            List of LIMEExplanation objects
        """
        explanations = []
        
        if isinstance(instances, pd.DataFrame):
            instances_array = instances.values
        else:
            instances_array = instances
        
        for i in range(len(instances_array)):
            exp = self.explain_instance(
                instances_array[i],
                num_features=num_features,
                num_samples=num_samples
            )
            explanations.append(exp)
        
        return explanations
    
    def get_global_importance(
        self,
        sample_data: Union[pd.DataFrame, np.ndarray],
        num_features: int = 10,
        num_samples_per: int = 1000
    ) -> pd.DataFrame:
        """
        Estimate global feature importance using multiple local explanations.
        
        Args:
            sample_data: Sample of data to explain
            num_features: Features per explanation
            num_samples_per: Samples per local model
            
        Returns:
            DataFrame with global importance scores
        """
        if isinstance(sample_data, pd.DataFrame):
            sample_array = sample_data.values
        else:
            sample_array = sample_data
        
        # Collect feature weights across instances
        feature_weights = {i: [] for i in range(len(self.feature_names))}
        
        for i in range(len(sample_array)):
            exp = self.explain_instance(
                sample_array[i],
                num_features=num_features,
                num_samples=num_samples_per
            )
            
            for feat_idx, weight in exp.local_exp:
                feature_weights[feat_idx].append(abs(weight))
        
        # Calculate mean importance
        importance_scores = []
        for feat_idx, weights in feature_weights.items():
            if weights:
                mean_importance = np.mean(weights)
            else:
                mean_importance = 0.0
            
            importance_scores.append({
                'feature': self.feature_names[feat_idx],
                'importance': mean_importance,
                'std': np.std(weights) if weights else 0.0,
                'count': len(weights)
            })
        
        df = pd.DataFrame(importance_scores)
        return df.sort_values('importance', ascending=False)
    
    def plot_explanation(
        self,
        explanation: LIMEExplanation,
        num_features: int = 10
    ):
        """
        Plot LIME explanation.
        
        Args:
            explanation: LIME explanation to plot
            num_features: Number of features to show
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return
        
        # Get top features
        top_features = explanation.get_top_features(num_features)
        
        # Prepare data for plotting
        features = [f for f, _ in top_features]
        weights = [w for _, w in top_features]
        colors = ['green' if w > 0 else 'red' for w in weights]
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, weights, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Weight')
        ax.set_title('LIME Feature Importance')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for i, (feature, weight) in enumerate(top_features):
            ax.text(weight, i, f'{weight:.3f}', 
                   ha='left' if weight > 0 else 'right',
                   va='center')
        
        plt.tight_layout()
        plt.show()
    
    def check_consistency(
        self,
        instance: Union[pd.Series, np.ndarray],
        num_runs: int = 10,
        num_features: int = 10
    ) -> Dict[str, float]:
        """
        Check consistency of explanations across multiple runs.
        
        Args:
            instance: Instance to explain
            num_runs: Number of runs
            num_features: Features per explanation
            
        Returns:
            Dictionary with consistency metrics
        """
        all_weights = {i: [] for i in range(len(self.feature_names))}
        
        for _ in range(num_runs):
            exp = self.explain_instance(instance, num_features=num_features)
            
            for feat_idx, weight in exp.local_exp:
                all_weights[feat_idx].append(weight)
        
        # Calculate consistency metrics
        consistency_scores = {}
        
        for feat_idx, weights in all_weights.items():
            if len(weights) > 1:
                # Coefficient of variation
                mean_weight = np.mean(weights)
                std_weight = np.std(weights)
                cv = std_weight / (abs(mean_weight) + 1e-10)
                consistency_scores[self.feature_names[feat_idx]] = 1.0 / (1.0 + cv)
            else:
                consistency_scores[self.feature_names[feat_idx]] = 0.0
        
        return consistency_scores