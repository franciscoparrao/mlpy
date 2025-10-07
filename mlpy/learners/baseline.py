"""Baseline learners for MLPY.

These learners provide simple baseline predictions and are useful
for benchmarking and debugging.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from collections import Counter

from .base import Learner
from .classification import LearnerClassif
from .regression import LearnerRegr
from ..tasks import Task, TaskClassif, TaskRegr
from ..predictions import PredictionClassif, PredictionRegr
from ..utils.registry import mlpy_learners


class LearnerClassifFeatureless(LearnerClassif):
    """Featureless classification learner.
    
    This learner ignores features and makes predictions based solely
    on the target distribution in the training data.
    
    Parameters
    ----------
    method : str, default="mode"
        Prediction method:
        - "mode": Always predict the most frequent class
        - "sample": Sample from the training distribution
        - "weighted": Sample weighted by class frequencies
    predict_response : str, default="mode"
        What to predict when using predict_type="response"
    """
    
    def __init__(
        self,
        method: str = "mode",
        predict_response: str = "mode",
        id: Optional[str] = None,
        predict_type: str = "response",
        **kwargs
    ):
        super().__init__(
            id=id or "classif.featureless",
            predict_type=predict_type,
            **kwargs
        )
        
        self.method = method
        self.predict_response = predict_response
        
        # Will be set during training
        self.class_counts_: Optional[Dict[str, int]] = None
        self.classes_: Optional[List[str]] = None
        self.class_probs_: Optional[np.ndarray] = None
        
    def _train(self, task: TaskClassif, row_ids: Optional[List[int]] = None) -> "LearnerClassifFeatureless":
        """Train the learner."""
            
        # Get training data
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        y = task.truth(rows=row_ids)
        
        if isinstance(y, pd.Series):
            y = y.values
            
        # Count class frequencies
        counts = Counter(y)
        
        # Store training info
        self.classes_ = sorted(counts.keys())
        self.class_counts_ = counts
        
        # Calculate probabilities
        total = sum(counts.values())
        self.class_probs_ = np.array([counts[c] / total for c in self.classes_])
        
        # Mark as trained
        self._model = self
        self._train_task = task
        
        return self
        
    def _predict(self, task: TaskClassif, row_ids: Optional[List[int]] = None) -> PredictionClassif:
        """Make predictions."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
            
        # Get prediction data
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        n_samples = len(row_ids)
        truth = task.truth(rows=row_ids)
        
        # Make predictions based on method
        if self.predict_type == "response":
            if self.predict_response == "mode":
                # Always predict most frequent class
                mode_class = max(self.class_counts_, key=self.class_counts_.get)
                response = np.repeat(mode_class, n_samples)
            elif self.predict_response == "sample":
                # Sample uniformly from classes
                response = np.random.choice(self.classes_, size=n_samples)
            elif self.predict_response == "weighted":
                # Sample according to training distribution
                response = np.random.choice(
                    self.classes_, 
                    size=n_samples,
                    p=self.class_probs_
                )
            else:
                response = np.repeat(self.classes_[0], n_samples)
                
            return PredictionClassif(
                task=task,
                learner_id=self.id,
                row_ids=row_ids,
                truth=truth,
                response=response
            )
        else:  # prob
            # Return training distribution probabilities for all samples
            prob_array = np.tile(self.class_probs_, (n_samples, 1))
            prob_df = pd.DataFrame(prob_array, columns=self.classes_)
            
            return PredictionClassif(
                task=task,
                learner_id=self.id,
                row_ids=row_ids,
                truth=truth,
                response=None,
                prob=prob_df
            )
            
    @property
    def task_type(self) -> str:
        """Type of task this learner can handle."""
        return "classif"
        
    @property
    def _properties(self) -> set:
        """Properties for this learner type."""
        return {"multiclass", "twoclass", "featureless"}
        
    @property
    def is_trained(self) -> bool:
        """Check if learner is trained."""
        return hasattr(self, 'classes_') and self.classes_ is not None
        
    def reset(self) -> "LearnerClassifFeatureless":
        """Reset the learner to untrained state."""
        super().reset()
        self.class_counts_ = None
        self.classes_ = None
        self.class_probs_ = None
        return self
        
    def clone(self, deep: bool = True) -> "LearnerClassifFeatureless":
        """Create a copy of the learner."""
        return LearnerClassifFeatureless(
            method=self.method,
            predict_response=self.predict_response,
            id=self.id,
            predict_type=self.predict_type
        )


class LearnerRegrFeatureless(LearnerRegr):
    """Featureless regression learner.
    
    This learner ignores features and makes predictions based solely
    on the target distribution in the training data.
    
    Parameters
    ----------
    method : str, default="mean"
        Prediction method:
        - "mean": Always predict the mean
        - "median": Always predict the median
        - "sample": Sample from training targets
    """
    
    def __init__(
        self,
        method: str = "mean",
        id: Optional[str] = None,
        predict_type: str = "response",
        **kwargs
    ):
        super().__init__(
            id=id or "regr.featureless",
            predict_type=predict_type,
            **kwargs
        )
        
        self.method = method
        
        # Will be set during training
        self.center_: Optional[float] = None
        self.scale_: Optional[float] = None
        self.y_train_: Optional[np.ndarray] = None
        
    def _train(self, task: TaskRegr, row_ids: Optional[List[int]] = None) -> "LearnerRegrFeatureless":
        """Train the learner."""
            
        # Get training data
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        y = task.truth(rows=row_ids)
        
        if isinstance(y, pd.Series):
            y = y.values
            
        self.y_train_ = np.array(y).copy()
        
        # Calculate statistics
        if self.method == "mean":
            self.center_ = np.mean(y)
            self.scale_ = np.std(y)
        elif self.method == "median":
            self.center_ = np.median(y)
            self.scale_ = np.std(y)
        elif self.method == "sample":
            self.center_ = np.mean(y)
            self.scale_ = np.std(y)
            
        # Mark as trained
        self._model = self
        self._train_task = task
        
        return self
        
    def _predict(self, task: TaskRegr, row_ids: Optional[List[int]] = None) -> PredictionRegr:
        """Make predictions."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
            
        # Get prediction data
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        n_samples = len(row_ids)
        truth = task.truth(rows=row_ids)
        
        # Make predictions
        if self.method in ["mean", "median"]:
            response = np.repeat(self.center_, n_samples)
        elif self.method == "sample":
            # Sample from training targets
            response = np.random.choice(self.y_train_, size=n_samples)
        else:
            response = np.repeat(self.center_, n_samples)
            
        # Standard errors if requested
        se = None
        if self.predict_type == "se":
            se = np.repeat(self.scale_ / np.sqrt(len(self.y_train_)), n_samples)
            
        return PredictionRegr(
            task=task,
            learner_id=self.id,
            row_ids=row_ids,
            truth=truth,
            response=response,
            se=se
        )
        
    @property
    def task_type(self) -> str:
        """Type of task this learner can handle."""
        return "regr"
        
    @property
    def _properties(self) -> set:
        """Properties for this learner type."""
        return {"featureless"}
        
    @property
    def is_trained(self) -> bool:
        """Check if learner is trained."""
        return hasattr(self, 'center_') and self.center_ is not None
        
    def reset(self) -> "LearnerRegrFeatureless":
        """Reset the learner to untrained state."""
        super().reset()
        self.center_ = None
        self.scale_ = None
        self.y_train_ = None
        return self
        
    def clone(self, deep: bool = True) -> "LearnerRegrFeatureless":
        """Create a copy of the learner."""
        return LearnerRegrFeatureless(
            method=self.method,
            id=self.id,
            predict_type=self.predict_type
        )


class LearnerClassifDebug(LearnerClassif):
    """Debug classification learner.
    
    This learner is for testing and debugging. It can be configured
    to succeed, fail, or produce specific outputs.
    
    Parameters
    ----------
    error_train : float, default=0.0
        Probability of error during training
    error_predict : float, default=0.0  
        Probability of error during prediction
    predict_response : str, default="sample"
        What to return for predictions
    """
    
    def __init__(
        self,
        error_train: float = 0.0,
        error_predict: float = 0.0,
        predict_response: str = "sample",
        id: Optional[str] = None,
        predict_type: str = "response",
        **kwargs
    ):
        super().__init__(
            id=id or "classif.debug",
            predict_type=predict_type,
            **kwargs
        )
        
        self.error_train = error_train
        self.error_predict = error_predict
        self.predict_response = predict_response
        
        # Storage
        self.n_train_calls = 0
        self.n_predict_calls = 0
        self.classes_ = None
        
    def _train(self, task: TaskClassif, row_ids: Optional[List[int]] = None) -> "LearnerClassifDebug":
        """Train with configurable behavior."""
        self.n_train_calls += 1
            
        # Simulate error
        if np.random.random() < self.error_train:
            raise RuntimeError(f"Debug error during training (call {self.n_train_calls})")
            
        # Get training data
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        y = task.truth(rows=row_ids)
        
        if isinstance(y, pd.Series):
            y = y.values
            
        # Store basic info
        self.classes_ = sorted(np.unique(y).tolist())
        
        # Mark as trained
        self._model = self
        self._train_task = task
        
        return self
        
    def _predict(self, task: TaskClassif, row_ids: Optional[List[int]] = None) -> PredictionClassif:
        """Predict with configurable behavior."""
        self.n_predict_calls += 1
        
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
            
        # Simulate error
        if np.random.random() < self.error_predict:
            raise RuntimeError(f"Debug error during prediction (call {self.n_predict_calls})")
            
        # Get prediction data
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        n_samples = len(row_ids)
        truth = task.truth(rows=row_ids)
        
        # Make predictions
        if self.predict_type == "response":
            # Return random predictions
            response = np.random.choice(self.classes_, size=n_samples)
            
            return PredictionClassif(
                task=task,
                learner_id=self.id,
                row_ids=row_ids,
                truth=truth,
                response=response
            )
        else:  # prob
            # Return random probabilities
            n_classes = len(self.classes_)
            probs = np.random.random((n_samples, n_classes))
            # Normalize to sum to 1
            probs = probs / probs.sum(axis=1, keepdims=True)
            prob_df = pd.DataFrame(probs, columns=self.classes_)
            
            return PredictionClassif(
                task=task,
                learner_id=self.id,
                row_ids=row_ids,
                truth=truth,
                response=None,
                prob=prob_df
            )
            
    @property
    def task_type(self) -> str:
        """Type of task this learner can handle."""
        return "classif"
        
    @property
    def _properties(self) -> set:
        """Properties for this learner type."""
        return {"multiclass", "twoclass", "debug"}
        
    @property
    def is_trained(self) -> bool:
        """Check if learner is trained."""
        return self.classes_ is not None
        
    def reset(self) -> "LearnerClassifDebug":
        """Reset the learner to untrained state."""
        super().reset()
        self.classes_ = None
        self.n_train_calls = 0
        self.n_predict_calls = 0
        return self
        
    def clone(self, deep: bool = True) -> "LearnerClassifDebug":
        """Create a copy of the learner."""
        return LearnerClassifDebug(
            error_train=self.error_train,
            error_predict=self.error_predict,
            predict_response=self.predict_response,
            id=self.id,
            predict_type=self.predict_type
        )


class LearnerRegrDebug(LearnerRegr):
    """Debug regression learner.
    
    This learner is for testing and debugging.
    
    Parameters
    ----------
    error_train : float, default=0.0
        Probability of error during training
    error_predict : float, default=0.0
        Probability of error during prediction
    """
    
    def __init__(
        self,
        error_train: float = 0.0,
        error_predict: float = 0.0,
        id: Optional[str] = None,
        predict_type: str = "response",
        **kwargs
    ):
        super().__init__(
            id=id or "regr.debug",
            predict_type=predict_type,
            **kwargs
        )
        
        self.error_train = error_train
        self.error_predict = error_predict
        
        # Storage
        self.n_train_calls = 0
        self.n_predict_calls = 0
        self.y_mean_ = None
        self.y_std_ = None
        
    def _train(self, task: TaskRegr, row_ids: Optional[List[int]] = None) -> "LearnerRegrDebug":
        """Train with configurable behavior."""
        self.n_train_calls += 1
            
        # Simulate error
        if np.random.random() < self.error_train:
            raise RuntimeError(f"Debug error during training (call {self.n_train_calls})")
            
        # Get training data
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        y = task.truth(rows=row_ids)
        
        if isinstance(y, pd.Series):
            y = y.values
            
        # Store basic statistics
        self.y_mean_ = np.mean(y)
        self.y_std_ = np.std(y)
        
        # Mark as trained
        self._model = self
        self._train_task = task
        
        return self
        
    def _predict(self, task: TaskRegr, row_ids: Optional[List[int]] = None) -> PredictionRegr:
        """Predict with configurable behavior."""
        self.n_predict_calls += 1
        
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
            
        # Simulate error
        if np.random.random() < self.error_predict:
            raise RuntimeError(f"Debug error during prediction (call {self.n_predict_calls})")
            
        # Get prediction data
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        n_samples = len(row_ids)
        truth = task.truth(rows=row_ids)
        
        # Return random values around training mean
        response = np.random.normal(self.y_mean_, self.y_std_, size=n_samples)
        
        # Standard errors if requested
        se = None
        if self.predict_type == "se":
            se = np.repeat(self.y_std_, n_samples)
            
        return PredictionRegr(
            task=task,
            learner_id=self.id,
            row_ids=row_ids,
            truth=truth,
            response=response,
            se=se
        )
        
    @property
    def task_type(self) -> str:
        """Type of task this learner can handle."""
        return "regr"
        
    @property
    def _properties(self) -> set:
        """Properties for this learner type."""
        return {"debug"}
        
    @property
    def is_trained(self) -> bool:
        """Check if learner is trained."""
        return self.y_mean_ is not None
        
    def reset(self) -> "LearnerRegrDebug":
        """Reset the learner to untrained state."""
        super().reset()
        self.y_mean_ = None
        self.y_std_ = None
        self.n_train_calls = 0
        self.n_predict_calls = 0
        return self
        
    def clone(self, deep: bool = True) -> "LearnerRegrDebug":
        """Create a copy of the learner."""
        return LearnerRegrDebug(
            error_train=self.error_train,
            error_predict=self.error_predict,
            id=self.id,
            predict_type=self.predict_type
        )


# Register learners
mlpy_learners.register("classif.featureless", LearnerClassifFeatureless)
mlpy_learners.register("regr.featureless", LearnerRegrFeatureless)
mlpy_learners.register("classif.debug", LearnerClassifDebug)
mlpy_learners.register("regr.debug", LearnerRegrDebug)


# Convenience aliases
LearnerBaseline = LearnerClassifFeatureless


__all__ = [
    "LearnerClassifFeatureless",
    "LearnerRegrFeatureless", 
    "LearnerClassifDebug",
    "LearnerRegrDebug",
    "LearnerBaseline",
]