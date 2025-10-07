"""
Prediction classes for MLPY.

Predictions encapsulate the results of applying a learner to a task,
including predicted values, true values, and methods for evaluation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from mlpy.core.base import MLPYObject


class Prediction(MLPYObject, ABC):
    """
    Abstract base class for predictions.
    
    Parameters
    ----------
    row_ids : List[int]
        Row IDs for the predictions
    truth : array-like, optional
        True values
    task : Task, optional
        The task this prediction was made on
    """
    
    def __init__(
        self,
        row_ids: List[int],
        truth: Optional[np.ndarray] = None,
        task: Optional["Task"] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.row_ids = np.array(row_ids)
        self.truth = np.array(truth) if truth is not None else None
        self.task = task
        
        # Validate lengths
        if self.truth is not None and len(self.row_ids) != len(self.truth):
            raise ValueError(
                f"Length mismatch: {len(self.row_ids)} row_ids vs "
                f"{len(self.truth)} truth values"
            )
    
    @property
    def n(self) -> int:
        """Number of predictions."""
        return len(self.row_ids)
    
    @property
    def has_truth(self) -> bool:
        """Whether truth values are available."""
        return self.truth is not None
    
    @property
    @abstractmethod
    def predict_types(self) -> Dict[str, bool]:
        """Available prediction types."""
        pass
    
    def as_data_frame(self) -> pd.DataFrame:
        """
        Convert predictions to DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with predictions and truth
        """
        data = {"row_id": self.row_ids}
        
        if self.has_truth:
            data["truth"] = self.truth
        
        # Add prediction-specific columns
        data.update(self._get_prediction_data())
        
        return pd.DataFrame(data)
    
    @abstractmethod
    def _get_prediction_data(self) -> Dict[str, np.ndarray]:
        """
        Get prediction data for DataFrame conversion.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Column name to data mapping
        """
        pass
    
    def score(self, measures: Union["Measure", List["Measure"]]) -> Dict[str, float]:
        """
        Calculate performance measures.
        
        Parameters
        ----------
        measures : Measure or List[Measure]
            Measures to calculate
            
        Returns
        -------
        Dict[str, float]
            Measure IDs to scores
        """
        if not isinstance(measures, list):
            measures = [measures]
        
        scores = {}
        for measure in measures:
            scores[measure.id] = measure.score(self)
        
        return scores
    
    def confusion_matrix(self) -> Optional[np.ndarray]:
        """
        Get confusion matrix (classification only).
        
        Returns
        -------
        np.ndarray or None
            Confusion matrix
        """
        return None
    
    def _get_params_for_hash(self) -> Dict[str, Any]:
        """Include prediction data in hash."""
        params = super()._get_params_for_hash()
        params["n_predictions"] = self.n
        params["has_truth"] = self.has_truth
        return params
    
    @property
    def _properties(self) -> set[str]:
        """Prediction properties."""
        props = set()
        
        if self.has_truth:
            props.add("has_truth")
            
        return props


class PredictionClassif(Prediction):
    """
    Classification predictions.
    
    Parameters
    ----------
    row_ids : List[int]
        Row IDs
    truth : array-like, optional
        True class labels
    response : array-like, optional
        Predicted class labels
    prob : array-like, optional
        Class probabilities (n_samples x n_classes)
    task : TaskClassif, optional
        Classification task
    """
    
    def __init__(
        self,
        row_ids: List[int],
        truth: Optional[np.ndarray] = None,
        response: Optional[np.ndarray] = None,
        prob: Optional[np.ndarray] = None,
        task: Optional["TaskClassif"] = None,
        **kwargs
    ):
        super().__init__(row_ids=row_ids, truth=truth, task=task, **kwargs)
        
        self.response = np.array(response) if response is not None else None
        self.prob = np.array(prob) if prob is not None else None
        
        # Validate
        if self.response is None and self.prob is None:
            raise ValueError("At least one of response or prob must be provided")
        
        if self.response is not None and len(self.response) != self.n:
            raise ValueError(f"Length mismatch: {self.n} predictions vs {len(self.response)} responses")
        
        if self.prob is not None:
            if self.prob.ndim == 1:
                # Binary classification with single probability
                if task and task.n_classes != 2:
                    raise ValueError("1D probabilities only valid for binary classification")
                # Expand to 2D
                self.prob = np.column_stack([1 - self.prob, self.prob])
            elif self.prob.shape[0] != self.n:
                raise ValueError(f"Shape mismatch: {self.n} predictions vs {self.prob.shape[0]} probability rows")
    
    @property
    def predict_types(self) -> Dict[str, bool]:
        """Available prediction types."""
        return {
            "response": self.response is not None,
            "prob": self.prob is not None,
        }
    
    @property
    def class_names(self) -> Optional[List[str]]:
        """Class names from task."""
        if self.task:
            return self.task.class_names
        return None
    
    @property
    def n_classes(self) -> Optional[int]:
        """Number of classes."""
        if self.task:
            return self.task.n_classes
        elif self.prob is not None:
            return self.prob.shape[1]
        return None
    
    def get_response(self) -> np.ndarray:
        """
        Get response predictions.
        
        Returns
        -------
        np.ndarray
            Predicted class labels
        """
        if self.response is not None:
            return self.response
        
        if self.prob is not None and self.class_names:
            # Get response from probabilities
            class_indices = np.argmax(self.prob, axis=1)
            return np.array([self.class_names[i] for i in class_indices])
        
        raise RuntimeError("Cannot compute response without class names")
    
    def get_prob(self, class_name: Optional[str] = None) -> np.ndarray:
        """
        Get probability predictions.
        
        Parameters
        ----------
        class_name : str, optional
            If provided, return probabilities for this class only
            
        Returns
        -------
        np.ndarray
            Class probabilities
        """
        if self.prob is None:
            raise RuntimeError("No probabilities available")
        
        if class_name is None:
            return self.prob
        
        # Get probability for specific class
        if not self.class_names:
            raise RuntimeError("Cannot get class probabilities without class names")
        
        try:
            class_idx = self.class_names.index(class_name)
            return self.prob[:, class_idx]
        except ValueError:
            raise ValueError(f"Class '{class_name}' not found in {self.class_names}")
    
    def confusion_matrix(self) -> Optional[np.ndarray]:
        """
        Calculate confusion matrix.
        
        Returns
        -------
        np.ndarray or None
            Confusion matrix (true x predicted)
        """
        if not self.has_truth or self.response is None:
            return None
        
        if not self.class_names:
            return None
        
        # Create confusion matrix
        n_classes = len(self.class_names)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        # Map labels to indices
        label_to_idx = {label: i for i, label in enumerate(self.class_names)}
        
        for true_label, pred_label in zip(self.truth, self.response):
            true_idx = label_to_idx.get(str(true_label))
            pred_idx = label_to_idx.get(str(pred_label))
            
            if true_idx is not None and pred_idx is not None:
                cm[true_idx, pred_idx] += 1
        
        return cm
    
    def _get_prediction_data(self) -> Dict[str, np.ndarray]:
        """Get prediction data for DataFrame."""
        data = {}
        
        if self.response is not None:
            data["response"] = self.response
        
        if self.prob is not None and self.class_names:
            # Add probability columns
            for i, class_name in enumerate(self.class_names):
                data[f"prob.{class_name}"] = self.prob[:, i]
        
        return data
    
    @property
    def _properties(self) -> set[str]:
        """Classification prediction properties."""
        props = super()._properties
        props.add("classif")
        
        if self.n_classes == 2:
            props.add("binary")
        else:
            props.add("multiclass")
            
        return props


class PredictionRegr(Prediction):
    """
    Regression predictions.
    
    Parameters
    ----------
    row_ids : List[int]
        Row IDs
    truth : array-like, optional
        True values
    response : array-like
        Predicted values
    se : array-like, optional
        Standard errors of predictions
    task : TaskRegr, optional
        Regression task
    """
    
    def __init__(
        self,
        row_ids: List[int],
        truth: Optional[np.ndarray] = None,
        response: Optional[np.ndarray] = None,
        se: Optional[np.ndarray] = None,
        task: Optional["TaskRegr"] = None,
        **kwargs
    ):
        super().__init__(row_ids=row_ids, truth=truth, task=task, **kwargs)
        
        if response is None:
            raise ValueError("Response predictions are required for regression")
        
        self.response = np.array(response)
        self.se = np.array(se) if se is not None else None
        
        # Validate
        if len(self.response) != self.n:
            raise ValueError(
                f"Length mismatch: {self.n} predictions vs {len(self.response)} responses"
            )
        
        if self.se is not None and len(self.se) != self.n:
            raise ValueError(
                f"Length mismatch: {self.n} predictions vs {len(self.se)} standard errors"
            )
    
    @property
    def predict_types(self) -> Dict[str, bool]:
        """Available prediction types."""
        return {
            "response": True,  # Always available for regression
            "se": self.se is not None,
        }
    
    def residuals(self) -> Optional[np.ndarray]:
        """
        Calculate residuals (truth - predicted).
        
        Returns
        -------
        np.ndarray or None
            Residuals if truth is available
        """
        if not self.has_truth:
            return None
        
        return self.truth - self.response
    
    def abs_residuals(self) -> Optional[np.ndarray]:
        """
        Calculate absolute residuals.
        
        Returns
        -------
        np.ndarray or None
            Absolute residuals if truth is available
        """
        residuals = self.residuals()
        return np.abs(residuals) if residuals is not None else None
    
    def prediction_intervals(
        self, 
        alpha: float = 0.05
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Calculate prediction intervals.
        
        Parameters
        ----------
        alpha : float
            Significance level (default 0.05 for 95% intervals)
            
        Returns
        -------
        Dict[str, np.ndarray] or None
            Dictionary with 'lower' and 'upper' bounds
        """
        if self.se is None:
            return None
        
        # Assume normal distribution
        from scipy import stats
        z = stats.norm.ppf(1 - alpha / 2)
        
        return {
            "lower": self.response - z * self.se,
            "upper": self.response + z * self.se,
        }
    
    def _get_prediction_data(self) -> Dict[str, np.ndarray]:
        """Get prediction data for DataFrame."""
        data = {"response": self.response}
        
        if self.se is not None:
            data["se"] = self.se
        
        return data
    
    @property
    def _properties(self) -> set[str]:
        """Regression prediction properties."""
        props = super()._properties
        props.add("regr")
        
        if self.se is not None:
            props.add("se")
            
        return props


__all__ = ["Prediction", "PredictionClassif", "PredictionRegr"]