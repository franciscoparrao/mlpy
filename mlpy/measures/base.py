"""Base classes for measures in MLPY."""

from abc import abstractmethod
from typing import Optional, List, Union, Dict, Any, Type
import numpy as np
import pandas as pd

from ..base import MLPYObject
from ..predictions import Prediction, PredictionClassif, PredictionRegr
from ..utils.registry import mlpy_measures


class Measure(MLPYObject):
    """Abstract base class for all measures.
    
    A measure quantifies the performance of predictions. Each measure operates on a
    specific type of prediction (classification or regression) and can have properties
    like whether lower values are better (minimize) or higher values are better (maximize).
    
    Parameters
    ----------
    id : str
        Unique identifier for the measure.
    param_set : dict, optional
        Parameters for the measure.
    minimize : bool, default=None
        Whether lower values are better. If None, determined by measure type.
    range : tuple, optional
        Valid range of the measure as (lower, upper). Use -np.inf or np.inf for unbounded.
    properties : set, optional
        Set of properties like {'requires_task', 'requires_model'}.
    predict_type : str, optional
        Required prediction type (e.g., 'response', 'prob').
    predict_sets : set, optional
        Which sets to calculate on ('train', 'test'). Default is {'test'}.
    task_type : str, optional
        Compatible task type ('classif' or 'regr').
    average : str, optional
        Averaging method for multiclass problems ('macro', 'micro', 'weighted').
    """
    
    @property
    def _properties(self):
        """Return measure properties for hashing."""
        return self.properties if hasattr(self, 'properties') else set()
    
    def __init__(
        self,
        id: str,
        param_set: Optional[Dict[str, Any]] = None,
        minimize: Optional[bool] = None,
        range: Optional[tuple] = None,
        properties: Optional[set] = None,
        predict_type: Optional[str] = None,
        predict_sets: Optional[set] = None,
        task_type: Optional[str] = None,
        average: Optional[str] = None
    ):
        super().__init__(id=id)
        self.param_set = param_set if param_set is not None else {}
        self.minimize = minimize
        self.range = range if range is not None else (-np.inf, np.inf)
        self.properties = properties if properties is not None else set()
        self.predict_type = predict_type
        self.predict_sets = predict_sets if predict_sets is not None else {'test'}
        self.task_type = task_type
        self.average = average
        
    @abstractmethod
    def _score(self, prediction: Prediction, task=None, **kwargs) -> float:
        """Calculate the measure score.
        
        Parameters
        ----------
        prediction : Prediction
            The prediction object to score.
        task : Task, optional
            The task object, required for some measures.
        **kwargs
            Additional arguments for the measure.
            
        Returns
        -------
        float
            The calculated score.
        """
        pass
    
    def score(self, prediction, task=None, **kwargs) -> float:
        """Calculate the measure score with validation.
        
        Parameters
        ----------
        prediction : Prediction, array-like, or two arrays (y_true, y_pred)
            The prediction object to score, or raw predictions.
            Can be:
            - Prediction object
            - Single array/list of predictions (requires 'truth' in kwargs or as first arg)
            - Two arrays/lists (y_true, y_pred) when called as score(y_true, y_pred)
        task : Task, optional
            The task object, required for some measures.
            If prediction is array-like, task can be y_pred (second argument).
        **kwargs
            Additional arguments for the measure.
            
        Returns
        -------
        float
            The calculated score.
        """
        # Handle different input formats
        if isinstance(prediction, (list, np.ndarray, pd.Series)):
            # prediction is y_true, task might be y_pred
            if task is not None and isinstance(task, (list, np.ndarray, pd.Series)):
                # score(y_true, y_pred) format - task is actually y_pred
                y_true = prediction
                y_pred = task
                # Create a simple prediction object
                if self.task_type == 'classif':
                    prediction = PredictionClassif(
                        task=None,
                        learner_id="direct",
                        row_ids=list(range(len(y_pred))),
                        truth=y_true,
                        response=y_pred
                    )
                else:
                    prediction = PredictionRegr(
                        task=None,
                        learner_id="direct",
                        row_ids=list(range(len(y_pred))),
                        truth=y_true,
                        response=y_pred
                    )
            else:
                # Single array with truth in kwargs
                y_pred = prediction
                y_true = kwargs.get('truth', kwargs.get('y_true', None))
                if y_true is None:
                    raise ValueError("Truth values required for scoring")
                
                if self.task_type == 'classif':
                    prediction = PredictionClassif(
                        task=None,
                        learner_id="direct",
                        row_ids=list(range(len(y_pred))),
                        truth=y_true,
                        response=y_pred
                    )
                else:
                    prediction = PredictionRegr(
                        task=None,
                        learner_id="direct",
                        row_ids=list(range(len(y_pred))),
                        truth=y_true,
                        response=y_pred
                    )
        
        # Validate prediction type
        if self.task_type == 'classif' and not isinstance(prediction, PredictionClassif):
            raise TypeError(f"{self.id} requires PredictionClassif, got {type(prediction)}")
        if self.task_type == 'regr' and not isinstance(prediction, PredictionRegr):
            raise TypeError(f"{self.id} requires PredictionRegr, got {type(prediction)}")
            
        # Check if task is required
        if 'requires_task' in self.properties and task is None:
            raise ValueError(f"{self.id} requires task to be provided")
            
        # Check prediction type
        if self.predict_type == 'prob' and isinstance(prediction, PredictionClassif):
            if prediction.prob is None:
                raise ValueError(f"{self.id} requires probability predictions")
                
        # Calculate score
        score = self._score(prediction, task=task, **kwargs)
        
        # Validate range (skip validation for NaN)
        if not np.isnan(score) and not (self.range[0] <= score <= self.range[1]):
            raise ValueError(f"Score {score} outside valid range {self.range}")
            
        return score
    
    def is_applicable(self, task) -> bool:
        """Check if this measure is applicable to a given task.
        
        Parameters
        ----------
        task : Task
            The task to check compatibility with.
            
        Returns
        -------
        bool
            True if the measure can be used with this task.
        """
        # Check task type compatibility
        if hasattr(task, 'task_type'):
            return task.task_type == self.task_type
        return False
    
    def aggregate(self, scores: List[float]) -> Dict[str, float]:
        """Aggregate multiple scores into summary statistics.
        
        Parameters
        ----------
        scores : list of float
            List of scores to aggregate.
            
        Returns
        -------
        dict
            Dictionary with aggregated statistics (mean, std, min, max, median).
        """
        if not scores:
            return {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "median": np.nan
            }
        
        scores_array = np.array(scores)
        return {
            "mean": np.mean(scores_array),
            "std": np.std(scores_array, ddof=1) if len(scores_array) > 1 else 0.0,
            "min": np.min(scores_array),
            "max": np.max(scores_array),
            "median": np.median(scores_array)
        }
    
    def __repr__(self):
        return f"<{self.__class__.__name__}:{self.id}>"


class MeasureClassif(Measure):
    """Base class for classification measures.
    
    Sets default task_type to 'classif'.
    """
    
    def __init__(self, **kwargs):
        kwargs.setdefault('task_type', 'classif')
        super().__init__(**kwargs)


class MeasureRegr(Measure):
    """Base class for regression measures.
    
    Sets default task_type to 'regr'.
    """
    
    def __init__(self, **kwargs):
        kwargs.setdefault('task_type', 'regr')
        super().__init__(**kwargs)


class MeasureSimple(Measure):
    """Simple measure that uses a scoring function.
    
    Parameters
    ----------
    id : str
        Unique identifier for the measure.
    score_func : callable
        Function that takes (truth, response) and returns score.
    **kwargs
        Additional arguments passed to Measure.
    """
    
    def __init__(self, id: str, score_func, **kwargs):
        super().__init__(id=id, **kwargs)
        self.score_func = score_func
        
    def _score(self, prediction: Prediction, task=None, **kwargs) -> float:
        """Calculate score using the provided function."""
        if prediction.truth is None:
            raise ValueError("Prediction must have truth values for scoring")
            
        truth = prediction.truth
        response = prediction.response
        
        # Handle missing values
        mask = ~(pd.isna(truth) | pd.isna(response))
        if not mask.any():
            return np.nan
            
        return self.score_func(truth[mask], response[mask])


def register_measure(cls: Type[Measure]) -> Type[Measure]:
    """Decorator to register a measure class.
    
    Parameters
    ----------
    cls : type
        The measure class to register.
        
    Returns
    -------
    type
        The unchanged class.
    """
    # Create instance with default parameters
    instance = cls()
    mlpy_measures[instance.id] = instance
    return cls