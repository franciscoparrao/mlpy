"""Base prediction class for MLPY.

This module provides the abstract base class for all predictions
in the MLPY framework.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Any
import numpy as np
import pandas as pd

from ..base import MLPYObject


class Prediction(MLPYObject, ABC):
    """Abstract base class for predictions.
    
    Stores the predicted values along with the true values
    and metadata about the prediction.
    
    Parameters
    ----------
    task : Task
        The task the predictions are for.
    learner_id : str
        ID of the learner that made the predictions.
    row_ids : array-like
        Row indices for the predictions.
    truth : array-like
        True target values.
    """
    
    def __init__(
        self,
        task: Optional["Task"],
        learner_id: str,
        row_ids: List[int],
        truth: Any,
        **kwargs
    ):
        # Handle None task for direct score calculations
        if task is None:
            pred_id = f"pred_{learner_id}_direct"
        else:
            pred_id = f"pred_{learner_id}_{task.id}"
        
        super().__init__(id=pred_id)
        
        self.task = task
        self.learner_id = learner_id
        self.row_ids = np.array(row_ids)
        self.truth = np.array(truth)
        
        # Store additional prediction data
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    @property
    @abstractmethod
    def predict_type(self) -> str:
        """Type of prediction (e.g., 'response', 'prob')."""
        pass
        
    @property
    def n(self) -> int:
        """Number of predictions."""
        return len(self.row_ids)
        
    def __repr__(self) -> str:
        """String representation."""
        return f"<{self.__class__.__name__}> for {self.n} observations"