"""Base regression learner for MLPY.

This module provides the abstract base class for regression learners.
"""

from abc import ABC
from typing import Optional, List, Dict, Any

from .base import Learner
from ..tasks import TaskRegr
from ..predictions import PredictionRegr


class LearnerRegr(Learner, ABC):
    """Abstract base class for regression learners.
    
    This class extends the base Learner class specifically for
    regression tasks.
    """
    
    @property
    def task_type(self) -> str:
        """Type of task this learner can handle."""
        return "regr"
    
    def train(self, task: TaskRegr, row_ids: Optional[List[int]] = None) -> "LearnerRegr":
        """Train the learner on a regression task.
        
        Parameters
        ----------
        task : TaskRegr
            The regression task to train on.
        row_ids : list of int, optional
            Subset of rows to use for training.
            
        Returns
        -------
        self : LearnerRegr
            The trained learner.
        """
        if not isinstance(task, TaskRegr):
            raise TypeError(f"Expected TaskRegr, got {type(task).__name__}")
        
        # Delegate to subclass implementation
        return self._train(task, row_ids)
    
    def predict(self, task: TaskRegr, row_ids: Optional[List[int]] = None) -> PredictionRegr:
        """Make predictions on a regression task.
        
        Parameters
        ----------
        task : TaskRegr
            The task to predict on.
        row_ids : list of int, optional
            Subset of rows to predict.
            
        Returns
        -------
        PredictionRegr
            The predictions.
        """
        if not isinstance(task, TaskRegr):
            raise TypeError(f"Expected TaskRegr, got {type(task).__name__}")
        
        # Delegate to subclass implementation
        return self._predict(task, row_ids)
    
    def _train(self, task: TaskRegr, row_ids: Optional[List[int]] = None) -> "LearnerRegr":
        """Subclass implementation of training."""
        raise NotImplementedError
    
    def _predict(self, task: TaskRegr, row_ids: Optional[List[int]] = None) -> PredictionRegr:
        """Subclass implementation of prediction."""
        raise NotImplementedError