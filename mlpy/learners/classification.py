"""Base classification learner for MLPY.

This module provides the abstract base class for classification learners.
"""

from abc import ABC
from typing import Optional, List, Dict, Any

from .base import Learner
from ..tasks import TaskClassif
from ..predictions import PredictionClassif


class LearnerClassif(Learner, ABC):
    """Abstract base class for classification learners.
    
    This class extends the base Learner class specifically for
    classification tasks.
    """
    
    @property
    def task_type(self) -> str:
        """Type of task this learner can handle."""
        return "classif"
    
    def train(self, task: TaskClassif, row_ids: Optional[List[int]] = None) -> "LearnerClassif":
        """Train the learner on a classification task.
        
        Parameters
        ----------
        task : TaskClassif
            The classification task to train on.
        row_ids : list of int, optional
            Subset of rows to use for training.
            
        Returns
        -------
        self : LearnerClassif
            The trained learner.
        """
        if not isinstance(task, TaskClassif):
            raise TypeError(f"Expected TaskClassif, got {type(task).__name__}")
        
        # Delegate to subclass implementation
        return self._train(task, row_ids)
    
    def predict(self, task: TaskClassif, row_ids: Optional[List[int]] = None) -> PredictionClassif:
        """Make predictions on a classification task.
        
        Parameters
        ----------
        task : TaskClassif
            The task to predict on.
        row_ids : list of int, optional
            Subset of rows to predict.
            
        Returns
        -------
        PredictionClassif
            The predictions.
        """
        if not isinstance(task, TaskClassif):
            raise TypeError(f"Expected TaskClassif, got {type(task).__name__}")
        
        # Delegate to subclass implementation
        return self._predict(task, row_ids)
    
    def _train(self, task: TaskClassif, row_ids: Optional[List[int]] = None) -> "LearnerClassif":
        """Subclass implementation of training."""
        raise NotImplementedError
    
    def _predict(self, task: TaskClassif, row_ids: Optional[List[int]] = None) -> PredictionClassif:
        """Subclass implementation of prediction."""
        raise NotImplementedError