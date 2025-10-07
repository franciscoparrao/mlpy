"""Regression predictions for MLPY."""

from typing import Optional, List, Any
import numpy as np

from .base import Prediction


class PredictionRegr(Prediction):
    """Predictions for regression tasks.
    
    Parameters
    ----------
    task : TaskRegr
        The regression task.
    learner_id : str
        ID of the learner that made predictions.
    row_ids : array-like
        Row indices for the predictions.
    truth : array-like
        True target values.
    response : array-like
        Predicted values.
    se : array-like, optional
        Standard errors of predictions.
    """
    
    def __init__(
        self,
        task: "TaskRegr",
        learner_id: str,
        row_ids: List[int],
        truth: Any,
        response: Any,
        se: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(
            task=task,
            learner_id=learner_id,
            row_ids=row_ids,
            truth=truth,
            **kwargs
        )
        
        # Store predictions
        self.response = np.array(response)
        self.se = np.array(se) if se is not None else None
        
        # Validate
        if len(self.response) != self.n:
            raise ValueError(
                f"response length ({len(self.response)}) does not match "
                f"number of observations ({self.n})"
            )
            
        if self.se is not None and len(self.se) != self.n:
            raise ValueError(
                f"se length ({len(self.se)}) does not match "
                f"number of observations ({self.n})"
            )
            
    @property
    def predict_type(self) -> str:
        """Type of prediction."""
        if self.se is not None:
            return "se"
        return "response"
        
    def get_response(self) -> np.ndarray:
        """Get predicted values.
        
        Returns
        -------
        np.ndarray
            Predicted values.
        """
        return self.response
        
    def get_se(self) -> Optional[np.ndarray]:
        """Get standard errors.
        
        Returns
        -------
        np.ndarray or None
            Standard errors if available.
        """
        return self.se