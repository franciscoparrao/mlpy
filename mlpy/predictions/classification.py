"""Classification predictions for MLPY."""

from typing import Optional, List, Any
import numpy as np
import pandas as pd

from .base import Prediction


class PredictionClassif(Prediction):
    """Predictions for classification tasks.
    
    Parameters
    ----------
    task : TaskClassif
        The classification task.
    learner_id : str
        ID of the learner that made predictions.
    row_ids : array-like
        Row indices for the predictions.
    truth : array-like
        True class labels.
    response : array-like, optional
        Predicted class labels.
    prob : DataFrame, optional
        Predicted probabilities for each class.
    """
    
    def __init__(
        self,
        task: "TaskClassif",
        learner_id: str,
        row_ids: List[int],
        truth: Any,
        response: Optional[Any] = None,
        prob: Optional[pd.DataFrame] = None,
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
        self.response = np.array(response) if response is not None else None
        self.prob = prob
        
        # Validate
        if response is None and prob is None:
            raise ValueError("At least one of response or prob must be provided")
            
        if self.response is not None and len(self.response) != self.n:
            raise ValueError(
                f"response length ({len(self.response)}) does not match "
                f"number of observations ({self.n})"
            )
            
        if self.prob is not None:
            if len(self.prob) != self.n:
                raise ValueError(
                    f"prob length ({len(self.prob)}) does not match "
                    f"number of observations ({self.n})"
                )
                
    @property
    def predict_type(self) -> str:
        """Type of prediction."""
        if self.prob is not None:
            return "prob"
        return "response"
        
    def get_response(self) -> np.ndarray:
        """Get predicted class labels.
        
        If only probabilities are available, converts them to
        class labels by taking the argmax.
        
        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        if self.response is not None:
            return self.response
            
        # Convert probabilities to response
        if self.prob is not None:
            # Get class with highest probability
            return self.prob.idxmax(axis=1).values
            
        raise RuntimeError("No predictions available")
        
    def get_prob(self, class_name: Optional[str] = None) -> Any:
        """Get predicted probabilities.
        
        Parameters
        ----------
        class_name : str, optional
            Specific class to get probabilities for.
            If None, returns all probabilities.
            
        Returns
        -------
        array-like
            Probabilities for the specified class or all classes.
        """
        if self.prob is None:
            raise RuntimeError("No probability predictions available")
            
        if class_name is not None:
            if class_name not in self.prob.columns:
                raise ValueError(f"Class '{class_name}' not found in predictions")
            return self.prob[class_name].values
            
        return self.prob