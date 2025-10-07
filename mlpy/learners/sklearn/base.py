"""Base class for scikit-learn learner wrappers."""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from abc import abstractmethod
import warnings
import copy

from ..base import Learner
from ..classification import LearnerClassif
from ..regression import LearnerRegr
from ...tasks import Task, TaskClassif, TaskRegr
from ...predictions import PredictionClassif, PredictionRegr


class LearnerSKLearn(Learner):
    """Base wrapper class for scikit-learn models.
    
    This class provides a common interface for wrapping scikit-learn estimators
    to work within the MLPY framework.
    
    Parameters
    ----------
    estimator_class : type
        The scikit-learn estimator class (not instance).
    id : str, optional
        Unique identifier for the learner.
    predict_type : str, optional
        Type of prediction for classification ("response" or "prob").
    **kwargs
        Parameters to pass to the estimator constructor.
    """
    
    def __init__(
        self,
        estimator_class: type,
        id: Optional[str] = None,
        predict_type: str = "response",
        **kwargs
    ):
        """Initialize sklearn learner wrapper."""
        # Set ID based on estimator class name if not provided
        if id is None:
            id = estimator_class.__name__.lower()
            
        super().__init__(
            id=id,
            predict_type=predict_type
        )
        
        self.estimator_class = estimator_class
        self.estimator_params = kwargs
        self.estimator = None
        
        # Store original parameters for cloning
        self._init_params = {
            'id': id,
            'predict_type': predict_type,
            **kwargs
        }
        
    @staticmethod
    def _infer_task_type(estimator_class: type) -> str:
        """Infer task type from estimator class.
        
        Parameters
        ----------
        estimator_class : type
            Scikit-learn estimator class.
            
        Returns
        -------
        str
            Task type ("classif" or "regr").
        """
        # Check class name and base classes
        class_name = estimator_class.__name__.lower()
        
        # Common patterns
        if any(x in class_name for x in ['classifier', 'classif', 'logistic']):
            return "classif"
        elif any(x in class_name for x in ['regressor', 'regr', 'regression']):
            return "regr"
            
        # Check for _estimator_type attribute (sklearn convention)
        if hasattr(estimator_class, '_estimator_type'):
            if estimator_class._estimator_type == 'classifier':
                return "classif"
            elif estimator_class._estimator_type == 'regressor':
                return "regr"
                
        # Default based on common sklearn patterns
        raise ValueError(
            f"Cannot infer task type for {estimator_class.__name__}. "
            "Please use LearnerClassifSKLearn or LearnerRegrSKLearn directly."
        )
        
    def _train(self, task: Task, row_ids: Optional[List[int]] = None) -> "LearnerSKLearn":
        """Train the scikit-learn model.
        
        Parameters
        ----------
        task : Task
            The task containing training data.
        row_ids : list of int, optional
            Subset of rows to use for training.
            
        Returns
        -------
        self : LearnerSKLearn
            The trained learner.
        """
        # Get training data
        if row_ids is not None:
            X = task.data(rows=row_ids, cols=task.feature_names)
            y = task.truth(rows=row_ids)
        else:
            X = task.data(cols=task.feature_names)
            y = task.truth()
        
        # Convert to numpy if needed (sklearn typically prefers numpy)
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Create estimator instance
        self.estimator = self.estimator_class(**self.estimator_params)
        
        # Fit the model
        self.estimator.fit(X, y)
        
        # Store feature names for later use
        self.feature_names = task.feature_names
        
        # Mark as trained
        self._model = self.estimator
        self._train_task = task
        
        return self
        
    def _predict(self, task: Task, row_ids: Optional[List[int]] = None):
        """Make predictions using the trained model.

        Parameters
        ----------
        task : Task
            The task containing test data.
        row_ids : list of int, optional
            Subset of rows to predict.

        Returns
        -------
        Prediction
            Predictions for the given task.
        """
        if not self.is_trained:
            raise ValueError("Learner must be trained before making predictions")

        # Get test data
        if row_ids is not None:
            X = task.data(rows=row_ids, cols=task.feature_names)
            truth = task.truth(rows=row_ids)
        else:
            X = task.data(cols=task.feature_names)
            truth = task.truth()
            row_ids = list(range(len(X)))

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Make predictions based on task type
        if self.task_type == "classif":
            return self._predict_classif(X, task, row_ids, truth)
        else:
            return self._predict_regr(X, task, row_ids, truth)
            
    def _predict_classif(self, X: np.ndarray, task: TaskClassif, row_ids: List[int], truth: np.ndarray) -> PredictionClassif:
        """Make classification predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        task : TaskClassif
            Classification task.
            
        Returns
        -------
        PredictionClassif
            Classification predictions.
        """
        if self.predict_type == "response":
            # Class predictions
            response = self.estimator.predict(X)
            prob = None
        else:
            # Probability predictions
            if hasattr(self.estimator, 'predict_proba'):
                prob = self.estimator.predict_proba(X)
                # Also get class predictions
                response = self.estimator.predict(X)
                
                # Convert to DataFrame with class names as columns
                if hasattr(self.estimator, 'classes_'):
                    prob = pd.DataFrame(
                        prob,
                        columns=self.estimator.classes_,
                        index=task.data.index if hasattr(task.data, 'index') else None
                    )
            else:
                warnings.warn(
                    f"{self.estimator.__class__.__name__} does not support "
                    "probability predictions. Falling back to response."
                )
                response = self.estimator.predict(X)
                prob = None
                
        # Convert response to pandas Series
        if not isinstance(response, pd.Series):
            response = pd.Series(
                response,
                index=task.data.index if hasattr(task.data, 'index') else None
            )
            
        return PredictionClassif(
            task=task,
            learner_id=self.id,
            row_ids=row_ids,
            truth=truth,
            response=response,
            prob=prob
        )
        
    def _predict_regr(self, X: np.ndarray, task: TaskRegr, row_ids: List[int], truth: np.ndarray) -> PredictionRegr:
        """Make regression predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        task : TaskRegr
            Regression task.
            
        Returns
        -------
        PredictionRegr
            Regression predictions.
        """
        # Make predictions
        response = self.estimator.predict(X)
        
        # Get prediction intervals if available
        se = None
        if hasattr(self.estimator, 'predict_interval'):
            # Some sklearn models might have this
            lower, upper = self.estimator.predict_interval(X)
            # Estimate SE from interval (assuming normal distribution)
            se = (upper - lower) / (2 * 1.96)
        elif hasattr(self.estimator, 'predict_std'):
            # Some models provide standard deviation
            _, se = self.estimator.predict_std(X, return_std=True)
            
        # Convert to pandas
        if not isinstance(response, pd.Series):
            response = pd.Series(
                response,
                index=task.data.index if hasattr(task.data, 'index') else None
            )
        if se is not None and not isinstance(se, pd.Series):
            se = pd.Series(
                se,
                index=task.data.index if hasattr(task.data, 'index') else None
            )
            
        return PredictionRegr(
            task=task,
            learner_id=self.id,
            row_ids=row_ids,
            truth=truth,
            response=response,
            se=se
        )
        
    def clone(self) -> "LearnerSKLearn":
        """Create a deep copy of the learner.
        
        Returns
        -------
        LearnerSKLearn
            A new instance with the same parameters.
        """
        return self.__class__(**self._init_params)
        
    def reset(self) -> "LearnerSKLearn":
        """Reset the learner to untrained state.
        
        Returns
        -------
        self : LearnerSKLearn
            The reset learner.
        """
        self.estimator = None
        self.model = None
        self.is_trained = False
        return self
        
    def get_params(self) -> Dict[str, Any]:
        """Get learner parameters.
        
        Returns
        -------
        dict
            Dictionary of parameters.
        """
        params = {
            'id': self.id,
            'predict_type': self.predict_type,
            **self.estimator_params
        }
        return params
        
    def set_params(self, **params) -> "LearnerSKLearn":
        """Set learner parameters.
        
        Parameters
        ----------
        **params
            Parameters to set.
            
        Returns
        -------
        self : LearnerSKLearn
            The learner with updated parameters.
        """
        # Handle special parameters
        if 'id' in params:
            self.id = params.pop('id')
        if 'predict_type' in params:
            self.predict_type = params.pop('predict_type')
            
        # Rest are estimator parameters
        self.estimator_params.update(params)
        
        # Update init params for cloning
        self._init_params.update(params)
        
        # If already trained, we need to retrain with new params
        if self.is_trained:
            warnings.warn(
                "Learner was already trained. Parameters updated but model "
                "needs to be retrained for changes to take effect."
            )
            
        return self
        
    @property
    def model(self):
        """Get the underlying sklearn model."""
        return self.estimator
        
    @model.setter 
    def model(self, value):
        """Set the underlying sklearn model."""
        self.estimator = value
        

class LearnerClassifSKLearn(LearnerSKLearn, LearnerClassif):
    """Base wrapper for scikit-learn classification models."""
    
    def __init__(
        self,
        estimator_class: type,
        id: Optional[str] = None,
        predict_type: str = "response",
        **kwargs
    ):
        """Initialize classification wrapper."""
        super().__init__(
            estimator_class=estimator_class,
            id=id,
            predict_type=predict_type,
            **kwargs
        )
        

class LearnerRegrSKLearn(LearnerSKLearn, LearnerRegr):
    """Base wrapper for scikit-learn regression models."""
    
    def __init__(
        self,
        estimator_class: type,
        id: Optional[str] = None,
        **kwargs
    ):
        """Initialize regression wrapper."""
        # Remove predict_type from kwargs if present
        kwargs.pop('predict_type', None)
        
        super().__init__(
            estimator_class=estimator_class,
            id=id,
            predict_type="response",  # Regression only has response
            **kwargs
        )
