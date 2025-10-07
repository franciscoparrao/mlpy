"""Scikit-learn integration for MLPY.

This module provides learners that wrap scikit-learn estimators,
allowing them to be used seamlessly within the MLPY framework.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union
from copy import deepcopy
import warnings

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from .base import Learner
from ..tasks import Task, TaskClassif, TaskRegr
from ..predictions import PredictionClassif, PredictionRegr
from ..utils.registry import mlpy_learners


class LearnerSklearn(Learner):
    """Base learner for scikit-learn integration.
    
    This class wraps scikit-learn estimators to work with MLPY's
    Task/Learner/Prediction interface.
    
    Parameters
    ----------
    estimator : sklearn estimator
        A scikit-learn estimator instance (classifier or regressor).
    id : str, optional
        Unique identifier. If None, uses estimator class name.
    predict_type : str, optional
        Type of prediction ('response' or 'prob' for classifiers).
    **kwargs
        Additional parameters passed to parent class.
    """
    
    def __init__(
        self,
        estimator: BaseEstimator,
        id: Optional[str] = None,
        predict_type: str = "response",
        **kwargs
    ):
        # Validate estimator
        if not isinstance(estimator, BaseEstimator):
            raise TypeError(
                f"estimator must be a scikit-learn BaseEstimator, "
                f"got {type(estimator)}"
            )
            
        # Auto-generate ID if not provided
        if id is None:
            id = f"sklearn.{estimator.__class__.__name__.lower()}"
            
        # Detect properties
        properties = self._detect_properties(estimator)
        
        # Detect required packages
        packages = self._detect_packages(estimator)
        
        super().__init__(
            id=id,
            predict_type=predict_type,
            properties=properties,
            packages=packages,
            **kwargs
        )
        
        self.estimator = estimator
        self._fitted_estimator = None
        
    def _detect_properties(self, estimator: BaseEstimator) -> set:
        """Auto-detect learner properties from sklearn estimator.
        
        Parameters
        ----------
        estimator : BaseEstimator
            The scikit-learn estimator.
            
        Returns
        -------
        set
            Set of detected properties.
        """
        properties = set()
        
        # Check basic sklearn interfaces
        if hasattr(estimator, 'predict_proba'):
            properties.add('prob')
            
        if hasattr(estimator, 'predict_log_proba'):
            properties.add('log_prob')
            
        if hasattr(estimator, 'decision_function'):
            properties.add('decision')
            
        if hasattr(estimator, 'feature_importances_'):
            properties.add('importance')
            
        if hasattr(estimator, 'oob_score_'):
            properties.add('oob')
            
        # Check for specific algorithm types
        estimator_name = estimator.__class__.__name__.lower()
        
        if 'tree' in estimator_name or 'forest' in estimator_name:
            properties.add('tree_based')
            
        if 'linear' in estimator_name or 'logistic' in estimator_name:
            properties.add('linear')
            
        if 'svm' in estimator_name or 'svc' in estimator_name:
            properties.add('kernel')
            
        if 'boost' in estimator_name:
            properties.add('boosting')
            
        if 'bagging' in estimator_name or 'forest' in estimator_name:
            properties.add('ensemble')
            
        return properties
        
    def _detect_packages(self, estimator: BaseEstimator) -> List[str]:
        """Detect required packages from estimator.
        
        Parameters
        ----------
        estimator : BaseEstimator
            The scikit-learn estimator.
            
        Returns
        -------
        list of str
            List of required packages.
        """
        packages = ['scikit-learn']
        
        # Check if it's from a specific sklearn submodule
        module = estimator.__class__.__module__
        
        if 'xgboost' in module:
            packages.append('xgboost')
        elif 'lightgbm' in module:
            packages.append('lightgbm')
        elif 'catboost' in module:
            packages.append('catboost')
            
        return packages
        
    @property
    def task_type(self) -> str:
        """Type of task this learner can handle."""
        if isinstance(self.estimator, ClassifierMixin):
            return 'classif'
        elif isinstance(self.estimator, RegressorMixin):
            return 'regr'
        else:
            # Try to infer from method names
            if hasattr(self.estimator, 'predict_proba'):
                return 'classif'
            else:
                return 'regr'
                
    def train(self, task: Task, row_ids: Optional[List[int]] = None) -> "LearnerSklearn":
        """Train the sklearn estimator.
        
        Parameters
        ----------
        task : Task
            The task to train on.
        row_ids : list of int, optional
            Subset of rows to use for training.
            
        Returns
        -------
        self : LearnerSklearn
            The trained learner.
        """
        # Validate task type
        if self.task_type == 'classif' and not isinstance(task, TaskClassif):
            raise TypeError(f"Classifier requires TaskClassif, got {type(task)}")
        elif self.task_type == 'regr' and not isinstance(task, TaskRegr):
            raise TypeError(f"Regressor requires TaskRegr, got {type(task)}")
            
        # Get training data
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        X = task.data(rows=row_ids, cols=task.feature_names)
        y = task.truth(rows=row_ids)
        
        # Clone estimator for fresh training
        self._fitted_estimator = deepcopy(self.estimator)
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Train the model
        try:
            self._fitted_estimator.fit(X, y)
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}") from e
            
        # Store training info
        self._model = self._fitted_estimator
        self._train_task = task
        
        return self
        
    def predict(self, task: Task, row_ids: Optional[List[int]] = None) -> Union[PredictionClassif, PredictionRegr]:
        """Make predictions using the trained estimator.
        
        Parameters
        ----------
        task : Task
            The task to predict on.
        row_ids : list of int, optional
            Subset of rows to predict.
            
        Returns
        -------
        Prediction
            The predictions.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
            
        # Get prediction data
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        X = task.data(rows=row_ids, cols=task.feature_names)
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Get truth values
        truth = task.truth(rows=row_ids)
        
        if self.task_type == 'classif':
            return self._predict_classif(X, truth, task, row_ids)
        else:
            return self._predict_regr(X, truth, task, row_ids)
            
    def _predict_classif(
        self, 
        X: np.ndarray, 
        truth: np.ndarray,
        task: TaskClassif,
        row_ids: List[int]
    ) -> PredictionClassif:
        """Make classification predictions.
        
        Parameters
        ----------
        X : array-like
            Feature matrix.
        truth : array-like
            True labels.
        task : TaskClassif
            The classification task.
        row_ids : list of int
            Row indices.
            
        Returns
        -------
        PredictionClassif
            Classification predictions.
        """
        # Get response predictions
        response = self._fitted_estimator.predict(X)
        
        # Get probability predictions if requested and available
        prob = None
        if self.predict_type == 'prob' and hasattr(self._fitted_estimator, 'predict_proba'):
            prob_array = self._fitted_estimator.predict_proba(X)
            
            # Get class names from sklearn model
            if hasattr(self._fitted_estimator, 'classes_'):
                classes = self._fitted_estimator.classes_
            else:
                classes = task.class_names
                
            # Create probability DataFrame
            prob = pd.DataFrame(prob_array, columns=classes)
            
        return PredictionClassif(
            task=task,
            learner_id=self.id,
            row_ids=row_ids,
            truth=truth,
            response=response,
            prob=prob
        )
        
    def _predict_regr(
        self,
        X: np.ndarray,
        truth: np.ndarray,
        task: TaskRegr,
        row_ids: List[int]
    ) -> PredictionRegr:
        """Make regression predictions.
        
        Parameters
        ----------
        X : array-like
            Feature matrix.
        truth : array-like
            True values.
        task : TaskRegr
            The regression task.
        row_ids : list of int
            Row indices.
            
        Returns
        -------
        PredictionRegr
            Regression predictions.
        """
        # Get response predictions
        response = self._fitted_estimator.predict(X)
        
        # Some sklearn models provide standard errors
        se = None
        if hasattr(self._fitted_estimator, 'predict_std'):
            se = self._fitted_estimator.predict_std(X)
            
        return PredictionRegr(
            task=task,
            learner_id=self.id,
            row_ids=row_ids,
            truth=truth,
            response=response,
            se=se
        )
        
    @property
    def model(self):
        """Access to the underlying sklearn estimator."""
        return self._fitted_estimator
        
    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        """Feature importances if available."""
        if self.is_trained and hasattr(self._fitted_estimator, 'feature_importances_'):
            return self._fitted_estimator.feature_importances_
        return None
        
    def clone(self, deep: bool = True) -> "LearnerSklearn":
        """Create a copy of the learner.
        
        Parameters
        ----------
        deep : bool, default=True
            Whether to make a deep copy.
            
        Returns
        -------
        LearnerSklearn
            A copy of the learner.
        """
        if deep:
            new_estimator = deepcopy(self.estimator)
        else:
            new_estimator = self.estimator
            
        cloned = LearnerSklearn(
            estimator=new_estimator,
            id=self.id,
            predict_type=self.predict_type
        )
        
        # Don't copy fitted state
        cloned._fitted_estimator = None
        cloned._model = None
        cloned._train_task = None
        
        return cloned
        
    @property
    def _properties(self) -> set:
        """Properties for this learner type."""
        return self.properties


@mlpy_learners.register('sklearn.classif', aliases=['sklearn.classifier'])
class LearnerClassifSklearn(LearnerSklearn):
    """Scikit-learn classifier wrapper.
    
    This class specifically wraps scikit-learn classifiers.
    
    Parameters
    ----------
    estimator : sklearn classifier
        A scikit-learn classifier instance.
    id : str, optional
        Unique identifier.
    predict_type : str, optional
        Type of prediction ('response' or 'prob').
    **kwargs
        Additional parameters.
        
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from mlpy.learners import LearnerClassifSklearn
    >>> 
    >>> # Create learner with sklearn classifier
    >>> rf = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> learner = LearnerClassifSklearn(rf)
    >>> 
    >>> # Train and predict
    >>> learner.train(task)
    >>> predictions = learner.predict(task)
    """
    
    def __init__(
        self,
        estimator: BaseEstimator,
        id: Optional[str] = None,
        predict_type: str = "response",
        **kwargs
    ):
        # Validate it's a classifier
        if not isinstance(estimator, ClassifierMixin):
            # Check if it has classifier methods
            if not hasattr(estimator, 'predict_proba'):
                raise TypeError(
                    f"estimator must be a scikit-learn classifier, "
                    f"got {type(estimator)}"
                )
                
        super().__init__(
            estimator=estimator,
            id=id,
            predict_type=predict_type,
            **kwargs
        )
        
    @property
    def task_type(self) -> str:
        """This is always a classifier."""
        return 'classif'


@mlpy_learners.register('sklearn.regr', aliases=['sklearn.regressor'])  
class LearnerRegrSklearn(LearnerSklearn):
    """Scikit-learn regressor wrapper.
    
    This class specifically wraps scikit-learn regressors.
    
    Parameters
    ----------
    estimator : sklearn regressor
        A scikit-learn regressor instance.
    id : str, optional
        Unique identifier.
    **kwargs
        Additional parameters.
        
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from mlpy.learners import LearnerRegrSklearn
    >>> 
    >>> # Create learner with sklearn regressor
    >>> rf = RandomForestRegressor(n_estimators=100, random_state=42)
    >>> learner = LearnerRegrSklearn(rf)
    >>> 
    >>> # Train and predict
    >>> learner.train(task)
    >>> predictions = learner.predict(task)
    """
    
    def __init__(
        self,
        estimator: BaseEstimator,
        id: Optional[str] = None,
        **kwargs
    ):
        # Validate it's a regressor
        if not isinstance(estimator, RegressorMixin):
            # More flexible check
            if hasattr(estimator, 'predict_proba') or hasattr(estimator, 'classes_'):
                raise TypeError(
                    f"estimator appears to be a classifier, not a regressor"
                )
                
        # Regressors only support response predictions
        # Remove predict_type from kwargs if present
        kwargs.pop('predict_type', None)
        
        super().__init__(
            estimator=estimator,
            id=id,
            predict_type="response",
            **kwargs
        )
        
    @property  
    def task_type(self) -> str:
        """This is always a regressor."""
        return 'regr'


# Convenience function for auto-detection
def learner_sklearn(estimator: BaseEstimator, **kwargs) -> LearnerSklearn:
    """Create an appropriate sklearn learner based on the estimator type.
    
    This function automatically detects whether the estimator is a
    classifier or regressor and returns the appropriate wrapper.
    
    Parameters
    ----------
    estimator : sklearn estimator
        A scikit-learn estimator.
    **kwargs
        Additional parameters passed to the learner.
        
    Returns
    -------
    LearnerSklearn
        Either LearnerClassifSklearn or LearnerRegrSklearn.
        
    Examples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from mlpy.learners import learner_sklearn
    >>> 
    >>> # Automatically creates LearnerClassifSklearn
    >>> learner = learner_sklearn(DecisionTreeClassifier())
    """
    if isinstance(estimator, ClassifierMixin) or hasattr(estimator, 'predict_proba'):
        return LearnerClassifSklearn(estimator, **kwargs)
    else:
        return LearnerRegrSklearn(estimator, **kwargs)


__all__ = [
    'LearnerSklearn',
    'LearnerClassifSklearn', 
    'LearnerRegrSklearn',
    'learner_sklearn'
]