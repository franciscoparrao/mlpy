"""XGBoost integration for MLPY.

This module provides learners that wrap XGBoost models,
allowing them to be used seamlessly within the MLPY framework.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union
import warnings

try:
    import xgboost as xgb
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    xgb = None

from .base import Learner
from .classification import LearnerClassif
from .regression import LearnerRegr
from ..tasks import Task, TaskClassif, TaskRegr
from ..predictions import PredictionClassif, PredictionRegr
from ..utils.registry import mlpy_learners


class LearnerXGBoost(Learner):
    """Base learner for XGBoost integration.
    
    This class wraps XGBoost to work with MLPY's Task/Learner/Prediction interface.
    
    Parameters
    ----------
    objective : str, optional
        XGBoost objective function. If None, will be inferred from task type.
    n_estimators : int, default=100
        Number of boosting rounds.
    max_depth : int, default=6
        Maximum tree depth.
    learning_rate : float, default=0.3
        Step size shrinkage.
    subsample : float, default=1.0
        Subsample ratio of training instances.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    eval_metric : str or list, optional
        Evaluation metrics for validation data.
    early_stopping_rounds : int, optional
        Activates early stopping.
    id : str, optional
        Unique identifier.
    **kwargs
        Additional XGBoost parameters.
    """
    
    def __init__(
        self,
        objective: Optional[str] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        eval_metric: Optional[Union[str, List[str]]] = None,
        early_stopping_rounds: Optional[int] = None,
        id: Optional[str] = None,
        **kwargs
    ):
        if not _XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            )
            
        # Store parameters
        self.objective = objective
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.xgb_params = kwargs
        
        # Build XGBoost parameters
        self._params = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            **kwargs
        }
        
        # Model will be set during training
        self._booster = None
        self._feature_names = None
        
        # Auto-generate ID if not provided
        if id is None:
            id = "xgboost"
            
        # XGBoost properties
        properties = {"importance", "prob", "shap"}
        
        super().__init__(
            id=id,
            properties=properties,
            packages={"xgboost"},
            **kwargs
        )
        
    @property
    def task_type(self) -> str:
        """Infer task type from objective."""
        if self.objective:
            if any(x in self.objective for x in ['binary', 'multi', 'softmax']):
                return 'classif'
            elif any(x in self.objective for x in ['reg:', 'squared', 'gamma', 'tweedie']):
                return 'regr'
                
        # Will be determined during training
        return None
        
    def train(self, task: Task, row_ids: Optional[List[int]] = None) -> "LearnerXGBoost":
        """Train the XGBoost model.
        
        Parameters
        ----------
        task : Task
            The task to train on.
        row_ids : list of int, optional
            Subset of rows to use for training.
            
        Returns
        -------
        self : LearnerXGBoost
            The trained learner.
        """
        # Determine objective if not set
        if self.objective is None:
            if isinstance(task, TaskClassif):
                if task.n_classes == 2:
                    self.objective = 'binary:logistic'
                else:
                    self.objective = 'multi:softprob'
                    self._params['num_class'] = task.n_classes
            else:
                self.objective = 'reg:squarederror'
                
        self._params['objective'] = self.objective
        
        # Set evaluation metric if not specified
        if self.eval_metric is None:
            if isinstance(task, TaskClassif):
                self.eval_metric = 'logloss' if task.n_classes == 2 else 'mlogloss'
            else:
                self.eval_metric = 'rmse'
                
        # Get training data
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        X = task.data(rows=row_ids, cols=task.feature_names, data_format='array')
        y = task.truth(rows=row_ids)
        
        # For classification, encode labels to 0, 1, 2, ...
        if isinstance(task, TaskClassif):
            from sklearn.preprocessing import LabelEncoder
            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y)
            
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y, feature_names=task.feature_names)
        self._feature_names = task.feature_names
        
        # Setup evaluation
        evals = [(dtrain, 'train')]
        
        # Train model
        self._booster = xgb.train(
            params=self._params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False
        )
        
        self._train_task = task
        
        return self
        
    def predict(self, task: Task, row_ids: Optional[List[int]] = None) -> Union[PredictionClassif, PredictionRegr]:
        """Make predictions using the trained XGBoost model.
        
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
            
        X = task.data(rows=row_ids, cols=self._feature_names, data_format='array')
        dtest = xgb.DMatrix(X, feature_names=self._feature_names)
        
        # Get truth values
        truth = task.truth(rows=row_ids)
        
        if isinstance(self._train_task, TaskClassif):
            return self._predict_classif(dtest, truth, task, row_ids)
        else:
            return self._predict_regr(dtest, truth, task, row_ids)
            
    def _predict_classif(
        self, 
        dtest: 'xgb.DMatrix',
        truth: np.ndarray,
        task: TaskClassif,
        row_ids: List[int]
    ) -> PredictionClassif:
        """Make classification predictions."""
        # Get probabilities
        prob = self._booster.predict(dtest)
        
        # Handle binary vs multiclass
        if len(prob.shape) == 1:
            # Binary classification - XGBoost returns prob of positive class
            prob = np.column_stack([1 - prob, prob])
            
        # Get predicted classes
        response_idx = np.argmax(prob, axis=1)
        response = self._label_encoder.inverse_transform(response_idx)
        
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
        dtest: 'xgb.DMatrix',
        truth: np.ndarray,
        task: TaskRegr,
        row_ids: List[int]
    ) -> PredictionRegr:
        """Make regression predictions."""
        response = self._booster.predict(dtest)
        
        return PredictionRegr(
            task=task,
            learner_id=self.id,
            row_ids=row_ids,
            truth=truth,
            response=response,
            se=None  # XGBoost doesn't provide standard errors
        )
        
    @property
    def model(self):
        """Access to the underlying XGBoost booster."""
        return self._booster
        
    @property
    def feature_importances(self) -> Optional[Dict[str, float]]:
        """Get feature importances."""
        if self.is_trained:
            return self._booster.get_score(importance_type='gain')
        return None
        
    def plot_importance(self, importance_type='gain', max_num_features=20):
        """Plot feature importance.
        
        Parameters
        ----------
        importance_type : str
            Type of importance: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'
        max_num_features : int
            Maximum number of features to display
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
            
        import matplotlib.pyplot as plt
        xgb.plot_importance(
            self._booster, 
            importance_type=importance_type,
            max_num_features=max_num_features
        )
        plt.show()
        
    def plot_tree(self, num_trees=0):
        """Plot a tree from the model.
        
        Parameters
        ----------
        num_trees : int
            Which tree to plot (0-indexed)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
            
        xgb.plot_tree(self._booster, num_trees=num_trees)
        
    def get_shap_values(self, task: Task, row_ids: Optional[List[int]] = None):
        """Get SHAP values for predictions.
        
        Parameters
        ----------
        task : Task
            The task to explain.
        row_ids : list of int, optional
            Subset of rows to explain.
            
        Returns
        -------
        shap_values : np.ndarray
            SHAP values for each feature and sample
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
            
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        X = task.data(rows=row_ids, cols=self._feature_names, data_format='array')
        dtest = xgb.DMatrix(X, feature_names=self._feature_names)
        
        # Get SHAP values from XGBoost
        shap_values = self._booster.predict(dtest, pred_contribs=True)
        
        # Remove bias term (last column)
        return shap_values[:, :-1]
        
    def clone(self, deep: bool = True) -> "LearnerXGBoost":
        """Create a copy of the learner."""
        new_learner = self.__class__(
            objective=self.objective,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds,
            id=self.id,
            **self.xgb_params
        )
        
        if deep and self.is_trained:
            # XGBoost models can be saved/loaded
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
                self._booster.save_model(tmp.name)
                new_learner._booster = xgb.Booster()
                new_learner._booster.load_model(tmp.name)
                os.unlink(tmp.name)
                
            new_learner._feature_names = self._feature_names.copy() if self._feature_names else None
            new_learner._train_task = self._train_task
            if hasattr(self, '_label_encoder'):
                from sklearn.preprocessing import LabelEncoder
                new_learner._label_encoder = LabelEncoder()
                new_learner._label_encoder.classes_ = self._label_encoder.classes_
                
        return new_learner
        
    def reset(self) -> "LearnerXGBoost":
        """Reset the learner to untrained state."""
        self._booster = None
        self._feature_names = None
        self._train_task = None
        if hasattr(self, '_label_encoder'):
            delattr(self, '_label_encoder')
        return self


class LearnerXGBoostClassif(LearnerXGBoost, LearnerClassif):
    """XGBoost classifier wrapper."""
    
    def __init__(self, **kwargs):
        # Set default objective for classification
        if 'objective' not in kwargs:
            kwargs['objective'] = 'binary:logistic'
        super().__init__(**kwargs)


class LearnerXGBoostRegr(LearnerXGBoost, LearnerRegr):
    """XGBoost regressor wrapper."""
    
    def __init__(self, **kwargs):
        # Set default objective for regression
        if 'objective' not in kwargs:
            kwargs['objective'] = 'reg:squarederror'
        super().__init__(predict_type="response", **kwargs)


# Register learners
if _XGBOOST_AVAILABLE:
    mlpy_learners.register("xgboost", LearnerXGBoost)
    mlpy_learners.register("xgboost.classif", LearnerXGBoostClassif)
    mlpy_learners.register("xgboost.regr", LearnerXGBoostRegr)


def learner_xgboost(**kwargs) -> Union[LearnerXGBoostClassif, LearnerXGBoostRegr]:
    """Create an XGBoost learner with automatic type detection.
    
    Parameters
    ----------
    **kwargs
        Parameters passed to XGBoost
        
    Returns
    -------
    LearnerXGBoost
        Either LearnerXGBoostClassif or LearnerXGBoostRegr based on objective.
        
    Examples
    --------
    >>> from mlpy.learners import learner_xgboost
    >>> 
    >>> # For classification
    >>> xgb_clf = learner_xgboost(objective='binary:logistic', n_estimators=100)
    >>> 
    >>> # For regression  
    >>> xgb_reg = learner_xgboost(objective='reg:squarederror', n_estimators=100)
    >>> 
    >>> # Auto-detect from task
    >>> xgb_auto = learner_xgboost(n_estimators=100)
    """
    objective = kwargs.get('objective', None)
    
    if objective:
        if any(x in objective for x in ['binary', 'multi', 'softmax', 'logistic']):
            return LearnerXGBoostClassif(**kwargs)
        elif any(x in objective for x in ['reg:', 'squarederror', 'tweedie', 'gamma', 'poisson']):
            return LearnerXGBoostRegr(**kwargs)
        else:
            # Default to classifier if objective not recognized
            return LearnerXGBoostClassif(**kwargs)
    else:
        # Default to classifier when no objective specified
        # User can specify objective='reg:squarederror' if they want regression
        return LearnerXGBoostClassif(**kwargs)