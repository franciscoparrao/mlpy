"""LightGBM integration for MLPY.

This module provides learners that wrap LightGBM models,
allowing them to be used seamlessly within the MLPY framework.

LightGBM advantages:
- Faster training speed and higher efficiency
- Lower memory usage
- Better accuracy with large-scale data
- Support for parallel and GPU learning
- Capable of handling large-scale data
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union, Tuple
import warnings
import logging

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False
    lgb = None

from .base import Learner
from .classification import LearnerClassif
from .regression import LearnerRegr
from ..tasks import Task, TaskClassif, TaskRegr
from ..predictions import PredictionClassif, PredictionRegr
from ..utils.registry import mlpy_learners


class LearnerLightGBM(Learner):
    """Base learner for LightGBM integration.
    
    This class wraps LightGBM to work with MLPY's Task/Learner/Prediction interface.
    LightGBM uses a leaf-wise tree growth algorithm which can be faster and more
    memory efficient than XGBoost's level-wise approach.
    
    Parameters
    ----------
    objective : str, optional
        LightGBM objective function. If None, will be inferred from task type.
    n_estimators : int, default=100
        Number of boosting iterations.
    max_depth : int, default=-1
        Maximum tree depth. -1 means no limit.
    learning_rate : float, default=0.1
        Boosting learning rate.
    num_leaves : int, default=31
        Maximum tree leaves for base learners.
    min_child_samples : int, default=20
        Minimum number of data points in a leaf.
    subsample : float, default=1.0
        Subsample ratio of training instances.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    reg_alpha : float, default=0.0
        L1 regularization term.
    reg_lambda : float, default=0.0
        L2 regularization term.
    categorical_features : Union[List[int], List[str], str], optional
        Categorical features. 'auto' to use pandas categorical dtype.
    device_type : str, default='cpu'
        Device for training ('cpu', 'gpu', 'cuda').
    gpu_platform_id : int, default=0
        GPU platform ID.
    gpu_device_id : int, default=0
        GPU device ID.
    early_stopping_rounds : int, optional
        Activates early stopping.
    eval_metric : str or list, optional
        Evaluation metrics for validation data.
    importance_type : str, default='gain'
        Type of feature importance ('gain' or 'split').
    id : str, optional
        Unique identifier.
    **kwargs
        Additional LightGBM parameters.
    """
    
    def __init__(
        self,
        objective: Optional[str] = None,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        categorical_features: Optional[Union[List[int], List[str], str]] = None,
        device_type: str = 'cpu',
        gpu_platform_id: int = 0,
        gpu_device_id: int = 0,
        early_stopping_rounds: Optional[int] = None,
        eval_metric: Optional[Union[str, List[str]]] = None,
        importance_type: str = 'gain',
        id: Optional[str] = None,
        **kwargs
    ):
        if not _LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Install it with: pip install lightgbm"
            )
            
        # Store parameters
        self.objective = objective
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.categorical_features = categorical_features
        self.device_type = device_type
        self.gpu_platform_id = gpu_platform_id
        self.gpu_device_id = gpu_device_id
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.importance_type = importance_type
        self.lgb_params = kwargs
        
        # Build LightGBM parameters
        self._params = {
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'verbosity': -1,
            'num_threads': 0,  # Use all available threads
            **kwargs
        }
        
        # Configure GPU if requested
        if device_type in ['gpu', 'cuda']:
            self._params.update({
                'device_type': 'gpu',
                'gpu_platform_id': gpu_platform_id,
                'gpu_device_id': gpu_device_id,
            })
            logger.info(f"LightGBM configured for GPU training on device {gpu_device_id}")
        
        # Model will be set during training
        self._booster = None
        self._feature_names = None
        self._categorical_feature_indices = None
        
        # Auto-generate ID if not provided
        if id is None:
            id = "lightgbm"
            
        # LightGBM properties
        properties = {"importance", "prob", "shap", "categorical"}
        
        super().__init__(
            id=id,
            properties=properties,
            packages={"lightgbm"},
            **kwargs
        )
        
    def _process_categorical_features(
        self, 
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: List[str]
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], List[int]]:
        """Process and identify categorical features.
        
        Parameters
        ----------
        X : array-like
            Input features.
        feature_names : list
            Feature names.
            
        Returns
        -------
        X : array-like
            Processed features.
        categorical_indices : list
            Indices of categorical features.
        """
        categorical_indices = []
        
        if isinstance(X, pd.DataFrame):
            # Auto-detect categorical columns from pandas
            if self.categorical_features == 'auto':
                categorical_indices = [
                    i for i, col in enumerate(X.columns)
                    if X[col].dtype.name in ['category', 'object']
                ]
            elif isinstance(self.categorical_features, list):
                # Convert feature names to indices if needed
                if self.categorical_features and isinstance(self.categorical_features[0], str):
                    categorical_indices = [
                        i for i, col in enumerate(X.columns)
                        if col in self.categorical_features
                    ]
                else:
                    categorical_indices = self.categorical_features
        else:
            # For numpy arrays, use provided indices
            if isinstance(self.categorical_features, list) and self.categorical_features:
                if isinstance(self.categorical_features[0], int):
                    categorical_indices = self.categorical_features
                    
        return X, categorical_indices
        
    def train(self, task: Task, row_ids: Optional[List[int]] = None) -> "LearnerLightGBM":
        """Train the LightGBM model.
        
        Parameters
        ----------
        task : Task
            The task to train on.
        row_ids : list of int, optional
            Subset of rows to use for training.
            
        Returns
        -------
        self : LearnerLightGBM
            The trained learner.
        """
        # Determine objective if not set
        if self.objective is None:
            if isinstance(task, TaskClassif):
                if task.n_classes == 2:
                    self.objective = 'binary'
                else:
                    self.objective = 'multiclass'
                    self._params['num_class'] = task.n_classes
            else:
                self.objective = 'regression'
                
        self._params['objective'] = self.objective
        
        # Set evaluation metric if not specified
        if self.eval_metric is None:
            if isinstance(task, TaskClassif):
                if task.n_classes == 2:
                    self.eval_metric = 'binary_logloss'
                else:
                    self.eval_metric = 'multi_logloss'
            else:
                self.eval_metric = 'rmse'
                
        if isinstance(self.eval_metric, str):
            self._params['metric'] = self.eval_metric
        else:
            self._params['metric'] = self.eval_metric
            
        # Get training data
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        # Get data - prefer DataFrame for categorical support
        X = task.data(rows=row_ids, cols=task.feature_names, data_format='dataframe')
        y = task.truth(rows=row_ids)
        
        # Process categorical features
        X, categorical_indices = self._process_categorical_features(X, task.feature_names)
        self._categorical_feature_indices = categorical_indices
        
        # For classification, encode labels to 0, 1, 2, ...
        if isinstance(task, TaskClassif):
            from sklearn.preprocessing import LabelEncoder
            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y)
            
        # Create Dataset
        train_data = lgb.Dataset(
            X, 
            label=y,
            feature_name=task.feature_names,
            categorical_feature=categorical_indices if categorical_indices else 'auto'
        )
        
        self._feature_names = task.feature_names
        
        # Setup callbacks
        callbacks = []
        if self.early_stopping_rounds:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
        callbacks.append(lgb.log_evaluation(period=0))  # Suppress output
        
        # Train model
        self._booster = lgb.train(
            params=self._params,
            train_set=train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[train_data],
            valid_names=['train'],
            callbacks=callbacks
        )
        
        self._train_task = task
        
        logger.info(f"LightGBM trained with {self._booster.num_trees()} trees")
        
        return self
        
    def predict(self, task: Task, row_ids: Optional[List[int]] = None) -> Union[PredictionClassif, PredictionRegr]:
        """Make predictions using the trained LightGBM model.
        
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
            
        X = task.data(rows=row_ids, cols=self._feature_names, data_format='dataframe')
        
        # Process categorical features
        X, _ = self._process_categorical_features(X, self._feature_names)
        
        # Get truth values
        truth = task.truth(rows=row_ids)
        
        if isinstance(self._train_task, TaskClassif):
            return self._predict_classif(X, truth, task, row_ids)
        else:
            return self._predict_regr(X, truth, task, row_ids)
            
    def _predict_classif(
        self, 
        X: Union[np.ndarray, pd.DataFrame],
        truth: np.ndarray,
        task: TaskClassif,
        row_ids: List[int]
    ) -> PredictionClassif:
        """Make classification predictions."""
        # Get probabilities
        prob = self._booster.predict(X, num_iteration=self._booster.best_iteration)
        
        # Handle binary vs multiclass
        if len(prob.shape) == 1:
            # Binary classification - LightGBM returns prob of positive class
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
        X: Union[np.ndarray, pd.DataFrame],
        truth: np.ndarray,
        task: TaskRegr,
        row_ids: List[int]
    ) -> PredictionRegr:
        """Make regression predictions."""
        response = self._booster.predict(X, num_iteration=self._booster.best_iteration)
        
        return PredictionRegr(
            task=task,
            learner_id=self.id,
            row_ids=row_ids,
            truth=truth,
            response=response,
            se=None  # LightGBM doesn't provide standard errors directly
        )
        
    @property
    def model(self):
        """Access to the underlying LightGBM booster."""
        return self._booster
        
    @property
    def feature_importances(self) -> Optional[Dict[str, float]]:
        """Get feature importances."""
        if self.is_trained:
            importance = self._booster.feature_importance(importance_type=self.importance_type)
            return dict(zip(self._feature_names, importance))
        return None
        
    def plot_importance(self, importance_type='gain', max_num_features=20, figsize=(10, 6)):
        """Plot feature importance.
        
        Parameters
        ----------
        importance_type : str
            Type of importance: 'gain' or 'split'
        max_num_features : int
            Maximum number of features to display
        figsize : tuple
            Figure size
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
            
        import matplotlib.pyplot as plt
        
        # Get importance
        importance = self._booster.feature_importance(importance_type=importance_type)
        feature_names = self._feature_names
        
        # Sort and select top features
        indices = np.argsort(importance)[::-1][:max_num_features]
        
        # Plot
        plt.figure(figsize=figsize)
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel(f'Feature Importance ({importance_type})')
        plt.title('LightGBM Feature Importance')
        plt.tight_layout()
        plt.show()
        
    def plot_tree(self, tree_index=0, show_info=None):
        """Plot a tree from the model.
        
        Parameters
        ----------
        tree_index : int
            Index of tree to plot
        show_info : list
            What information to show in nodes
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
            
        import matplotlib.pyplot as plt
        
        # Use LightGBM's plot_tree function
        lgb.plot_tree(
            self._booster,
            tree_index=tree_index,
            show_info=show_info or ['split_gain', 'leaf_count', 'internal_value']
        )
        plt.show()
        
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
            
        X = task.data(rows=row_ids, cols=self._feature_names, data_format='dataframe')
        X, _ = self._process_categorical_features(X, self._feature_names)
        
        # Get SHAP values from LightGBM
        shap_values = self._booster.predict(X, pred_contrib=True)
        
        # Remove bias term (last column) if present
        if shap_values.shape[1] == len(self._feature_names) + 1:
            shap_values = shap_values[:, :-1]
            
        return shap_values
        
    def clone(self, deep: bool = True) -> "LearnerLightGBM":
        """Create a copy of the learner."""
        new_learner = self.__class__(
            objective=self.objective,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            categorical_features=self.categorical_features,
            device_type=self.device_type,
            gpu_platform_id=self.gpu_platform_id,
            gpu_device_id=self.gpu_device_id,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric=self.eval_metric,
            importance_type=self.importance_type,
            id=self.id,
            **self.lgb_params
        )
        
        if deep and self.is_trained:
            # LightGBM models can be saved/loaded
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
                self._booster.save_model(tmp.name)
                new_learner._booster = lgb.Booster(model_file=tmp.name)
                os.unlink(tmp.name)
                
            new_learner._feature_names = self._feature_names.copy() if self._feature_names else None
            new_learner._categorical_feature_indices = self._categorical_feature_indices
            new_learner._train_task = self._train_task
            if hasattr(self, '_label_encoder'):
                from sklearn.preprocessing import LabelEncoder
                new_learner._label_encoder = LabelEncoder()
                new_learner._label_encoder.classes_ = self._label_encoder.classes_
                
        return new_learner
        
    def reset(self) -> "LearnerLightGBM":
        """Reset the learner to untrained state."""
        self._booster = None
        self._feature_names = None
        self._categorical_feature_indices = None
        self._train_task = None
        if hasattr(self, '_label_encoder'):
            delattr(self, '_label_encoder')
        return self


class LearnerLightGBMClassif(LearnerLightGBM, LearnerClassif):
    """LightGBM classifier wrapper.
    
    Optimized for classification tasks with automatic handling of:
    - Binary and multiclass problems
    - Class imbalance via is_unbalance parameter
    - Categorical features
    """
    
    def __init__(self, is_unbalance: bool = False, **kwargs):
        # Set default objective for classification
        if 'objective' not in kwargs:
            kwargs['objective'] = 'binary'
            
        # Handle class imbalance
        if is_unbalance:
            kwargs['is_unbalance'] = True
            
        super().__init__(**kwargs)


class LearnerLightGBMRegr(LearnerLightGBM, LearnerRegr):
    """LightGBM regressor wrapper.
    
    Optimized for regression tasks with support for various objectives:
    - regression (L2 loss)
    - regression_l1 (L1 loss)
    - huber (Huber loss)
    - quantile (Quantile regression)
    - mape (MAPE loss)
    - tweedie (Tweedie regression)
    """
    
    def __init__(self, **kwargs):
        # Set default objective for regression
        if 'objective' not in kwargs:
            kwargs['objective'] = 'regression'
        super().__init__(predict_type="response", **kwargs)


# Register learners
if _LIGHTGBM_AVAILABLE:
    mlpy_learners.register("lightgbm", LearnerLightGBM)
    mlpy_learners.register("lightgbm.classif", LearnerLightGBMClassif)
    mlpy_learners.register("lightgbm.regr", LearnerLightGBMRegr)
    mlpy_learners.register("lgb", LearnerLightGBM)  # Alias
    mlpy_learners.register("lgb.classif", LearnerLightGBMClassif)  # Alias
    mlpy_learners.register("lgb.regr", LearnerLightGBMRegr)  # Alias


def learner_lightgbm(**kwargs) -> Union[LearnerLightGBMClassif, LearnerLightGBMRegr]:
    """Create a LightGBM learner with automatic type detection.
    
    Parameters
    ----------
    **kwargs
        Parameters passed to LightGBM
        
    Returns
    -------
    LearnerLightGBM
        Either LearnerLightGBMClassif or LearnerLightGBMRegr based on objective.
        
    Examples
    --------
    >>> from mlpy.learners import learner_lightgbm
    >>> 
    >>> # For classification
    >>> lgb_clf = learner_lightgbm(objective='binary', n_estimators=100)
    >>> 
    >>> # For regression  
    >>> lgb_reg = learner_lightgbm(objective='regression', n_estimators=100)
    >>> 
    >>> # With GPU support
    >>> lgb_gpu = learner_lightgbm(device_type='gpu', n_estimators=100)
    >>> 
    >>> # With categorical features
    >>> lgb_cat = learner_lightgbm(categorical_features='auto')
    """
    objective = kwargs.get('objective', None)
    
    if objective:
        if any(x in objective for x in ['binary', 'multiclass', 'classification']):
            return LearnerLightGBMClassif(**kwargs)
        elif any(x in objective for x in ['regression', 'reg:', 'l1', 'l2', 'huber', 'quantile', 'mape', 'tweedie']):
            return LearnerLightGBMRegr(**kwargs)
        else:
            # Default to classifier if objective not recognized
            return LearnerLightGBMClassif(**kwargs)
    else:
        # Default to classifier when no objective specified
        # User can specify objective='regression' if they want regression
        return LearnerLightGBMClassif(**kwargs)