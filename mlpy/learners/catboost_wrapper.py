"""CatBoost integration for MLPY.

This module provides learners that wrap CatBoost models,
allowing them to be used seamlessly within the MLPY framework.

CatBoost advantages:
- Superior handling of categorical features (no preprocessing needed)
- Ordered boosting to reduce overfitting
- Fast GPU training
- Built-in cross-validation
- Automatic handling of missing values
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union, Tuple
import warnings
import logging

logger = logging.getLogger(__name__)

try:
    import catboost as cb
    from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, Pool
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False
    cb = None
    CatBoost = None
    CatBoostClassifier = None
    CatBoostRegressor = None
    Pool = None

from .base import Learner
from .classification import LearnerClassif
from .regression import LearnerRegr
from ..tasks import Task, TaskClassif, TaskRegr
from ..predictions import PredictionClassif, PredictionRegr
from ..utils.registry import mlpy_learners


class LearnerCatBoost(Learner):
    """Base learner for CatBoost integration.
    
    This class wraps CatBoost to work with MLPY's Task/Learner/Prediction interface.
    CatBoost excels at handling categorical features without preprocessing and
    uses ordered boosting to reduce overfitting.
    
    Parameters
    ----------
    objective : str, optional
        CatBoost objective function. If None, will be inferred from task type.
    n_estimators : int, default=100
        Number of boosting iterations.
    max_depth : int, default=6
        Maximum tree depth.
    learning_rate : float, default=0.03
        Step size shrinkage. CatBoost default is lower than XGBoost/LightGBM.
    l2_leaf_reg : float, default=3.0
        L2 regularization coefficient.
    subsample : float, optional
        Sample rate for bagging. None means no bagging.
    colsample_bylevel : float, default=1.0
        Subsample ratio of columns for each split.
    random_strength : float, default=1.0
        Random strength for scoring splits.
    border_count : int, default=254
        Number of splits for numerical features.
    cat_features : Union[List[int], List[str], str], optional
        Categorical features. 'auto' to detect automatically.
    text_features : Union[List[int], List[str]], optional
        Text features for built-in text processing.
    embedding_features : Union[List[int], List[str]], optional
        Embedding features.
    task_type : str, default='CPU'
        Training device ('CPU' or 'GPU').
    devices : str, optional
        GPU devices to use (e.g., '0:1' for GPUs 0 and 1).
    early_stopping_rounds : int, optional
        Activates early stopping.
    eval_metric : str, optional
        Evaluation metric for validation data.
    auto_class_weights : Union[str, Dict], optional
        Auto class weights. 'Balanced' or 'SqrtBalanced' or custom dict.
    langevin : bool, default=False
        Enable Langevin boosting for better uncertainty estimates.
    posterior_sampling : bool, default=False
        Enable posterior sampling for uncertainty.
    boosting_type : str, default='Plain'
        Boosting scheme ('Plain', 'Ordered').
    feature_border_type : str, default='GreedyLogSum'
        Feature discretization ('Median', 'Uniform', 'UniformAndQuantiles', etc).
    id : str, optional
        Unique identifier.
    verbose : bool, default=False
        Verbosity mode.
    **kwargs
        Additional CatBoost parameters.
    """
    
    def __init__(
        self,
        objective: Optional[str] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 3.0,
        subsample: Optional[float] = None,
        colsample_bylevel: float = 1.0,
        random_strength: float = 1.0,
        border_count: int = 254,
        cat_features: Optional[Union[List[int], List[str], str]] = None,
        text_features: Optional[Union[List[int], List[str]]] = None,
        embedding_features: Optional[Union[List[int], List[str]]] = None,
        task_type: str = 'CPU',
        devices: Optional[str] = None,
        early_stopping_rounds: Optional[int] = None,
        eval_metric: Optional[str] = None,
        auto_class_weights: Optional[Union[str, Dict]] = None,
        langevin: bool = False,
        posterior_sampling: bool = False,
        boosting_type: str = 'Plain',
        feature_border_type: str = 'GreedyLogSum',
        id: Optional[str] = None,
        verbose: bool = False,
        **kwargs
    ):
        if not _CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoost is not installed. Install it with: pip install catboost"
            )
            
        # Store parameters
        self.objective = objective
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.colsample_bylevel = colsample_bylevel
        self.random_strength = random_strength
        self.border_count = border_count
        self.cat_features = cat_features
        self.text_features = text_features
        self.embedding_features = embedding_features
        self.device_type = task_type  # In CatBoost, task_type refers to CPU/GPU
        self.devices = devices
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.auto_class_weights = auto_class_weights
        self.langevin = langevin
        self.posterior_sampling = posterior_sampling
        self.boosting_type = boosting_type
        self.feature_border_type = feature_border_type
        self.verbose = verbose
        self.catboost_params = kwargs
        
        # Build CatBoost parameters
        self._params = {
            'iterations': n_estimators,
            'depth': max_depth,
            'learning_rate': learning_rate,
            'l2_leaf_reg': l2_leaf_reg,
            'colsample_bylevel': colsample_bylevel,
            'random_strength': random_strength,
            'border_count': border_count,
            'boosting_type': boosting_type,
            'feature_border_type': feature_border_type,
            'verbose': verbose,
            'allow_writing_files': False,
            'thread_count': -1,  # Use all available threads
            **kwargs
        }
        
        # Add optional parameters
        if subsample is not None:
            self._params['subsample'] = subsample
            self._params['bootstrap_type'] = 'Bernoulli'
            
        if langevin:
            self._params['langevin'] = True
            self._params['diffusion_temperature'] = kwargs.get('diffusion_temperature', 10000)
            
        if posterior_sampling:
            self._params['posterior_sampling'] = True
            
        # Configure GPU if requested
        if task_type == 'GPU':
            self._params['task_type'] = 'GPU'
            if devices:
                self._params['devices'] = devices
            logger.info(f"CatBoost configured for GPU training")
            
        # Model will be set during training
        self._model = None
        self._feature_names = None
        self._cat_feature_indices = None
        self._text_feature_indices = None
        
        # Auto-generate ID if not provided
        if id is None:
            id = "catboost"
            
        # CatBoost properties
        properties = {"importance", "prob", "shap", "categorical", "uncertainty"}
        
        super().__init__(
            id=id,
            properties=properties,
            packages={"catboost"},
            **kwargs
        )
        
    def _process_features(
        self, 
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: List[str]
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], List[int], List[int]]:
        """Process and identify categorical and text features.
        
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
        cat_indices : list
            Indices of categorical features.
        text_indices : list
            Indices of text features.
        """
        cat_indices = []
        text_indices = []
        
        if isinstance(X, pd.DataFrame):
            # Auto-detect categorical columns from pandas
            if self.cat_features == 'auto':
                cat_indices = [
                    i for i, col in enumerate(X.columns)
                    if X[col].dtype.name in ['category', 'object', 'bool']
                ]
            elif isinstance(self.cat_features, list):
                # Convert feature names to indices if needed
                if self.cat_features and isinstance(self.cat_features[0], str):
                    cat_indices = [
                        i for i, col in enumerate(X.columns)
                        if col in self.cat_features
                    ]
                else:
                    cat_indices = self.cat_features
                    
            # Process text features
            if self.text_features:
                if isinstance(self.text_features[0], str):
                    text_indices = [
                        i for i, col in enumerate(X.columns)
                        if col in self.text_features
                    ]
                else:
                    text_indices = self.text_features
        else:
            # For numpy arrays, use provided indices
            if isinstance(self.cat_features, list) and self.cat_features:
                if isinstance(self.cat_features[0], int):
                    cat_indices = self.cat_features
                    
            if isinstance(self.text_features, list) and self.text_features:
                if isinstance(self.text_features[0], int):
                    text_indices = self.text_features
                    
        return X, cat_indices, text_indices
        
    def train(self, task: Task, row_ids: Optional[List[int]] = None) -> "LearnerCatBoost":
        """Train the CatBoost model.
        
        Parameters
        ----------
        task : Task
            The task to train on.
        row_ids : list of int, optional
            Subset of rows to use for training.
            
        Returns
        -------
        self : LearnerCatBoost
            The trained learner.
        """
        # Determine objective if not set
        is_classification = isinstance(task, TaskClassif)
        
        if self.objective is None:
            if is_classification:
                if task.n_classes == 2:
                    self.objective = 'Logloss'
                else:
                    self.objective = 'MultiClass'
            else:
                self.objective = 'RMSE'
                
        # Set loss function
        if is_classification:
            self._params['loss_function'] = self.objective
            if task.n_classes > 2:
                self._params['classes_count'] = task.n_classes
                
            # Handle class weights
            if self.auto_class_weights:
                self._params['auto_class_weights'] = self.auto_class_weights
        else:
            self._params['loss_function'] = self.objective
            
        # Set evaluation metric if not specified
        if self.eval_metric:
            self._params['eval_metric'] = self.eval_metric
            
        # Get training data
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        # Get data - prefer DataFrame for categorical support
        X = task.data(rows=row_ids, cols=task.feature_names, data_format='dataframe')
        y = task.truth(rows=row_ids)
        
        # Process features
        X, cat_indices, text_indices = self._process_features(X, task.feature_names)
        self._cat_feature_indices = cat_indices
        self._text_feature_indices = text_indices
        
        # For classification, encode labels if needed
        if is_classification:
            from sklearn.preprocessing import LabelEncoder
            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y)
            
        # Create Pool (CatBoost's data structure)
        train_pool = Pool(
            data=X,
            label=y,
            cat_features=cat_indices if cat_indices else None,
            text_features=text_indices if text_indices else None,
            feature_names=task.feature_names
        )
        
        self._feature_names = task.feature_names
        
        # Create model
        if is_classification:
            self._model = CatBoostClassifier(**self._params)
        else:
            self._model = CatBoostRegressor(**self._params)
            
        # Train model
        self._model.fit(
            train_pool,
            eval_set=train_pool if self.early_stopping_rounds else None,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose
        )
        
        self._train_task = task
        
        logger.info(f"CatBoost trained with {self._model.tree_count_} trees")
        
        return self
        
    def predict(self, task: Task, row_ids: Optional[List[int]] = None) -> Union[PredictionClassif, PredictionRegr]:
        """Make predictions using the trained CatBoost model.
        
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
        
        # Process features
        X, cat_indices, text_indices = self._process_features(X, self._feature_names)
        
        # Create Pool for prediction
        test_pool = Pool(
            data=X,
            cat_features=cat_indices if cat_indices else None,
            text_features=text_indices if text_indices else None,
            feature_names=self._feature_names
        )
        
        # Get truth values
        truth = task.truth(rows=row_ids)
        
        if isinstance(self._train_task, TaskClassif):
            return self._predict_classif(test_pool, truth, task, row_ids)
        else:
            return self._predict_regr(test_pool, truth, task, row_ids)
            
    def _predict_classif(
        self, 
        test_pool: 'Pool',
        truth: np.ndarray,
        task: TaskClassif,
        row_ids: List[int]
    ) -> PredictionClassif:
        """Make classification predictions."""
        # Get probabilities
        prob = self._model.predict_proba(test_pool)
        
        # Handle binary vs multiclass
        if len(prob.shape) == 1:
            # Binary classification
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
        test_pool: 'Pool',
        truth: np.ndarray,
        task: TaskRegr,
        row_ids: List[int]
    ) -> PredictionRegr:
        """Make regression predictions."""
        response = self._model.predict(test_pool)
        
        # Get uncertainty estimates if available
        se = None
        if self.posterior_sampling or self.langevin:
            # CatBoost can provide uncertainty estimates with special settings
            # This would require virtual ensembles prediction
            try:
                # Get predictions with uncertainty
                preds = self._model.virtual_ensembles_predict(
                    test_pool,
                    prediction_type='TotalUncertainty'
                )
                if len(preds.shape) > 1:
                    se = preds[:, 1]  # Uncertainty in second column
            except:
                pass
        
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
        """Access to the underlying CatBoost model."""
        return self._model
        
    @property
    def feature_importances(self) -> Optional[Dict[str, float]]:
        """Get feature importances."""
        if self.is_trained:
            importance = self._model.get_feature_importance()
            return dict(zip(self._feature_names, importance))
        return None
        
    def plot_importance(self, importance_type='FeatureImportance', max_num_features=20):
        """Plot feature importance.
        
        Parameters
        ----------
        importance_type : str
            Type of importance: 'FeatureImportance', 'ShapValues', 'Interaction', 'PredictionDiff'
        max_num_features : int
            Maximum number of features to display
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
            
        import matplotlib.pyplot as plt
        
        # Get importance based on type
        if importance_type == 'FeatureImportance':
            importance = self._model.get_feature_importance()
        elif importance_type == 'ShapValues':
            # Would need reference data
            importance = self._model.get_feature_importance()
        else:
            importance = self._model.get_feature_importance()
            
        feature_names = self._feature_names
        
        # Sort and select top features
        indices = np.argsort(importance)[::-1][:max_num_features]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel(f'Feature Importance ({importance_type})')
        plt.title('CatBoost Feature Importance')
        plt.tight_layout()
        plt.show()
        
    def plot_tree(self, tree_index=0):
        """Plot a tree from the model.
        
        Parameters
        ----------
        tree_index : int
            Index of tree to plot
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
            
        # CatBoost tree plotting
        self._model.plot_tree(tree_idx=tree_index)
        
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
        X, cat_indices, text_indices = self._process_features(X, self._feature_names)
        
        # Create Pool
        test_pool = Pool(
            data=X,
            cat_features=cat_indices if cat_indices else None,
            text_features=text_indices if text_indices else None,
            feature_names=self._feature_names
        )
        
        # Get SHAP values from CatBoost
        shap_values = self._model.get_feature_importance(
            test_pool,
            type='ShapValues'
        )
        
        # Remove bias term if present
        if shap_values.shape[1] == len(self._feature_names) + 1:
            shap_values = shap_values[:, :-1]
            
        return shap_values
        
    def get_uncertainty(self, task: Task, row_ids: Optional[List[int]] = None) -> np.ndarray:
        """Get prediction uncertainty (if model supports it).
        
        Parameters
        ----------
        task : Task
            The task to get uncertainty for.
        row_ids : list of int, optional
            Subset of rows.
            
        Returns
        -------
        uncertainty : np.ndarray
            Uncertainty estimates for each prediction.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
            
        if not (self.posterior_sampling or self.langevin):
            raise RuntimeError("Model must be trained with posterior_sampling=True or langevin=True for uncertainty")
            
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        X = task.data(rows=row_ids, cols=self._feature_names, data_format='dataframe')
        X, cat_indices, text_indices = self._process_features(X, self._feature_names)
        
        # Create Pool
        test_pool = Pool(
            data=X,
            cat_features=cat_indices if cat_indices else None,
            text_features=text_indices if text_indices else None,
            feature_names=self._feature_names
        )
        
        # Get uncertainty using virtual ensembles
        uncertainty = self._model.virtual_ensembles_predict(
            test_pool,
            prediction_type='TotalUncertainty'
        )
        
        return uncertainty
        
    def clone(self, deep: bool = True) -> "LearnerCatBoost":
        """Create a copy of the learner."""
        new_learner = self.__class__(
            objective=self.objective,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            subsample=self.subsample,
            colsample_bylevel=self.colsample_bylevel,
            random_strength=self.random_strength,
            border_count=self.border_count,
            cat_features=self.cat_features,
            text_features=self.text_features,
            embedding_features=self.embedding_features,
            task_type=self.device_type,
            devices=self.devices,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric=self.eval_metric,
            auto_class_weights=self.auto_class_weights,
            langevin=self.langevin,
            posterior_sampling=self.posterior_sampling,
            boosting_type=self.boosting_type,
            feature_border_type=self.feature_border_type,
            id=self.id,
            verbose=self.verbose,
            **self.catboost_params
        )
        
        if deep and self.is_trained:
            # CatBoost models can be saved/loaded
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.cbm') as tmp:
                self._model.save_model(tmp.name)
                if isinstance(self._train_task, TaskClassif):
                    new_learner._model = CatBoostClassifier()
                else:
                    new_learner._model = CatBoostRegressor()
                new_learner._model.load_model(tmp.name)
                os.unlink(tmp.name)
                
            new_learner._feature_names = self._feature_names.copy() if self._feature_names else None
            new_learner._cat_feature_indices = self._cat_feature_indices
            new_learner._text_feature_indices = self._text_feature_indices
            new_learner._train_task = self._train_task
            if hasattr(self, '_label_encoder'):
                from sklearn.preprocessing import LabelEncoder
                new_learner._label_encoder = LabelEncoder()
                new_learner._label_encoder.classes_ = self._label_encoder.classes_
                
        return new_learner
        
    def reset(self) -> "LearnerCatBoost":
        """Reset the learner to untrained state."""
        self._model = None
        self._feature_names = None
        self._cat_feature_indices = None
        self._text_feature_indices = None
        self._train_task = None
        if hasattr(self, '_label_encoder'):
            delattr(self, '_label_encoder')
        return self


class LearnerCatBoostClassif(LearnerCatBoost, LearnerClassif):
    """CatBoost classifier wrapper.
    
    Optimized for classification with:
    - Automatic handling of imbalanced classes
    - Built-in cross-validation
    - Native categorical feature support
    - GPU acceleration
    """
    
    def __init__(self, **kwargs):
        # Set default objective for classification
        if 'objective' not in kwargs:
            kwargs['objective'] = 'Logloss'
        super().__init__(**kwargs)


class LearnerCatBoostRegr(LearnerCatBoost, LearnerRegr):
    """CatBoost regressor wrapper.
    
    Supports various regression objectives:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - Quantile (Quantile regression)
    - LogLinQuantile (Logarithmic quantile)
    - MAPE (Mean Absolute Percentage Error)
    - Poisson (Poisson regression)
    - Tweedie (Tweedie regression)
    """
    
    def __init__(self, **kwargs):
        # Set default objective for regression
        if 'objective' not in kwargs:
            kwargs['objective'] = 'RMSE'
        super().__init__(predict_type="response", **kwargs)


# Register learners
if _CATBOOST_AVAILABLE:
    mlpy_learners.register("catboost", LearnerCatBoost)
    mlpy_learners.register("catboost.classif", LearnerCatBoostClassif)
    mlpy_learners.register("catboost.regr", LearnerCatBoostRegr)
    mlpy_learners.register("cb", LearnerCatBoost)  # Alias
    mlpy_learners.register("cb.classif", LearnerCatBoostClassif)  # Alias
    mlpy_learners.register("cb.regr", LearnerCatBoostRegr)  # Alias


def learner_catboost(**kwargs) -> Union[LearnerCatBoostClassif, LearnerCatBoostRegr]:
    """Create a CatBoost learner with automatic type detection.
    
    Parameters
    ----------
    **kwargs
        Parameters passed to CatBoost
        
    Returns
    -------
    LearnerCatBoost
        Either LearnerCatBoostClassif or LearnerCatBoostRegr based on objective.
        
    Examples
    --------
    >>> from mlpy.learners import learner_catboost
    >>> 
    >>> # For classification with auto categorical detection
    >>> cb_clf = learner_catboost(
    ...     objective='Logloss',
    ...     cat_features='auto',
    ...     n_estimators=100
    ... )
    >>> 
    >>> # For regression with uncertainty  
    >>> cb_reg = learner_catboost(
    ...     objective='RMSE',
    ...     posterior_sampling=True,
    ...     n_estimators=100
    ... )
    >>> 
    >>> # With GPU support
    >>> cb_gpu = learner_catboost(task_type='GPU', n_estimators=100)
    >>> 
    >>> # With class balancing
    >>> cb_balanced = learner_catboost(auto_class_weights='Balanced')
    """
    objective = kwargs.get('objective', kwargs.get('loss_function', None))
    
    if objective:
        classification_objectives = ['Logloss', 'CrossEntropy', 'MultiClass', 
                                    'MultiClassOneVsAll', 'AUC']
        regression_objectives = ['RMSE', 'MAE', 'MAPE', 'Quantile', 'LogLinQuantile',
                                'Poisson', 'Tweedie', 'Huber', 'Lq']
        if any(obj in objective for obj in classification_objectives):
            return LearnerCatBoostClassif(**kwargs)
        elif any(obj in objective for obj in regression_objectives):
            return LearnerCatBoostRegr(**kwargs)
        else:
            # Default to classifier if objective not recognized
            return LearnerCatBoostClassif(**kwargs)
    else:
        # Default to classifier when no objective specified
        # User can specify loss_function='RMSE' if they want regression
        return LearnerCatBoostClassif(**kwargs)