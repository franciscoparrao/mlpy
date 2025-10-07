"""H2O integration for MLPY.

This module provides learners that wrap H2O estimators,
allowing them to be used seamlessly within the MLPY framework.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union, Tuple
import warnings
import tempfile
import os

try:
    import h2o
    from h2o.estimators.estimator_base import H2OEstimator
    from h2o.frame import H2OFrame
    _H2O_AVAILABLE = True
except ImportError:
    _H2O_AVAILABLE = False
    H2OEstimator = None
    H2OFrame = None

from .base import Learner
from .classification import LearnerClassif
from .regression import LearnerRegr
from ..tasks import Task, TaskClassif, TaskRegr
from ..predictions import PredictionClassif, PredictionRegr
from ..utils.registry import mlpy_learners


class LearnerH2O(Learner):
    """Base learner for H2O integration.
    
    This class wraps H2O estimators to work with MLPY's
    Task/Learner/Prediction interface.
    
    Parameters
    ----------
    estimator : H2OEstimator
        An H2O estimator instance.
    id : str, optional
        Unique identifier. If None, uses estimator class name.
    init_h2o : bool, default=True
        Whether to initialize H2O automatically.
    h2o_init_args : dict, optional
        Arguments to pass to h2o.init()
    **kwargs
        Additional parameters passed to parent class.
    """
    
    def __init__(
        self,
        estimator: 'H2OEstimator',
        id: Optional[str] = None,
        init_h2o: bool = True,
        h2o_init_args: Optional[Dict] = None,
        **kwargs
    ):
        if not _H2O_AVAILABLE:
            raise ImportError(
                "H2O is not installed. Install it with: pip install h2o"
            )
            
        # Validate estimator
        if not isinstance(estimator, H2OEstimator):
            raise TypeError(
                f"estimator must be an H2OEstimator, got {type(estimator)}"
            )
            
        # Initialize H2O if requested
        if init_h2o:
            try:
                h2o.init(**(h2o_init_args or {}))
            except Exception:
                # H2O might already be initialized
                pass
                
        # Auto-generate ID if not provided
        if id is None:
            id = f"h2o.{estimator.__class__.__name__.lower()}"
            
        # Detect properties
        properties = self._detect_properties(estimator)
        
        # H2O requires the h2o package
        packages = {"h2o"}
        
        super().__init__(
            id=id,
            properties=properties,
            packages=packages,
            **kwargs
        )
        
        self.estimator = estimator
        self._model = None
        self._feature_names = None
        self._target_name = None
        self._task_type = None  # Will be set during training
        
    def _detect_properties(self, estimator: 'H2OEstimator') -> set:
        """Auto-detect learner properties from H2O estimator.
        
        Parameters
        ----------
        estimator : H2OEstimator
            The H2O estimator.
            
        Returns
        -------
        set
            Set of detected properties.
        """
        properties = set()
        
        # Most H2O models support probabilities
        model_type = estimator.__class__.__name__
        
        # Classification models
        if any(x in model_type for x in ['Classifier', 'GBM', 'RandomForest', 'DeepLearning']):
            properties.add('prob')
            
        # Models with variable importance
        if hasattr(estimator, 'varimp'):
            properties.add('importance')
            
        # Models that support SHAP
        if model_type in ['GBMEstimator', 'XGBoostEstimator', 'RandomForestEstimator']:
            properties.add('shap')
            
        return properties
        
    @property
    def task_type(self) -> str:
        """Return task type - determined during training."""
        if self._task_type is not None:
            return self._task_type
            
        # If not trained yet, try to infer from estimator
        model_type = self.estimator.__class__.__name__
        
        if 'Classifier' in model_type or hasattr(self.estimator, '_problem_type'):
            if hasattr(self.estimator, '_problem_type'):
                if self.estimator._problem_type == 'classification':
                    return 'classif'
                elif self.estimator._problem_type == 'regression':
                    return 'regr'
                    
        # Default based on common patterns
        if any(x in model_type for x in ['Classifier', 'NaiveBayes']):
            return 'classif'
        else:
            return 'regr'
            
    def _prepare_h2o_frame(self, task: Task, row_ids: Optional[List[int]] = None) -> Tuple[H2OFrame, List[str], str]:
        """Convert task data to H2O Frame.
        
        Parameters
        ----------
        task : Task
            The task containing data.
        row_ids : list of int, optional
            Subset of rows to use.
            
        Returns
        -------
        h2o_frame : H2OFrame
            The H2O frame.
        feature_cols : list of str
            Feature column names.
        target_col : str
            Target column name.
        """
        # Get data
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        # Get features and target
        feature_cols = task.feature_names
        target_col = task.target_names[0]
        all_cols = feature_cols + [target_col]
        
        # Get data as DataFrame
        df = task.data(rows=row_ids, cols=all_cols, data_format='dataframe')
        
        # Convert to H2O Frame
        h2o_frame = h2o.H2OFrame(df)
        
        # Set column types for classification
        if isinstance(task, TaskClassif) or self._task_type == 'classif':
            h2o_frame[target_col] = h2o_frame[target_col].asfactor()
            
        return h2o_frame, feature_cols, target_col
        
    def train(self, task: Task, row_ids: Optional[List[int]] = None) -> "LearnerH2O":
        """Train the H2O model.
        
        Parameters
        ----------
        task : Task
            The task to train on.
        row_ids : list of int, optional
            Subset of rows to use for training.
            
        Returns
        -------
        self : LearnerH2O
            The trained learner.
        """
        # Set task type based on actual task
        if isinstance(task, TaskClassif):
            self._task_type = 'classif'
        elif isinstance(task, TaskRegr):
            self._task_type = 'regr'
        else:
            raise TypeError(f"Unknown task type: {type(task)}")
            
        # Prepare H2O data
        h2o_frame, feature_cols, target_col = self._prepare_h2o_frame(task, row_ids)
        
        # Store column names
        self._feature_names = feature_cols
        self._target_name = target_col
        
        # Train the model
        try:
            self.estimator.train(
                x=feature_cols,
                y=target_col,
                training_frame=h2o_frame
            )
            self._model = self.estimator
            self._train_task = task
        except Exception as e:
            raise RuntimeError(f"H2O training failed: {e}") from e
            
        return self
        
    def predict(self, task: Task, row_ids: Optional[List[int]] = None) -> Union[PredictionClassif, PredictionRegr]:
        """Make predictions using the trained H2O model.
        
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
            
        # Prepare H2O data (only features needed)
        if row_ids is None:
            row_ids = sorted(task.row_roles['use'])
            
        # Get features only
        df = task.data(rows=row_ids, cols=self._feature_names, data_format='dataframe')
        h2o_frame = h2o.H2OFrame(df)
        
        # Get truth values
        truth = task.truth(rows=row_ids)
        
        if self._task_type == 'classif':
            return self._predict_classif(h2o_frame, truth, task, row_ids)
        else:
            return self._predict_regr(h2o_frame, truth, task, row_ids)
            
    def _predict_classif(
        self, 
        h2o_frame: 'H2OFrame',
        truth: np.ndarray,
        task: TaskClassif,
        row_ids: List[int]
    ) -> PredictionClassif:
        """Make classification predictions."""
        # Get predictions
        h2o_preds = self._model.predict(h2o_frame)
        
        # Convert to pandas/numpy
        preds_df = h2o_preds.as_data_frame()
        
        # Get response (predicted class)
        response = preds_df['predict'].values
        
        # Get probabilities if available
        prob = None
        if self.predict_type == 'prob' or 'prob' in self.properties:
            # H2O probability columns are usually named p0, p1, etc.
            prob_cols = [col for col in preds_df.columns if col.startswith('p') and col[1:].isdigit()]
            if prob_cols:
                prob = preds_df[prob_cols].values
                
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
        h2o_frame: 'H2OFrame',
        truth: np.ndarray,
        task: TaskRegr,
        row_ids: List[int]
    ) -> PredictionRegr:
        """Make regression predictions."""
        # Get predictions
        h2o_preds = self._model.predict(h2o_frame)
        
        # Convert to pandas/numpy
        preds_df = h2o_preds.as_data_frame()
        
        # Get response
        response = preds_df['predict'].values
        
        return PredictionRegr(
            task=task,
            learner_id=self.id,
            row_ids=row_ids,
            truth=truth,
            response=response,
            se=None  # H2O doesn't provide standard errors by default
        )
        
    @property
    def model(self):
        """Access to the underlying H2O model."""
        return self._model
        
    @property
    def feature_importances(self) -> Optional[pd.DataFrame]:
        """Get feature importances if available."""
        if self.is_trained and hasattr(self._model, 'varimp'):
            try:
                return self._model.varimp(use_pandas=True)
            except:
                return None
        return None
        
    def clone(self, deep: bool = True) -> "LearnerH2O":
        """Create a copy of the learner.
        
        Parameters
        ----------
        deep : bool, default=True
            Whether to make a deep copy.
            
        Returns
        -------
        LearnerH2O
            A copy of the learner.
        """
        # Create new instance with same parameters
        new_learner = self.__class__(
            estimator=self.estimator.__class__(**self.estimator._parms),
            id=self.id,
            init_h2o=False  # Don't reinitialize H2O
        )
        
        # Copy state if trained
        if deep and self.is_trained:
            # H2O models can be saved/loaded
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h2o') as tmp:
                model_path = h2o.save_model(self._model, path=tmp.name, force=True)
                new_learner._model = h2o.load_model(model_path)
                os.unlink(model_path)
                
            new_learner._feature_names = self._feature_names.copy()
            new_learner._target_name = self._target_name
            new_learner._train_task = self._train_task
            new_learner._task_type = self._task_type
            
        return new_learner
        
    def reset(self) -> "LearnerH2O":
        """Reset the learner to untrained state.
        
        Returns
        -------
        self : LearnerH2O
            The reset learner.
        """
        self._model = None
        self._feature_names = None
        self._target_name = None
        self._train_task = None
        self._task_type = None
        return self


class LearnerClassifH2O(LearnerH2O, LearnerClassif):
    """H2O classifier wrapper.
    
    This class specifically wraps H2O classifiers.
    
    Parameters
    ----------
    estimator : H2O classifier
        An H2O classifier instance.
    **kwargs
        Additional parameters.
    """
    
    def __init__(
        self,
        estimator: 'H2OEstimator',
        predict_type: str = "response",
        **kwargs
    ):
        super().__init__(estimator, predict_type=predict_type, **kwargs)
        
        # Validate it's a classifier
        if self.task_type != 'classif':
            warnings.warn(
                f"{estimator.__class__.__name__} may not be a classifier. "
                f"Consider using LearnerRegrH2O instead."
            )


class LearnerRegrH2O(LearnerH2O, LearnerRegr):
    """H2O regressor wrapper.
    
    This class specifically wraps H2O regressors.
    
    Parameters
    ----------
    estimator : H2O regressor
        An H2O regressor instance.
    **kwargs
        Additional parameters.
    """
    
    def __init__(
        self,
        estimator: 'H2OEstimator',
        **kwargs
    ):
        # Regressors only support response predictions
        super().__init__(estimator, predict_type="response", **kwargs)
        
        # Validate it's a regressor
        if self.task_type != 'regr':
            warnings.warn(
                f"{estimator.__class__.__name__} may not be a regressor. "
                f"Consider using LearnerClassifH2O instead."
            )


# Register learners
mlpy_learners.register("h2o", LearnerH2O)
mlpy_learners.register("h2o.classif", LearnerClassifH2O)
mlpy_learners.register("h2o.regr", LearnerRegrH2O)


def learner_h2o(estimator: 'H2OEstimator', **kwargs) -> Union[LearnerClassifH2O, LearnerRegrH2O]:
    """Create an H2O learner with automatic type detection.
    
    This function automatically creates the appropriate learner type
    (classification or regression) based on the H2O estimator.
    
    Parameters
    ----------
    estimator : H2OEstimator
        An H2O estimator instance.
    **kwargs
        Additional parameters passed to the learner.
        
    Returns
    -------
    LearnerH2O
        Either LearnerClassifH2O or LearnerRegrH2O.
        
    Examples
    --------
    >>> import h2o
    >>> from h2o.estimators import H2ORandomForestEstimator
    >>> from mlpy.learners import learner_h2o
    >>> 
    >>> # Automatically creates LearnerClassifH2O
    >>> rf = H2ORandomForestEstimator(ntrees=100)
    >>> learner = learner_h2o(rf)
    """
    # Try to detect task type
    temp_learner = LearnerH2O(estimator, init_h2o=False)
    
    if temp_learner.task_type == 'classif':
        return LearnerClassifH2O(estimator, **kwargs)
    else:
        return LearnerRegrH2O(estimator, **kwargs)