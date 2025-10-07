"""
Ensemble learners for MLPY.

This module provides ensemble learning methods including:
- Voting (hard and soft)
- Stacking (with meta-learner)
- Blending (with holdout validation)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import warnings
from copy import deepcopy

from ..core.base import MLPYObject
from .base import Learner
from ..tasks import Task, TaskClassif, TaskRegr
from ..predictions import PredictionClassif, PredictionRegr
from ..resamplings import ResamplingHoldout, ResamplingCV
from ..measures import MeasureClassifAccuracy, MeasureRegrMSE


class LearnerEnsemble(Learner):
    """
    Base class for ensemble learners.
    
    All ensemble methods combine predictions from multiple base learners
    to produce a final prediction.
    """
    
    def __init__(
        self,
        base_learners: List[Learner],
        id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize ensemble learner.
        
        Parameters
        ----------
        base_learners : List[Learner]
            List of base learners to ensemble
        id : str, optional
            Learner identifier
        label : str, optional
            Learner label
        """
        super().__init__(id=id, label=label, **kwargs)
        
        if not base_learners:
            raise ValueError("At least one base learner is required")
        
        self.base_learners = base_learners
        self._trained_learners = None
        self._task_type = None
    
    @property
    def task_type(self) -> Optional[str]:
        """Type of task this learner can handle."""
        # Infer from base learners
        if self.base_learners:
            for learner in self.base_learners:
                if hasattr(learner, 'task_type') and learner.task_type:
                    return learner.task_type
        return self._task_type
    
    def _check_task_compatibility(self, task: Task):
        """Check if all base learners are compatible with the task."""
        task_type = "classif" if isinstance(task, TaskClassif) else "regr"
        
        for learner in self.base_learners:
            if hasattr(learner, 'task_type'):
                learner_type = learner.task_type
                if learner_type and learner_type != task_type:
                    raise ValueError(
                        f"Learner {learner.id} is for {learner_type} "
                        f"but task is {task_type}"
                    )
    
    def _get_base_predictions(
        self, 
        task: Task, 
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Get predictions from all base learners.
        
        Parameters
        ----------
        task : Task
            Task to predict on
        return_proba : bool
            Whether to return probabilities (classification only)
            
        Returns
        -------
        np.ndarray
            Array of shape (n_learners, n_samples) or 
            (n_learners, n_samples, n_classes) for probabilities
        """
        if self._trained_learners is None:
            raise RuntimeError("Ensemble not trained yet")
        
        predictions = []
        
        for learner in self._trained_learners:
            pred = learner.predict(task)
            
            if return_proba and isinstance(task, TaskClassif):
                # Get probabilities if available
                if hasattr(pred, 'prob') and pred.prob is not None:
                    # Convert to numpy array if it's a DataFrame
                    prob_array = pred.prob.values if hasattr(pred.prob, 'values') else pred.prob
                    predictions.append(prob_array)
                else:
                    # Convert hard predictions to one-hot if no probabilities
                    n_classes = len(task.class_names)
                    n_samples = len(pred.response)
                    proba = np.zeros((n_samples, n_classes))
                    for i, resp in enumerate(pred.response):
                        class_idx = task.class_names.index(str(resp))
                        proba[i, class_idx] = 1.0
                    predictions.append(proba)
            else:
                predictions.append(pred.response)
        
        return np.array(predictions)


class LearnerVoting(LearnerEnsemble):
    """
    Voting ensemble learner.
    
    Combines predictions using majority voting (hard) or 
    averaged probabilities (soft).
    
    Parameters
    ----------
    base_learners : List[Learner]
        List of base learners
    voting : str, default='hard'
        Voting type: 'hard' or 'soft'
    weights : List[float], optional
        Weights for each learner
    """
    
    def __init__(
        self,
        base_learners: List[Learner],
        voting: str = 'hard',
        weights: Optional[List[float]] = None,
        id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs
    ):
        super().__init__(base_learners=base_learners, id=id, label=label, **kwargs)
        
        if voting not in ['hard', 'soft']:
            raise ValueError("voting must be 'hard' or 'soft'")
        
        self.voting = voting
        
        if weights is not None:
            if len(weights) != len(base_learners):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of learners ({len(base_learners)})"
                )
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
        else:
            weights = np.ones(len(base_learners)) / len(base_learners)
        
        self.weights = weights
    
    def train(self, task: Task) -> 'LearnerVoting':
        """Train all base learners."""
        self._check_task_compatibility(task)
        self._task_type = "classif" if isinstance(task, TaskClassif) else "regr"
        
        # Train each base learner
        self._trained_learners = []
        for learner in self.base_learners:
            trained = deepcopy(learner)
            trained.train(task)
            self._trained_learners.append(trained)
        
        self._model = self._trained_learners  # Mark as trained
        return self
    
    def predict(self, task: Task) -> Union[PredictionClassif, PredictionRegr]:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        if isinstance(task, TaskClassif):
            return self._predict_classif(task)
        else:
            return self._predict_regr(task)
    
    def _predict_classif(self, task: TaskClassif) -> PredictionClassif:
        """Classification predictions using voting."""
        n_samples = task.nrow
        
        if self.voting == 'soft':
            # Get probability predictions
            all_probs = self._get_base_predictions(task, return_proba=True)
            
            # Weighted average of probabilities
            # all_probs shape: (n_learners, n_samples, n_classes)
            n_classes = all_probs.shape[2] if len(all_probs.shape) == 3 else all_probs.shape[1]
            avg_probs = np.zeros((n_samples, n_classes))
            for i, probs in enumerate(all_probs):
                avg_probs += self.weights[i] * probs
            
            # Get class with highest average probability
            predictions = []
            for proba_row in avg_probs:
                pred_idx = np.argmax(proba_row)
                predictions.append(task.class_names[pred_idx])
            
            return PredictionClassif(
                task=task,
                learner_id=self.id or "voting_ensemble",
                row_ids=list(range(n_samples)),
                truth=task.truth() if hasattr(task, 'truth') else None,
                response=predictions,
                prob=avg_probs
            )
        
        else:  # hard voting
            # Get hard predictions
            all_preds = self._get_base_predictions(task, return_proba=False)
            
            # Weighted voting
            predictions = []
            for sample_idx in range(n_samples):
                votes = {}
                for learner_idx, preds in enumerate(all_preds):
                    pred = str(preds[sample_idx])
                    if pred not in votes:
                        votes[pred] = 0
                    votes[pred] += self.weights[learner_idx]
                
                # Get class with most weighted votes
                best_class = max(votes, key=votes.get)
                predictions.append(best_class)
            
            return PredictionClassif(
                task=task,
                learner_id=self.id or "voting_ensemble",
                row_ids=list(range(n_samples)),
                truth=task.truth() if hasattr(task, 'truth') else None,
                response=predictions
            )
    
    def _predict_regr(self, task: TaskRegr) -> PredictionRegr:
        """Regression predictions using averaging."""
        # Get predictions from all learners
        all_preds = self._get_base_predictions(task, return_proba=False)
        
        # Weighted average
        predictions = np.zeros(task.nrow)
        for i, preds in enumerate(all_preds):
            predictions += self.weights[i] * preds
        
        return PredictionRegr(
            task=task,
            learner_id=self.id or "voting_ensemble",
            row_ids=list(range(task.nrow)),
            truth=task.truth() if hasattr(task, 'truth') else None,
            response=predictions
        )


class LearnerStacking(LearnerEnsemble):
    """
    Stacking ensemble learner.
    
    Uses a meta-learner to combine base learner predictions.
    
    Parameters
    ----------
    base_learners : List[Learner]
        List of base learners (level 0)
    meta_learner : Learner
        Meta-learner to combine predictions (level 1)
    use_proba : bool, default=False
        Whether to use probabilities as meta-features (classification only)
    cv_folds : int, default=5
        Number of CV folds for generating meta-features
    """
    
    def __init__(
        self,
        base_learners: List[Learner],
        meta_learner: Learner,
        use_proba: bool = False,
        cv_folds: int = 5,
        id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs
    ):
        super().__init__(base_learners=base_learners, id=id, label=label, **kwargs)
        
        self.meta_learner = meta_learner
        self.use_proba = use_proba
        self.cv_folds = cv_folds
        self._trained_meta = None
        self._meta_task = None
    
    def train(self, task: Task) -> 'LearnerStacking':
        """Train stacking ensemble."""
        self._check_task_compatibility(task)
        self._task_type = "classif" if isinstance(task, TaskClassif) else "regr"
        
        # Step 1: Generate meta-features using cross-validation
        meta_features = self._generate_meta_features(task)
        
        # Step 2: Train base learners on full dataset
        self._trained_learners = []
        for learner in self.base_learners:
            trained = deepcopy(learner)
            trained.train(task)
            self._trained_learners.append(trained)
        
        # Step 3: Create meta-task and train meta-learner
        self._create_meta_task(meta_features, task)
        self._trained_meta = deepcopy(self.meta_learner)
        self._trained_meta.train(self._meta_task)
        
        self._model = self._trained_meta  # Mark as trained
        return self
    
    def _generate_meta_features(self, task: Task) -> np.ndarray:
        """
        Generate meta-features using cross-validation.
        
        This prevents overfitting by ensuring base learner predictions
        are made on out-of-fold data.
        """
        n_samples = task.nrow
        
        # Determine number of meta-features
        if self.use_proba and isinstance(task, TaskClassif):
            n_classes = len(task.class_names)
            n_meta_features = len(self.base_learners) * n_classes
        else:
            n_meta_features = len(self.base_learners)
        
        meta_features = np.zeros((n_samples, n_meta_features))
        
        # Use cross-validation to generate out-of-fold predictions
        cv = ResamplingCV(folds=self.cv_folds, stratify=isinstance(task, TaskClassif))
        cv_instance = cv.instantiate(task)
        
        for fold in range(self.cv_folds):
            train_idx = cv_instance.train_set(fold)
            test_idx = cv_instance.test_set(fold)
            
            train_task = task.filter(train_idx)
            test_task = task.filter(test_idx)
            
            # Train each base learner on fold and predict
            feature_idx = 0
            for learner in self.base_learners:
                fold_learner = deepcopy(learner)
                fold_learner.train(train_task)
                pred = fold_learner.predict(test_task)
                
                if self.use_proba and isinstance(task, TaskClassif):
                    # Use probabilities as meta-features
                    if hasattr(pred, 'prob'):
                        meta_features[test_idx, feature_idx:feature_idx+n_classes] = pred.prob
                    else:
                        # One-hot encode predictions if no probabilities
                        for i, resp in enumerate(pred.response):
                            class_idx = task.class_names.index(str(resp))
                            meta_features[test_idx[i], feature_idx + class_idx] = 1.0
                    feature_idx += n_classes
                else:
                    # Use predictions as meta-features
                    if isinstance(task, TaskClassif):
                        # Encode categorical predictions
                        encoded = [task.class_names.index(str(r)) for r in pred.response]
                        meta_features[test_idx, feature_idx] = encoded
                    else:
                        meta_features[test_idx, feature_idx] = pred.response
                    feature_idx += 1
        
        # Check for NaN values and fill with zeros if imputation fails
        if np.isnan(meta_features).any():
            try:
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='constant', fill_value=0.0)
                meta_features = imputer.fit_transform(meta_features)
            except:
                # If imputation fails, fill with zeros
                meta_features = np.nan_to_num(meta_features, nan=0.0)
        
        return meta_features
    
    def _create_meta_task(self, meta_features: np.ndarray, original_task: Task):
        """Create task for meta-learner training."""
        # Create DataFrame with meta-features
        feature_names = []
        feature_idx = 0
        
        for i, learner in enumerate(self.base_learners):
            if self.use_proba and isinstance(original_task, TaskClassif):
                for class_name in original_task.class_names:
                    feature_names.append(f"learner_{i}_prob_{class_name}")
                    feature_idx += 1
            else:
                feature_names.append(f"learner_{i}_pred")
                feature_idx += 1
        
        # Validate feature names match meta_features shape
        if len(feature_names) != meta_features.shape[1]:
            # If mismatch, create simple feature names
            feature_names = [f"feature_{i}" for i in range(meta_features.shape[1])]
        
        meta_df = pd.DataFrame(meta_features, columns=feature_names)
        
        # Add original target
        target_data = original_task.data(cols=original_task.target_names)
        meta_df['target'] = target_data.values
        
        # Create appropriate task type
        if isinstance(original_task, TaskClassif):
            self._meta_task = TaskClassif(meta_df, target='target')
        else:
            self._meta_task = TaskRegr(meta_df, target='target')
    
    def predict(self, task: Task) -> Union[PredictionClassif, PredictionRegr]:
        """Make stacking predictions."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Get base learner predictions
        meta_features = self._get_meta_features_for_prediction(task)
        
        # Create meta-task for prediction
        meta_df = pd.DataFrame(
            meta_features,
            columns=self._meta_task.feature_names
        )
        
        if isinstance(task, TaskClassif):
            # Add dummy target for task creation - use all classes
            n_rows = len(meta_df)
            dummy_targets = [task.class_names[i % len(task.class_names)] for i in range(n_rows)]
            meta_df['target'] = dummy_targets
            pred_meta_task = TaskClassif(meta_df, target='target')
        else:
            meta_df['target'] = 0.0
            pred_meta_task = TaskRegr(meta_df, target='target')
        
        # Get meta-learner prediction
        return self._trained_meta.predict(pred_meta_task)
    
    def _get_meta_features_for_prediction(self, task: Task) -> np.ndarray:
        """Generate meta-features for prediction."""
        n_samples = task.nrow
        
        # Determine shape
        if self.use_proba and isinstance(task, TaskClassif):
            n_classes = len(task.class_names)
            n_meta_features = len(self.base_learners) * n_classes
        else:
            n_meta_features = len(self.base_learners)
        
        meta_features = np.zeros((n_samples, n_meta_features))
        
        # Get predictions from trained base learners
        feature_idx = 0
        for learner in self._trained_learners:
            pred = learner.predict(task)
            
            if self.use_proba and isinstance(task, TaskClassif):
                if hasattr(pred, 'prob'):
                    meta_features[:, feature_idx:feature_idx+n_classes] = pred.prob
                else:
                    # One-hot encode
                    for i, resp in enumerate(pred.response):
                        class_idx = task.class_names.index(str(resp))
                        meta_features[i, feature_idx + class_idx] = 1.0
                feature_idx += n_classes
            else:
                if isinstance(task, TaskClassif):
                    # Encode categorical predictions
                    encoded = [task.class_names.index(str(r)) for r in pred.response]
                    meta_features[:, feature_idx] = encoded
                else:
                    meta_features[:, feature_idx] = pred.response
                feature_idx += 1
        
        # Check for NaN values and fill with zeros
        if np.isnan(meta_features).any():
            meta_features = np.nan_to_num(meta_features, nan=0.0)
        
        return meta_features


class LearnerBlending(LearnerEnsemble):
    """
    Blending ensemble learner.
    
    Similar to stacking but uses a holdout validation set instead of CV
    to generate meta-features. This is faster but may be less robust.
    
    Parameters
    ----------
    base_learners : List[Learner]
        List of base learners
    meta_learner : Learner
        Meta-learner to combine predictions
    blend_ratio : float, default=0.2
        Ratio of data to use for blending (generating meta-features)
    use_proba : bool, default=False
        Whether to use probabilities as meta-features
    """
    
    def __init__(
        self,
        base_learners: List[Learner],
        meta_learner: Learner,
        blend_ratio: float = 0.2,
        use_proba: bool = False,
        id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs
    ):
        super().__init__(base_learners=base_learners, id=id, label=label, **kwargs)
        
        if not 0 < blend_ratio < 1:
            raise ValueError("blend_ratio must be between 0 and 1")
        
        self.meta_learner = meta_learner
        self.blend_ratio = blend_ratio
        self.use_proba = use_proba
        self._trained_meta = None
    
    def train(self, task: Task) -> 'LearnerBlending':
        """Train blending ensemble."""
        self._check_task_compatibility(task)
        self._task_type = "classif" if isinstance(task, TaskClassif) else "regr"
        
        # Split data into blend and train sets
        holdout = ResamplingHoldout(
            ratio=self.blend_ratio,
            stratify=isinstance(task, TaskClassif)
        )
        split = holdout.instantiate(task)
        
        train_idx = split.train_set(0)
        blend_idx = split.test_set(0)
        
        train_task = task.filter(train_idx)
        blend_task = task.filter(blend_idx)
        
        # Step 1: Train base learners on training set
        self._trained_learners = []
        for learner in self.base_learners:
            trained = deepcopy(learner)
            trained.train(train_task)
            self._trained_learners.append(trained)
        
        # Step 2: Generate meta-features on blend set
        meta_features = self._generate_blend_features(blend_task)
        
        # Step 3: Create meta-task and train meta-learner
        meta_df = self._create_meta_dataframe(meta_features, blend_task)
        
        if isinstance(task, TaskClassif):
            meta_task = TaskClassif(meta_df, target='target')
        else:
            meta_task = TaskRegr(meta_df, target='target')
        
        self._trained_meta = deepcopy(self.meta_learner)
        self._trained_meta.train(meta_task)
        
        # Store feature names for prediction
        self._meta_feature_names = list(meta_df.columns[:-1])  # Exclude target
        
        self._model = self._trained_meta  # Mark as trained
        return self
    
    def _generate_blend_features(self, task: Task) -> np.ndarray:
        """Generate meta-features from blend set."""
        n_samples = task.nrow
        
        # Determine number of meta-features
        if self.use_proba and isinstance(task, TaskClassif):
            n_classes = len(task.class_names)
            n_meta_features = len(self._trained_learners) * n_classes
        else:
            n_meta_features = len(self._trained_learners)
        
        meta_features = np.zeros((n_samples, n_meta_features))
        
        # Get predictions from base learners
        feature_idx = 0
        for learner in self._trained_learners:
            pred = learner.predict(task)
            
            if self.use_proba and isinstance(task, TaskClassif):
                if hasattr(pred, 'prob'):
                    meta_features[:, feature_idx:feature_idx+n_classes] = pred.prob
                else:
                    # One-hot encode predictions
                    for i, resp in enumerate(pred.response):
                        class_idx = task.class_names.index(str(resp))
                        meta_features[i, feature_idx + class_idx] = 1.0
                feature_idx += n_classes
            else:
                if isinstance(task, TaskClassif):
                    # Encode categorical predictions
                    encoded = [task.class_names.index(str(r)) for r in pred.response]
                    meta_features[:, feature_idx] = encoded
                else:
                    meta_features[:, feature_idx] = pred.response
                feature_idx += 1
        
        # Check for NaN values and fill with mean
        if np.isnan(meta_features).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='constant', fill_value=0.0)
            meta_features = imputer.fit_transform(meta_features)
        
        return meta_features
    
    def _create_meta_dataframe(
        self, 
        meta_features: np.ndarray, 
        task: Task
    ) -> pd.DataFrame:
        """Create DataFrame for meta-learner training."""
        feature_names = []
        
        for i in range(len(self._trained_learners)):
            if self.use_proba and isinstance(task, TaskClassif):
                for class_name in task.class_names:
                    feature_names.append(f"learner_{i}_prob_{class_name}")
            else:
                feature_names.append(f"learner_{i}_pred")
        
        meta_df = pd.DataFrame(meta_features, columns=feature_names)
        
        # Add target
        target_data = task.data(cols=task.target_names)
        meta_df['target'] = target_data.values
        
        return meta_df
    
    def predict(self, task: Task) -> Union[PredictionClassif, PredictionRegr]:
        """Make blending predictions."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Get predictions from base learners
        meta_features = self._generate_blend_features(task)
        
        # Create meta-task
        meta_df = pd.DataFrame(meta_features, columns=self._meta_feature_names)
        
        if isinstance(task, TaskClassif):
            # Add dummy target - use all classes
            n_rows = len(meta_df)
            dummy_targets = [task.class_names[i % len(task.class_names)] for i in range(n_rows)]
            meta_df['target'] = dummy_targets
            pred_task = TaskClassif(meta_df, target='target')
        else:
            meta_df['target'] = 0.0  # Dummy target
            pred_task = TaskRegr(meta_df, target='target')
        
        # Get meta-learner prediction
        return self._trained_meta.predict(pred_task)


# Helper function for creating ensemble learners
def create_ensemble(
    method: str,
    base_learners: List[Learner],
    **kwargs
) -> LearnerEnsemble:
    """
    Create an ensemble learner.
    
    Parameters
    ----------
    method : str
        Ensemble method: 'voting', 'stacking', or 'blending'
    base_learners : List[Learner]
        List of base learners
    **kwargs
        Additional arguments for the specific ensemble method
        
    Returns
    -------
    LearnerEnsemble
        Configured ensemble learner
    """
    methods = {
        'voting': LearnerVoting,
        'stacking': LearnerStacking,
        'blending': LearnerBlending
    }
    
    if method not in methods:
        raise ValueError(
            f"Unknown ensemble method: {method}. "
            f"Choose from {list(methods.keys())}"
        )
    
    return methods[method](base_learners=base_learners, **kwargs)


__all__ = [
    'LearnerEnsemble',
    'LearnerVoting', 
    'LearnerStacking',
    'LearnerBlending',
    'create_ensemble'
]