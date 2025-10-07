"""Cross-validation resampling strategies."""

import numpy as np
from typing import Dict, Any, Optional, List

from .base import Resampling, register_resampling
from ..tasks import Task


@register_resampling
class ResamplingCV(Resampling):
    """K-fold cross-validation.
    
    Splits data into k folds, using each fold as test set once.
    
    Parameters
    ----------
    folds : int, default=10
        Number of folds.
    stratify : bool, default=False
        Whether to stratify folds by target variable (classification only).
    seed : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        folds: int = 10,
        stratify: bool = False,
        seed: Optional[int] = None
    ):
        if folds < 2:
            raise ValueError(f"Number of folds must be at least 2, got {folds}")
            
        super().__init__(
            id='cv',
            param_set={'folds': folds, 'stratify': stratify, 'seed': seed},
            iters=folds,
            duplicated_ids=False
        )
        self.folds = folds
        self.stratify = stratify
        self.seed = seed
        
    def _materialize(self, task: Task) -> Dict[str, Any]:
        """Create the k-fold splits."""
        # Get indices in use
        row_ids = sorted(task.row_roles['use'])
        n = len(row_ids)
        
        if self.folds > n:
            raise ValueError(f"Cannot have more folds ({self.folds}) than samples ({n})")
            
        # Set random state
        rng = np.random.RandomState(self.seed)
        indices = np.array(row_ids)
        
        if self.stratify and hasattr(task, 'target_names'):
            # Stratified k-fold
            target_data = task.data(cols=task.target_names)
            if len(task.target_names) != 1:
                raise ValueError("Stratification only supported for single target")
                
            target = target_data[task.target_names[0]]
            
            # Create folds maintaining class proportions
            fold_indices = [[] for _ in range(self.folds)]
            
            for class_label in np.unique(target):
                class_indices = indices[target == class_label]
                # Shuffle within class
                class_indices = rng.permutation(class_indices)
                
                # Distribute across folds
                for i, idx in enumerate(class_indices):
                    fold_indices[i % self.folds].append(idx)
                    
            # Convert to arrays and shuffle within folds
            fold_indices = [rng.permutation(fold) for fold in fold_indices]
        else:
            # Simple k-fold
            # Shuffle all indices
            indices = rng.permutation(indices)
            
            # Create folds
            fold_indices = []
            fold_size = n // self.folds
            remainder = n % self.folds
            
            start = 0
            for i in range(self.folds):
                # Add 1 extra sample to first 'remainder' folds
                size = fold_size + (1 if i < remainder else 0)
                fold_indices.append(indices[start:start + size])
                start += size
                
        return {'fold_indices': fold_indices}
        
    def _get_train_set(self, i: int) -> np.ndarray:
        """Get training indices for fold i."""
        fold_indices = self._instance['fold_indices']
        # Train set is all folds except i
        train_indices = []
        for j, fold in enumerate(fold_indices):
            if j != i:
                train_indices.extend(fold)
        return np.array(train_indices)
        
    def _get_test_set(self, i: int) -> np.ndarray:
        """Get test indices for fold i."""
        return self._instance['fold_indices'][i]


@register_resampling
class ResamplingLOO(Resampling):
    """Leave-one-out cross-validation.
    
    Special case of k-fold CV where k equals the number of samples.
    Each iteration uses a single sample as test set.
    """
    
    def __init__(self):
        super().__init__(
            id='loo',
            param_set={},
            iters=1,  # Will be updated in instantiate
            duplicated_ids=False
        )
        
    def instantiate(self, task: Task) -> "Resampling":
        """Instantiate LOO with task."""
        # Update iters to match number of samples in use
        self.iters = len(task.row_roles['use'])
        return super().instantiate(task)
        
    def _materialize(self, task: Task) -> Dict[str, Any]:
        """Create LOO splits."""
        row_ids = sorted(task.row_roles['use'])
        n = len(row_ids)
        return {'n': n, 'indices': np.array(row_ids)}
        
    def _get_train_set(self, i: int) -> np.ndarray:
        """Get training indices (all except i)."""
        indices = self._instance['indices']
        return np.concatenate([indices[:i], indices[i+1:]])
        
    def _get_test_set(self, i: int) -> np.ndarray:
        """Get test index (just i)."""
        return np.array([i])


@register_resampling  
class ResamplingRepeatedCV(Resampling):
    """Repeated k-fold cross-validation.
    
    Repeats k-fold CV multiple times with different random splits.
    
    Parameters
    ----------
    folds : int, default=10
        Number of folds per repetition.
    repeats : int, default=10
        Number of CV repetitions.
    stratify : bool, default=False
        Whether to stratify folds by target variable.
    seed : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        folds: int = 10,
        repeats: int = 10,
        stratify: bool = False,
        seed: Optional[int] = None
    ):
        if folds < 2:
            raise ValueError(f"Number of folds must be at least 2, got {folds}")
        if repeats < 1:
            raise ValueError(f"Number of repeats must be at least 1, got {repeats}")
            
        super().__init__(
            id='repeated_cv',
            param_set={
                'folds': folds,
                'repeats': repeats,
                'stratify': stratify,
                'seed': seed
            },
            iters=folds * repeats,
            duplicated_ids=False
        )
        self.folds = folds
        self.repeats = repeats
        self.stratify = stratify
        self.seed = seed
        
    def _materialize(self, task: Task) -> Dict[str, Any]:
        """Create repeated CV splits."""
        # Use base seed to generate seeds for each repetition
        base_rng = np.random.RandomState(self.seed)
        repeat_seeds = base_rng.randint(0, 2**31, size=self.repeats)
        
        # Create CV resampling for each repetition
        all_fold_indices = []
        
        for r in range(self.repeats):
            cv = ResamplingCV(
                folds=self.folds,
                stratify=self.stratify,
                seed=int(repeat_seeds[r])
            )
            cv.instantiate(task)
            fold_indices = cv._instance['fold_indices']
            all_fold_indices.extend(fold_indices)
            
        return {'fold_indices': all_fold_indices}
        
    def _get_train_set(self, i: int) -> np.ndarray:
        """Get training indices for iteration i."""
        # Determine which repeat and fold
        repeat_idx = i // self.folds
        fold_idx = i % self.folds
        
        # Get folds for this repeat
        start_idx = repeat_idx * self.folds
        fold_indices = self._instance['fold_indices'][start_idx:start_idx + self.folds]
        
        # Train set is all folds except fold_idx
        train_indices = []
        for j, fold in enumerate(fold_indices):
            if j != fold_idx:
                train_indices.extend(fold)
        return np.array(train_indices)
        
    def _get_test_set(self, i: int) -> np.ndarray:
        """Get test indices for iteration i."""
        return self._instance['fold_indices'][i]