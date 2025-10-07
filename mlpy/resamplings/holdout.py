"""Holdout resampling strategy."""

import numpy as np
from typing import Dict, Any, Optional

from .base import Resampling, register_resampling
from ..tasks import Task


@register_resampling
class ResamplingHoldout(Resampling):
    """Holdout resampling (single train/test split).
    
    Splits the data into a single training and test set.
    
    Parameters
    ----------
    ratio : float, default=0.67
        Proportion of data to use for training (0 < ratio < 1).
    stratify : bool, default=False
        Whether to stratify splits by target variable (classification only).
    seed : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self, 
        ratio: float = 0.67,
        stratify: bool = False,
        seed: Optional[int] = None
    ):
        if not 0 < ratio < 1:
            raise ValueError(f"ratio must be between 0 and 1, got {ratio}")
            
        super().__init__(
            id='holdout',
            param_set={'ratio': ratio, 'stratify': stratify, 'seed': seed},
            iters=1,
            duplicated_ids=False
        )
        self.ratio = ratio
        self.stratify = stratify
        self.seed = seed
        
    def _materialize(self, task: Task) -> Dict[str, Any]:
        """Create the train/test split."""
        # Get indices in use
        row_ids = sorted(task.row_roles['use'])
        n = len(row_ids)
        train_size = int(n * self.ratio)
        
        # Set random state
        rng = np.random.RandomState(self.seed)
        
        if self.stratify and hasattr(task, 'target_names'):
            # Stratified split for classification
            target_data = task.data(rows=row_ids, cols=task.target_names)
            if len(task.target_names) != 1:
                raise ValueError("Stratification only supported for single target")
                
            target = target_data[task.target_names[0]]
            
            # Get indices for each class
            train_indices = []
            test_indices = []
            
            # Create mapping from position to actual row_id
            for class_label in np.unique(target):
                class_positions = np.where(target == class_label)[0]
                class_row_ids = [row_ids[pos] for pos in class_positions]
                n_class = len(class_row_ids)
                n_train_class = int(n_class * self.ratio)
                
                # Shuffle class indices
                shuffled = rng.permutation(class_row_ids)
                train_indices.extend(shuffled[:n_train_class])
                test_indices.extend(shuffled[n_train_class:])
                
            # Shuffle final indices
            train_indices = rng.permutation(train_indices)
            test_indices = rng.permutation(test_indices)
        else:
            # Simple random split
            shuffled_ids = rng.permutation(row_ids)
            train_indices = shuffled_ids[:train_size]
            test_indices = shuffled_ids[train_size:]
            
        return {
            'train_indices': np.array(train_indices),
            'test_indices': np.array(test_indices)
        }
        
    def _get_train_set(self, i: int) -> np.ndarray:
        """Get training indices."""
        return self._instance['train_indices']
        
    def _get_test_set(self, i: int) -> np.ndarray:
        """Get test indices."""
        return self._instance['test_indices']