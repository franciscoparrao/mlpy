"""Subsampling (Monte Carlo CV) resampling strategy."""

import numpy as np
from typing import Dict, Any, Optional

from .base import Resampling, register_resampling
from ..tasks import Task


@register_resampling
class ResamplingSubsampling(Resampling):
    """Repeated random subsampling (Monte Carlo cross-validation).
    
    Repeatedly splits data into random train/test sets.
    Unlike bootstrap, sampling is without replacement.
    
    Parameters
    ----------
    iters : int, default=30
        Number of subsampling iterations.
    ratio : float, default=0.67
        Proportion of data to use for training.
    stratify : bool, default=False
        Whether to stratify splits by target variable.
    seed : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        iters: int = 30,
        ratio: float = 0.67,
        stratify: bool = False,
        seed: Optional[int] = None
    ):
        if not 0 < ratio < 1:
            raise ValueError(f"ratio must be between 0 and 1, got {ratio}")
            
        super().__init__(
            id='subsampling',
            param_set={
                'iters': iters,
                'ratio': ratio,
                'stratify': stratify,
                'seed': seed
            },
            iters=iters,
            duplicated_ids=False
        )
        self.ratio = ratio
        self.stratify = stratify
        self.seed = seed
        
    def _materialize(self, task: Task) -> Dict[str, Any]:
        """Create subsampling splits."""
        # Get indices in use
        row_ids = sorted(task.row_roles['use'])
        n = len(row_ids)
        train_size = int(n * self.ratio)
        
        # Set random state
        rng = np.random.RandomState(self.seed)
        
        # Pre-allocate arrays for all iterations
        train_indices_list = []
        test_indices_list = []
        
        for i in range(self.iters):
            if self.stratify and hasattr(task, 'target_names'):
                # Stratified split
                target_data = task.data(rows=row_ids, cols=task.target_names)
                if len(task.target_names) != 1:
                    raise ValueError("Stratification only supported for single target")
                    
                target = target_data[task.target_names[0]]
                
                # Get indices for each class
                train_indices = []
                test_indices = []
                
                for class_label in np.unique(target):
                    class_positions = np.where(target == class_label)[0]
                    class_row_ids = [row_ids[pos] for pos in class_positions]
                    n_class = len(class_row_ids)
                    n_train_class = int(n_class * self.ratio)
                    
                    # Shuffle and split
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
                
            train_indices_list.append(train_indices)
            test_indices_list.append(test_indices)
            
        return {
            'train_indices': train_indices_list,
            'test_indices': test_indices_list
        }
        
    def _get_train_set(self, i: int) -> np.ndarray:
        """Get training indices for iteration i."""
        return self._instance['train_indices'][i]
        
    def _get_test_set(self, i: int) -> np.ndarray:
        """Get test indices for iteration i."""
        return self._instance['test_indices'][i]