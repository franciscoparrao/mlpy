"""Bootstrap resampling strategy."""

import numpy as np
from typing import Dict, Any, Optional

from .base import Resampling, register_resampling
from ..tasks import Task


@register_resampling
class ResamplingBootstrap(Resampling):
    """Bootstrap resampling.
    
    Creates bootstrap samples by sampling with replacement.
    Test sets can be either out-of-bag (OOB) samples or a fixed ratio.
    
    Parameters
    ----------
    iters : int, default=30
        Number of bootstrap iterations.
    ratio : float, default=1.0
        Size of bootstrap sample as proportion of original size.
    oob : bool, default=True
        If True, use out-of-bag samples as test set.
        If False, use a separate test set split.
    test_ratio : float, default=0.33
        If oob=False, proportion of data to use for test set.
    stratify : bool, default=False
        Whether to stratify bootstrap samples by target variable.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskClassif
    >>> df = pd.DataFrame({'x': list(range(10)), 'y': [0,1]*5})
    >>> task = TaskClassif(df, target='y')
    >>> boot = ResamplingBootstrap(iters=2, ratio=0.8, oob=True, seed=42)
    >>> boot.instantiate(task)
    <ResamplingBootstrap[instantiated]:bootstrap>
    >>> [len(boot.train_set(i)) for i in range(2)]
    [8, 8]
    >>> all(len(boot.test_set(i)) > 0 for i in range(2))
    True
    
    Notes
    -----
    - Bootstrap allows duplicate IDs in the training set.
    - With ``oob=True``, the test set size varies by iteration and can be empty
      in rare cases; an emergency small test subset is created in that case.
    """
    
    def __init__(
        self,
        iters: int = 30,
        ratio: float = 1.0,
        oob: bool = True,
        test_ratio: float = 0.33,
        stratify: bool = False,
        seed: Optional[int] = None
    ):
        if ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        if not oob and not 0 < test_ratio < 1:
            raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
            
        super().__init__(
            id='bootstrap',
            param_set={
                'iters': iters,
                'ratio': ratio,
                'oob': oob,
                'test_ratio': test_ratio,
                'stratify': stratify,
                'seed': seed
            },
            iters=iters,
            duplicated_ids=True  # Bootstrap allows duplicates
        )
        self.ratio = ratio
        self.oob = oob
        self.test_ratio = test_ratio
        self.stratify = stratify
        self.seed = seed
        
    def _materialize(self, task: Task) -> Dict[str, Any]:
        """Create bootstrap samples."""
        # Get indices in use
        row_ids = sorted(task.row_roles['use'])
        n = len(row_ids)
        sample_size = int(n * self.ratio)
        
        # Set random state
        rng = np.random.RandomState(self.seed)
        
        # Pre-allocate arrays for all iterations
        train_indices_list = []
        test_indices_list = []
        
        if not self.oob:
            # Create fixed test set first
            test_size = int(n * self.test_ratio)
            all_indices = np.array(row_ids)
            fixed_test_indices = rng.choice(all_indices, size=test_size, replace=False)
            train_pool = np.setdiff1d(all_indices, fixed_test_indices)
        
        for i in range(self.iters):
            if self.stratify and hasattr(task, 'target_names'):
                # Stratified bootstrap
                target_data = task.data(rows=row_ids, cols=task.target_names)
                if len(task.target_names) != 1:
                    raise ValueError("Stratification only supported for single target")
                    
                target = target_data[task.target_names[0]]
                
                # Sample from each class
                train_indices = []
                
                for class_label in np.unique(target):
                    if self.oob:
                        class_positions = np.where(target == class_label)[0]
                        class_row_ids = [row_ids[pos] for pos in class_positions]
                        n_class = len(class_row_ids)
                        # Maintain class proportions
                        class_sample_size = int(sample_size * n_class / n)
                        
                        # Bootstrap sample within class
                        class_sample = rng.choice(
                            class_row_ids,
                            size=class_sample_size,
                            replace=True
                        )
                        train_indices.extend(class_sample)
                    else:
                        # For fixed test set, we need to handle this differently
                        # Get target values for train pool indices
                        train_pool_positions = [i for i, rid in enumerate(row_ids) if rid in train_pool]
                        train_pool_target = target.iloc[train_pool_positions] if hasattr(target, 'iloc') else target[train_pool_positions]
                        class_mask = train_pool_target == class_label
                        class_indices = train_pool[class_mask]
                        
                        n_class = len(class_indices)
                        # Maintain class proportions  
                        class_sample_size = int(sample_size * n_class / len(train_pool))
                        
                        # Bootstrap sample within class
                        if n_class > 0:
                            class_sample = rng.choice(
                                class_indices,
                                size=class_sample_size,
                                replace=True
                            )
                            train_indices.extend(class_sample)
                    
                train_indices = np.array(train_indices)
                # Shuffle to mix classes
                train_indices = rng.permutation(train_indices)
            else:
                # Simple bootstrap
                if self.oob:
                    train_indices = rng.choice(row_ids, size=sample_size, replace=True)
                else:
                    train_indices = rng.choice(train_pool, size=sample_size, replace=True)
                    
            # Determine test set
            if self.oob:
                # Out-of-bag samples
                test_indices = np.setdiff1d(row_ids, np.unique(train_indices))
                if len(test_indices) == 0:
                    # Rare case: all samples were selected
                    # Use a small random subset as test
                    test_indices = rng.choice(row_ids, size=max(1, n // 10), replace=False)
            else:
                test_indices = fixed_test_indices
                
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