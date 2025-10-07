"""
Ensemble feature filters for MLPY.

These filters combine multiple filter methods to create
more robust feature selection.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Union, Callable
import warnings

from .base import Filter, FilterResult, filter_registry
from ..tasks.base import Task


class FilterEnsemble(Filter):
    """Ensemble filter combining multiple filter methods.
    
    Combines scores from multiple filters using various
    aggregation strategies.
    
    Parameters
    ----------
    filters : List[Filter] or List[str]
        List of filter instances or filter IDs.
    weights : List[float], optional
        Weights for each filter. If None, equal weights.
    aggregation : str or callable, default='mean'
        How to combine scores: 'mean', 'median', 'min', 'max',
        'rank_mean', or custom function.
    normalize : bool, default=True
        Whether to normalize scores before aggregation.
    """
    
    def __init__(
        self,
        filters: List[Union[Filter, str]],
        weights: Optional[List[float]] = None,
        aggregation: Union[str, Callable] = 'mean',
        normalize: bool = True
    ):
        super().__init__(
            id="ensemble",
            task_types={'any'},
            feature_types={'any'}
        )
        
        # Convert filter IDs to instances
        self.filters = []
        for f in filters:
            if isinstance(f, str):
                self.filters.append(filter_registry.create(f))
            else:
                self.filters.append(f)
                
        self.weights = weights
        if weights is not None and len(weights) != len(self.filters):
            raise ValueError("Number of weights must match number of filters")
            
        self.aggregation = aggregation
        self.normalize = normalize
        
    def _normalize_scores(self, scores: pd.Series) -> pd.Series:
        """Normalize scores to [0, 1] range."""
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
        else:
            return pd.Series(0.5, index=scores.index)
            
    def _aggregate_scores(self, all_scores: pd.DataFrame) -> pd.Series:
        """Aggregate scores from multiple filters."""
        if self.weights is not None:
            # Weighted aggregation
            weighted_scores = all_scores * self.weights
            
        if callable(self.aggregation):
            return self.aggregation(all_scores)
        elif self.aggregation == 'mean':
            return all_scores.mean(axis=1)
        elif self.aggregation == 'median':
            return all_scores.median(axis=1)
        elif self.aggregation == 'min':
            return all_scores.min(axis=1)
        elif self.aggregation == 'max':
            return all_scores.max(axis=1)
        elif self.aggregation == 'rank_mean':
            # Average of ranks
            ranks = all_scores.rank(method='average', ascending=False)
            return -ranks.mean(axis=1)  # Negative so higher is better
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
            
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        if features is None:
            features = task.feature_names
            
        # Collect scores from all filters
        all_scores = pd.DataFrame(index=features)
        filter_results = []
        
        for i, filter_obj in enumerate(self.filters):
            try:
                result = filter_obj.calculate(task, features)
                filter_results.append(result)
                
                # Align scores with feature list
                scores = result.scores.reindex(features, fill_value=0.0)
                
                if self.normalize:
                    scores = self._normalize_scores(scores)
                    
                all_scores[f"filter_{i}"] = scores
                
            except Exception as e:
                warnings.warn(f"Filter {filter_obj.id} failed: {e}")
                all_scores[f"filter_{i}"] = 0.0
                
        if all_scores.empty:
            raise RuntimeError("All filters failed")
            
        # Aggregate scores
        final_scores = self._aggregate_scores(all_scores)
        
        # Sort features by score
        sorted_features = final_scores.sort_values(ascending=False).index.tolist()
        
        # Determine task type from successful filters
        task_types = [r.task_type for r in filter_results if r is not None]
        task_type = task_types[0] if task_types else "unknown"
        
        return FilterResult(
            scores=final_scores,
            features=sorted_features,
            method="ensemble",
            task_type=task_type,
            params={
                "filters": [f.id for f in self.filters],
                "aggregation": self.aggregation,
                "normalize": self.normalize,
                "individual_scores": all_scores
            }
        )


class FilterStability(Filter):
    """Stability selection filter.
    
    Runs filters on multiple subsamples and selects features
    that are consistently selected.
    
    Parameters
    ----------
    filter : Filter or str
        Base filter to use.
    n_iterations : int, default=100
        Number of subsampling iterations.
    sample_fraction : float, default=0.5
        Fraction of samples to use in each iteration.
    threshold : float, default=0.6
        Minimum selection frequency for a feature.
    n_features : int, optional
        Number of features to select in each iteration.
    """
    
    def __init__(
        self,
        filter: Union[Filter, str],
        n_iterations: int = 100,
        sample_fraction: float = 0.5,
        threshold: float = 0.6,
        n_features: Optional[int] = None
    ):
        super().__init__(
            id="stability",
            task_types={'any'},
            feature_types={'any'}
        )
        
        if isinstance(filter, str):
            self.filter = filter_registry.create(filter)
        else:
            self.filter = filter
            
        self.n_iterations = n_iterations
        self.sample_fraction = sample_fraction
        self.threshold = threshold
        self.n_features = n_features
        
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        if features is None:
            features = task.feature_names
            
        # Determine number of features to select
        if self.n_features is None:
            n_select = max(1, len(features) // 2)
        else:
            n_select = min(self.n_features, len(features))
            
        # Count selections
        selection_counts = pd.Series(0, index=features)
        
        # Run multiple iterations
        n_samples = task.nrow
        sample_size = int(n_samples * self.sample_fraction)
        
        for i in range(self.n_iterations):
            # Random subsample
            sample_indices = np.random.choice(
                list(task.row_roles['use']),
                size=sample_size,
                replace=False
            )
            
            # Create subtask
            subtask = task.clone()
            subtask.set_row_roles({'use': sample_indices})
            
            try:
                # Run filter
                result = self.filter.calculate(subtask, features)
                
                # Select top features
                selected = result.select_top_k(n_select)
                
                # Update counts
                for feat in selected:
                    selection_counts[feat] += 1
                    
            except Exception as e:
                warnings.warn(f"Iteration {i} failed: {e}")
                
        # Calculate selection frequencies
        selection_freq = selection_counts / self.n_iterations
        
        # Sort by frequency
        sorted_features = selection_freq.sort_values(ascending=False).index.tolist()
        
        # Determine task type
        try:
            sample_result = self.filter.calculate(task, features[:5])
            task_type = sample_result.task_type
        except:
            task_type = "unknown"
            
        return FilterResult(
            scores=selection_freq,
            features=sorted_features,
            method="stability",
            task_type=task_type,
            params={
                "filter": self.filter.id,
                "n_iterations": self.n_iterations,
                "sample_fraction": self.sample_fraction,
                "threshold": self.threshold,
                "n_features": n_select
            }
        )


class FilterAutoSelect(Filter):
    """Automatic filter selection based on data characteristics.
    
    Automatically chooses appropriate filters based on:
    - Task type (classification/regression)
    - Number of features
    - Number of samples
    - Feature types
    
    Parameters
    ----------
    max_filters : int, default=3
        Maximum number of filters to use in ensemble.
    prefer_fast : bool, default=True
        Prefer faster filters for large datasets.
    """
    
    def __init__(self, max_filters: int = 3, prefer_fast: bool = True):
        super().__init__(
            id="auto_select",
            task_types={'any'},
            feature_types={'any'}
        )
        self.max_filters = max_filters
        self.prefer_fast = prefer_fast
        
    def _select_filters(self, task: Task, features: List[str]) -> List[Filter]:
        """Select appropriate filters based on data characteristics."""
        selected = []
        
        n_samples = task.nrow
        n_features = len(features)
        is_classification = hasattr(task, 'n_classes')
        
        # Always include variance filter for removing constants
        selected.append(filter_registry.create('variance'))
        
        # For small datasets, use all methods
        if n_samples < 1000 and n_features < 50:
            if is_classification:
                selected.extend([
                    filter_registry.create('anova'),
                    filter_registry.create('mutual_info'),
                    filter_registry.create('chi2')
                ])
            else:
                selected.extend([
                    filter_registry.create('f_regression'),
                    filter_registry.create('mutual_info'),
                    filter_registry.create('correlation')
                ])
                
        # For medium datasets
        elif n_samples < 10000:
            if is_classification:
                selected.extend([
                    filter_registry.create('anova'),
                    filter_registry.create('importance', n_estimators=50)
                ])
            else:
                selected.extend([
                    filter_registry.create('f_regression'),
                    filter_registry.create('correlation')
                ])
                
        # For large datasets, use fast methods
        else:
            if self.prefer_fast:
                if is_classification:
                    selected.append(filter_registry.create('anova'))
                else:
                    selected.append(filter_registry.create('f_regression'))
                selected.append(filter_registry.create('correlation'))
            else:
                selected.append(filter_registry.create('importance', n_estimators=30))
                
        # Limit number of filters
        return selected[:self.max_filters]
        
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        if features is None:
            features = task.feature_names
            
        # Select appropriate filters
        filters = self._select_filters(task, features)
        
        # Create ensemble
        ensemble = FilterEnsemble(
            filters=filters,
            aggregation='rank_mean',
            normalize=True
        )
        
        # Run ensemble
        result = ensemble.calculate(task, features)
        
        # Update method name and params
        result.method = "auto_select"
        result.params["selected_filters"] = [f.id for f in filters]
        result.params["n_samples"] = task.nrow
        result.params["n_features"] = len(features)
        
        return result


# Register ensemble filters
filter_registry.register(FilterEnsemble, "ensemble")
filter_registry.register(FilterStability, "stability")
filter_registry.register(FilterAutoSelect, "auto")