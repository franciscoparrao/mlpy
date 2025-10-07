"""
Ensemble ranking system for combining multiple feature selection methods.

This module implements the Cumulative Feature Selection (CFS) approach
similar to the one used in the R script, allowing combination of multiple
filter methods to create a robust feature ranking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings
from scipy.stats import rankdata

from .base import Filter, FilterResult
from ..tasks.base import Task
from ..utils.registry import mlpy_filters


@mlpy_filters.register("ensemble_ranking")
class EnsembleRanking(Filter):
    """
    Ensemble feature ranking by combining multiple filter methods.
    
    This filter combines results from multiple feature selection methods
    using various aggregation strategies to produce a robust ranking.
    Similar to the Cumulative Feature Selection (CFS) approach.
    
    Parameters
    ----------
    filters : List[Filter] or List[str]
        List of filter objects or filter names to use
    aggregation : str, default='mean_rank'
        Aggregation method: 'mean_rank', 'median_rank', 'min_rank',
        'borda_count', 'weighted_sum', 'robust_rank'
    weights : List[float], optional
        Weights for each filter (for weighted aggregation)
    normalize_scores : bool, default=True
        Whether to normalize scores before aggregation
    handle_missing : str, default='worst'
        How to handle features not scored by all filters:
        'worst', 'mean', 'median', 'exclude'
    top_k_per_filter : int, optional
        Only consider top k features from each filter
        
    Examples
    --------
    >>> from mlpy.filters import EnsembleRanking, MRMR, Relief, DISR
    >>> 
    >>> # Create ensemble with multiple filters
    >>> ensemble = EnsembleRanking(
    ...     filters=['mrmr', 'relief', 'disr', 'anova'],
    ...     aggregation='mean_rank'
    ... )
    >>> result = ensemble.calculate(task)
    >>> 
    >>> # Get top 30 features
    >>> top_features = result.select_top_k(30)
    """
    
    def __init__(
        self,
        filters: List[Union[Filter, str]],
        aggregation: str = 'mean_rank',
        weights: Optional[List[float]] = None,
        normalize_scores: bool = True,
        handle_missing: str = 'worst',
        top_k_per_filter: Optional[int] = None
    ):
        super().__init__(
            id="ensemble_ranking",
            task_types={'any'},
            feature_types={'any'}
        )
        self.filters = filters
        self.aggregation = aggregation
        self.weights = weights
        self.normalize_scores = normalize_scores
        self.handle_missing = handle_missing
        self.top_k_per_filter = top_k_per_filter
        
        # Validate parameters
        valid_aggregations = ['mean_rank', 'median_rank', 'min_rank', 
                             'borda_count', 'weighted_sum', 'robust_rank']
        if aggregation not in valid_aggregations:
            raise ValueError(f"Invalid aggregation: {aggregation}. "
                           f"Must be one of {valid_aggregations}")
        
        if weights is not None and len(weights) != len(filters):
            raise ValueError("Number of weights must match number of filters")
    
    def _get_filter_instance(self, filter_spec: Union[Filter, str]) -> Filter:
        """Get filter instance from specification."""
        if isinstance(filter_spec, str):
            # Import all filter modules to ensure registration
            from . import mutual_information, statistical
            
            # Get filter from registry
            filter_class = mlpy_filters.get(filter_spec)
            if filter_class is None:
                raise ValueError(f"Unknown filter: {filter_spec}")
            return filter_class()
        else:
            return filter_spec
    
    def _normalize_scores(self, scores: pd.Series) -> pd.Series:
        """Normalize scores to [0, 1] range."""
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            return pd.Series(0.5, index=scores.index)
        
        return (scores - min_score) / (max_score - min_score)
    
    def _scores_to_ranks(self, scores: pd.Series, ascending: bool = False) -> pd.Series:
        """Convert scores to ranks."""
        # Higher score = better = lower rank (1 is best)
        if ascending:
            ranks = rankdata(scores, method='average')
        else:
            ranks = rankdata(-scores, method='average')
        
        return pd.Series(ranks, index=scores.index)
    
    def calculate(self, task: Task) -> FilterResult:
        """Calculate ensemble ranking for features."""
        # Run all filters
        filter_results = []
        filter_names = []
        
        for i, filter_spec in enumerate(self.filters):
            try:
                # Get filter instance
                filter_obj = self._get_filter_instance(filter_spec)
                filter_name = filter_obj.id if hasattr(filter_obj, 'id') else f"filter_{i}"
                
                # Calculate scores
                result = filter_obj.calculate(task)
                filter_results.append(result)
                filter_names.append(filter_name)
                
            except Exception as e:
                warnings.warn(f"Filter {filter_spec} failed: {e}")
                continue
        
        if not filter_results:
            raise ValueError("No filters succeeded")
        
        # Collect all feature names
        all_features = set()
        for result in filter_results:
            all_features.update(result.scores.index)
        all_features = sorted(list(all_features))
        
        # Create score matrix (features x filters)
        score_matrix = pd.DataFrame(index=all_features, columns=filter_names)
        rank_matrix = pd.DataFrame(index=all_features, columns=filter_names)
        
        for i, (result, name) in enumerate(zip(filter_results, filter_names)):
            # Get scores
            scores = result.scores
            
            # Normalize if requested
            if self.normalize_scores:
                scores = self._normalize_scores(scores)
            
            # Apply top-k filtering if specified
            if self.top_k_per_filter is not None:
                top_features = result.select_top_k(self.top_k_per_filter)
                mask = ~scores.index.isin(top_features)
                scores[mask] = 0
            
            # Store scores
            score_matrix[name] = scores
            
            # Convert to ranks
            ranks = self._scores_to_ranks(scores)
            rank_matrix[name] = ranks
        
        # Handle missing values
        if self.handle_missing == 'worst':
            # Assign worst rank to missing values
            max_rank = len(all_features)
            score_matrix.fillna(0, inplace=True)
            rank_matrix.fillna(max_rank, inplace=True)
        elif self.handle_missing == 'mean':
            score_matrix.fillna(score_matrix.mean(axis=1), inplace=True)
            rank_matrix.fillna(rank_matrix.mean(axis=1), inplace=True)
        elif self.handle_missing == 'median':
            score_matrix.fillna(score_matrix.median(axis=1), inplace=True)
            rank_matrix.fillna(rank_matrix.median(axis=1), inplace=True)
        elif self.handle_missing == 'exclude':
            # Remove features with any missing values
            complete_mask = ~score_matrix.isna().any(axis=1)
            score_matrix = score_matrix[complete_mask]
            rank_matrix = rank_matrix[complete_mask]
            all_features = [f for f in all_features if f in score_matrix.index]
        
        # Apply aggregation
        if self.aggregation == 'mean_rank':
            # Average rank across filters
            if self.weights is not None:
                final_scores = (rank_matrix * self.weights).sum(axis=1) / sum(self.weights)
            else:
                final_scores = rank_matrix.mean(axis=1)
            # Lower rank is better, so invert for scores
            final_scores = len(all_features) + 1 - final_scores
            
        elif self.aggregation == 'median_rank':
            # Median rank across filters
            final_scores = rank_matrix.median(axis=1)
            final_scores = len(all_features) + 1 - final_scores
            
        elif self.aggregation == 'min_rank':
            # Best rank across filters
            final_scores = rank_matrix.min(axis=1)
            final_scores = len(all_features) + 1 - final_scores
            
        elif self.aggregation == 'borda_count':
            # Borda count: points based on ranking
            borda_scores = len(all_features) + 1 - rank_matrix
            if self.weights is not None:
                final_scores = (borda_scores * self.weights).sum(axis=1)
            else:
                final_scores = borda_scores.sum(axis=1)
            
        elif self.aggregation == 'weighted_sum':
            # Weighted sum of normalized scores
            if self.weights is not None:
                final_scores = (score_matrix * self.weights).sum(axis=1)
            else:
                final_scores = score_matrix.sum(axis=1)
            
        elif self.aggregation == 'robust_rank':
            # Robust ranking: trim outliers before averaging
            trimmed_ranks = rank_matrix.apply(
                lambda x: x[(x >= x.quantile(0.1)) & (x <= x.quantile(0.9))].mean(),
                axis=1
            )
            final_scores = len(all_features) + 1 - trimmed_ranks
        
        # Sort features by final score
        final_scores = pd.Series(final_scores, index=score_matrix.index)
        sorted_features = final_scores.sort_values(ascending=False).index.tolist()
        
        return FilterResult(
            scores=final_scores,
            features=sorted_features,
            method="ensemble_ranking",
            task_type=task.task_type,
            params={
                'filters': filter_names,
                'aggregation': self.aggregation,
                'weights': self.weights,
                'normalize_scores': self.normalize_scores,
                'handle_missing': self.handle_missing,
                'individual_scores': score_matrix.to_dict(),
                'individual_ranks': rank_matrix.to_dict()
            }
        )


@mlpy_filters.register("cumulative_ranking")
class CumulativeRanking(EnsembleRanking):
    """
    Cumulative Feature Selection (CFS) - Simplified ensemble ranking.
    
    This is a simplified version of EnsembleRanking that mimics the
    approach used in the R script for combining multiple filter methods.
    It normalizes scores, calculates ranks, and combines them.
    
    Parameters
    ----------
    filters : List[str]
        List of filter names to use. Default includes common filters.
    n_features : int, optional
        Number of features to select
        
    Examples
    --------
    >>> from mlpy.filters import CumulativeRanking
    >>> 
    >>> # Use default filters
    >>> cfs = CumulativeRanking()
    >>> result = cfs.calculate(task)
    >>> 
    >>> # Select top 30 features
    >>> selected_features = result.select_top_k(30)
    >>> print(selected_features)
    """
    
    def __init__(
        self,
        filters: Optional[List[str]] = None,
        n_features: Optional[int] = None
    ):
        # Default filters similar to R script
        if filters is None:
            filters = [
                'mrmr',      # Minimum Redundancy Maximum Relevance
                'cmim',      # Conditional Mutual Information Maximization
                'jmi',       # Joint Mutual Information
                'mim',       # Mutual Information Maximization
                'relief',    # Relief algorithm
                'disr',      # Double Input Symmetrical Relevance
                'anova',     # ANOVA F-statistic
                'variance'   # Variance threshold
            ]
        
        super().__init__(
            filters=filters,
            aggregation='mean_rank',
            normalize_scores=True,
            handle_missing='worst'
        )
        self.n_features = n_features
    
    def calculate(self, task: Task) -> FilterResult:
        """Calculate cumulative ranking."""
        # Get base ranking
        result = super().calculate(task)
        
        # If n_features specified, keep only top features
        if self.n_features is not None:
            result.features = result.features[:self.n_features]
        
        # Update method name
        result.method = "cumulative_ranking"
        
        return result
    
    def get_feature_importance_matrix(self, task: Task) -> pd.DataFrame:
        """
        Get detailed feature importance matrix showing scores from each filter.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with features as rows and filters as columns
        """
        result = self.calculate(task)
        
        # Extract individual scores from params
        scores_dict = result.params.get('individual_scores', {})
        ranks_dict = result.params.get('individual_ranks', {})
        
        # Create importance DataFrame
        importance_df = pd.DataFrame(scores_dict)
        importance_df['cumulative_rank'] = result.scores
        importance_df = importance_df.sort_values('cumulative_rank', ascending=False)
        
        return importance_df


def quick_feature_selection(
    task: Task,
    n_features: int = 30,
    methods: Optional[List[str]] = None,
    show_details: bool = False
) -> List[str]:
    """
    Quick feature selection using ensemble ranking.
    
    Convenience function for rapid feature selection using multiple methods.
    
    Parameters
    ----------
    task : Task
        The task with data
    n_features : int, default=30
        Number of features to select
    methods : List[str], optional
        Filter methods to use. If None, uses default set.
    show_details : bool, default=False
        Whether to print detailed results
        
    Returns
    -------
    List[str]
        Selected feature names
        
    Examples
    --------
    >>> from mlpy.filters import quick_feature_selection
    >>> selected = quick_feature_selection(task, n_features=20)
    >>> print(f"Selected {len(selected)} features: {selected[:5]}...")
    """
    # Create cumulative ranking filter
    cfs = CumulativeRanking(filters=methods, n_features=n_features)
    
    # Calculate ranking
    result = cfs.calculate(task)
    
    # Get selected features
    selected_features = result.select_top_k(n_features)
    
    if show_details:
        print(f"Feature Selection Results")
        print(f"=" * 50)
        print(f"Total features evaluated: {len(result.scores)}")
        print(f"Features selected: {n_features}")
        print(f"Methods used: {result.params['filters']}")
        print(f"\nTop 10 features:")
        for i, feat in enumerate(selected_features[:10], 1):
            score = result.scores[feat]
            print(f"  {i:2d}. {feat:20s} (score: {score:.4f})")
        
        # Show importance matrix if requested
        importance_df = cfs.get_feature_importance_matrix(task)
        print(f"\nFeature Importance Matrix (top 10):")
        print(importance_df.head(10))
    
    return selected_features