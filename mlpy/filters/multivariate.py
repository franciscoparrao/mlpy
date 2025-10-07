"""
Multivariate feature filters for MLPY.

These filters consider feature interactions and dependencies
when scoring features.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
import warnings

from .base import Filter, FilterResult, filter_registry
from ..tasks.base import Task
from ..tasks.supervised import TaskClassif
from ..tasks.supervised import TaskRegr


class FilterImportance(Filter):
    """Feature importance filter using tree-based models.
    
    Uses Random Forest to compute feature importances.
    Works with both classification and regression tasks.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees.
    random_state : int, optional
        Random state for reproducibility.
    importance_type : str, default='gini'
        Type of importance: 'gini' or 'permutation'.
    """
    
    def __init__(
        self, 
        n_estimators: int = 100,
        random_state: Optional[int] = None,
        importance_type: str = 'gini'
    ):
        super().__init__(
            id="importance",
            task_types={'classif', 'regr'},
            feature_types={'numeric'}
        )
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.importance_type = importance_type
        
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        self.check_task_compatibility(task)
        
        if features is None:
            features = task.feature_names
            
        # Get data
        X = task.data(cols=features, data_format='array')
        y = task.truth()
        
        # Choose model based on task type
        if isinstance(task, TaskClassif):
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
            task_type = "classif"
        else:
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
            task_type = "regr"
            
        # Fit model
        model.fit(X, y)
        
        # Get importances
        if self.importance_type == 'gini':
            importances = model.feature_importances_
        elif self.importance_type == 'permutation':
            from sklearn.inspection import permutation_importance
            result = permutation_importance(
                model, X, y, 
                n_repeats=10,
                random_state=self.random_state,
                n_jobs=-1
            )
            importances = result.importances_mean
        else:
            raise ValueError(f"Unknown importance type: {self.importance_type}")
            
        # Create result
        scores = pd.Series(importances, index=features)
        sorted_features = scores.sort_values(ascending=False).index.tolist()
        
        return FilterResult(
            scores=scores,
            features=sorted_features,
            method=f"importance_{self.importance_type}",
            task_type=task_type,
            params={
                "n_estimators": self.n_estimators,
                "importance_type": self.importance_type
            }
        )


class FilterRFE(Filter):
    """Recursive Feature Elimination filter.
    
    Uses RFE to rank features by recursively removing features
    and building a model on remaining features.
    
    Parameters
    ----------
    estimator : object, optional
        Estimator to use. If None, uses default based on task.
    n_features_to_select : int or float, optional
        Number of features to select. If float, interpreted as fraction.
    step : int or float, default=1
        Number of features to remove at each iteration.
    """
    
    def __init__(
        self,
        estimator=None,
        n_features_to_select=None,
        step=1
    ):
        super().__init__(
            id="rfe",
            task_types={'classif', 'regr'},
            feature_types={'numeric'}
        )
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        self.check_task_compatibility(task)
        
        if features is None:
            features = task.feature_names
            
        # Get data
        X = task.data(cols=features, data_format='array')
        y = task.truth()
        
        # Choose estimator if not provided
        if self.estimator is None:
            if isinstance(task, TaskClassif):
                estimator = LogisticRegression(max_iter=1000, random_state=42)
                task_type = "classif"
            else:
                estimator = Ridge(random_state=42)
                task_type = "regr"
        else:
            estimator = self.estimator
            task_type = "classif" if isinstance(task, TaskClassif) else "regr"
            
        # Determine number of features
        if self.n_features_to_select is None:
            n_features = max(1, len(features) // 2)
        elif isinstance(self.n_features_to_select, float):
            n_features = max(1, int(len(features) * self.n_features_to_select))
        else:
            n_features = self.n_features_to_select
            
        # Run RFE
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=n_features,
            step=self.step
        )
        rfe.fit(X, y)
        
        # Create scores (inverse of ranking)
        scores = pd.Series(
            1.0 / rfe.ranking_,
            index=features
        )
        sorted_features = scores.sort_values(ascending=False).index.tolist()
        
        return FilterResult(
            scores=scores,
            features=sorted_features,
            method="rfe",
            task_type=task_type,
            params={
                "n_features_to_select": n_features,
                "step": self.step,
                "ranking": pd.Series(rfe.ranking_, index=features)
            }
        )


class FilterMRMR(Filter):
    """Maximum Relevance Minimum Redundancy (mRMR) filter.
    
    Selects features that have high correlation with target
    but low correlation with each other.
    
    Parameters
    ----------
    n_features : int, default=10
        Number of features to select.
    relevance_func : str, default='f_classif'
        Function to compute relevance: 'f_classif', 'mutual_info'.
    redundancy_func : str, default='pearson'
        Function to compute redundancy: 'pearson', 'spearman'.
    """
    
    def __init__(
        self,
        n_features: int = 10,
        relevance_func: str = 'f_classif',
        redundancy_func: str = 'pearson'
    ):
        super().__init__(
            id="mrmr",
            task_types={'classif', 'regr'},
            feature_types={'numeric'}
        )
        self.n_features = n_features
        self.relevance_func = relevance_func
        self.redundancy_func = redundancy_func
        
    def _compute_relevance(self, X: np.ndarray, y: np.ndarray, task_type: str) -> np.ndarray:
        """Compute relevance scores."""
        if self.relevance_func == 'f_classif' and task_type == 'classif':
            scores, _ = f_classif(X, y)
        elif self.relevance_func == 'f_regression' and task_type == 'regr':
            scores, _ = f_regression(X, y)
        elif self.relevance_func == 'mutual_info':
            if task_type == 'classif':
                scores = mutual_info_classif(X, y, random_state=42)
            else:
                scores = mutual_info_regression(X, y, random_state=42)
        else:
            raise ValueError(f"Invalid relevance function: {self.relevance_func}")
            
        return np.nan_to_num(scores, nan=0.0)
        
    def _compute_redundancy(self, X: np.ndarray, selected_idx: int, selected_indices: List[int]) -> float:
        """Compute redundancy between a feature and selected features."""
        if not selected_indices:
            return 0.0
            
        if self.redundancy_func == 'pearson':
            correlations = [
                abs(np.corrcoef(X[:, selected_idx], X[:, idx])[0, 1])
                for idx in selected_indices
            ]
        elif self.redundancy_func == 'spearman':
            from scipy.stats import spearmanr
            correlations = [
                abs(spearmanr(X[:, selected_idx], X[:, idx])[0])
                for idx in selected_indices
            ]
        else:
            raise ValueError(f"Invalid redundancy function: {self.redundancy_func}")
            
        # Handle NaN values
        correlations = [c for c in correlations if not np.isnan(c)]
        return np.mean(correlations) if correlations else 0.0
        
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        self.check_task_compatibility(task)
        
        if features is None:
            features = task.feature_names
            
        # Get data
        X = task.data(cols=features, data_format='array')
        y = task.truth()
        
        task_type = "classif" if isinstance(task, TaskClassif) else "regr"
        
        # Compute relevance scores
        relevance_scores = self._compute_relevance(X, y, task_type)
        
        # mRMR algorithm
        n_features = min(self.n_features, len(features))
        selected_indices = []
        selected_features = []
        scores = np.zeros(len(features))
        
        # First feature: highest relevance
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        selected_features.append(features[first_idx])
        scores[first_idx] = relevance_scores[first_idx]
        
        # Remaining features
        for _ in range(1, n_features):
            mrmr_scores = []
            candidates = []
            
            for idx in range(len(features)):
                if idx not in selected_indices:
                    relevance = relevance_scores[idx]
                    redundancy = self._compute_redundancy(X, idx, selected_indices)
                    mrmr_score = relevance - redundancy
                    mrmr_scores.append(mrmr_score)
                    candidates.append(idx)
                    
            if not mrmr_scores:
                break
                
            # Select best feature
            best_idx = candidates[np.argmax(mrmr_scores)]
            selected_indices.append(best_idx)
            selected_features.append(features[best_idx])
            scores[best_idx] = max(mrmr_scores)
            
        # Create full ranking
        remaining_indices = [i for i in range(len(features)) if i not in selected_indices]
        remaining_scores = relevance_scores[remaining_indices]
        sorted_remaining = np.argsort(remaining_scores)[::-1]
        
        for i, idx in enumerate(sorted_remaining):
            actual_idx = remaining_indices[idx]
            selected_features.append(features[actual_idx])
            scores[actual_idx] = remaining_scores[idx] * 0.1  # Lower score for non-selected
            
        # Create result
        scores_series = pd.Series(scores, index=features)
        
        return FilterResult(
            scores=scores_series,
            features=selected_features,
            method="mrmr",
            task_type=task_type,
            params={
                "n_features": n_features,
                "relevance_func": self.relevance_func,
                "redundancy_func": self.redundancy_func
            }
        )


class FilterRelief(Filter):
    """Relief-based feature filter.
    
    Uses ReliefF algorithm to score features based on how well
    they distinguish between instances that are near to each other.
    
    Parameters
    ----------
    n_neighbors : int, default=10
        Number of neighbors to consider.
    sample_size : float, default=1.0
        Fraction of samples to use.
    """
    
    def __init__(self, n_neighbors: int = 10, sample_size: float = 1.0):
        super().__init__(
            id="relief",
            task_types={'classif', 'regr'},
            feature_types={'numeric'}
        )
        self.n_neighbors = n_neighbors
        self.sample_size = sample_size
        
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        self.check_task_compatibility(task)
        
        if features is None:
            features = task.feature_names
            
        # Get data
        X = task.data(cols=features, data_format='array')
        y = task.truth()
        
        # Sample if needed
        n_samples = int(len(X) * self.sample_size)
        if n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
            
        # Simple Relief implementation
        n_features = X.shape[1]
        scores = np.zeros(n_features)
        
        # Normalize features
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        
        for i in range(len(X_norm)):
            # Find k nearest neighbors
            distances = np.sum((X_norm - X_norm[i]) ** 2, axis=1)
            distances[i] = np.inf  # Exclude self
            
            # For classification: find nearest hit and miss
            if isinstance(task, TaskClassif):
                same_class = y == y[i]
                diff_class = ~same_class
                
                # Nearest hits
                hit_indices = np.where(same_class)[0]
                if len(hit_indices) > 0:
                    hit_distances = distances[hit_indices]
                    nearest_hits = hit_indices[np.argsort(hit_distances)[:self.n_neighbors]]
                    
                    # Nearest misses
                    miss_indices = np.where(diff_class)[0]
                    if len(miss_indices) > 0:
                        miss_distances = distances[miss_indices]
                        nearest_misses = miss_indices[np.argsort(miss_distances)[:self.n_neighbors]]
                        
                        # Update scores
                        for j in range(n_features):
                            diff_hits = np.mean(np.abs(X_norm[i, j] - X_norm[nearest_hits, j]))
                            diff_misses = np.mean(np.abs(X_norm[i, j] - X_norm[nearest_misses, j]))
                            scores[j] += diff_misses - diff_hits
            else:
                # For regression: weight by difference in target
                nearest_indices = np.argsort(distances)[:self.n_neighbors]
                
                for j in range(n_features):
                    feature_diffs = np.abs(X_norm[i, j] - X_norm[nearest_indices, j])
                    target_diffs = np.abs(y[i] - y[nearest_indices])
                    scores[j] -= np.mean(feature_diffs * target_diffs)
                    
        # Normalize scores
        scores = scores / len(X_norm)
        scores = scores - scores.min()  # Make non-negative
        
        task_type = "classif" if isinstance(task, TaskClassif) else "regr"
        
        # Create result
        scores_series = pd.Series(scores, index=features)
        sorted_features = scores_series.sort_values(ascending=False).index.tolist()
        
        return FilterResult(
            scores=scores_series,
            features=sorted_features,
            method="relief",
            task_type=task_type,
            params={
                "n_neighbors": self.n_neighbors,
                "sample_size": self.sample_size
            }
        )


# Register filters
filter_registry.register(FilterImportance, "importance")
filter_registry.register(FilterRFE, "rfe")
filter_registry.register(FilterMRMR, "mrmr")
filter_registry.register(FilterRelief, "relief")