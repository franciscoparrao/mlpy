"""
Statistical and distance-based feature selection methods for MLPY.

Includes Relief family algorithms and other statistical measures
for feature relevance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import warnings

from .base import Filter, FilterResult
from ..tasks.base import Task
from ..tasks.supervised import TaskClassif, TaskRegr
from ..utils.registry import mlpy_filters


@mlpy_filters.register("relief")
class Relief(Filter):
    """
    Relief feature selection algorithm.
    
    Relief estimates feature quality by how well features distinguish
    between instances that are near to each other. For each instance,
    it finds nearest hit (same class) and miss (different class) and
    updates feature weights based on their differences.
    
    Parameters
    ----------
    n_neighbors : int, default=10
        Number of neighbors to consider
    n_samples : int or float, optional
        Number of samples to use. If float, interpreted as fraction.
        If None, use all samples.
    random_state : int, optional
        Random state for reproducibility
        
    References
    ----------
    Kira, K., & Rendell, L. A. (1992). A practical approach to feature
    selection. In Machine learning proceedings 1992 (pp. 249-256).
    
    Examples
    --------
    >>> from mlpy.filters import Relief
    >>> filter = Relief(n_neighbors=10)
    >>> result = filter.calculate(task)
    >>> top_features = result.select_top_k(10)
    """
    
    def __init__(
        self,
        n_neighbors: int = 10,
        n_samples: Optional[Union[int, float]] = None,
        random_state: Optional[int] = None
    ):
        super().__init__(
            id="relief",
            task_types={'classif'},
            feature_types={'numeric'}
        )
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        self.random_state = random_state
    
    def calculate(self, task: Task) -> FilterResult:
        """Calculate Relief scores for features."""
        self.check_task_compatibility(task)
        
        if not isinstance(task, TaskClassif):
            raise ValueError("Relief only supports classification tasks")
        
        # Get data
        X = task.data(cols=task.feature_names, data_format="array")
        y = task.truth()
        feature_names = task.feature_names
        n_features = X.shape[1]
        n_instances = X.shape[0]
        
        # Normalize features to [0, 1]
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # Avoid division by zero
        X_norm = (X - X_min) / X_range
        
        # Determine samples to use
        if self.n_samples is None:
            sample_indices = np.arange(n_instances)
        elif isinstance(self.n_samples, float):
            n_samples_use = int(n_instances * self.n_samples)
            np.random.seed(self.random_state)
            sample_indices = np.random.choice(n_instances, n_samples_use, replace=False)
        else:
            n_samples_use = min(self.n_samples, n_instances)
            np.random.seed(self.random_state)
            sample_indices = np.random.choice(n_instances, n_samples_use, replace=False)
        
        # Initialize weights
        weights = np.zeros(n_features)
        
        # Get unique classes
        classes = np.unique(y)
        n_classes = len(classes)
        
        # For each sampled instance
        for idx in sample_indices:
            instance = X_norm[idx]
            instance_class = y[idx]
            
            # Find nearest neighbors
            distances = np.sum((X_norm - instance) ** 2, axis=1)
            distances[idx] = np.inf  # Exclude self
            
            # Find nearest hit (same class)
            same_class_mask = y == instance_class
            same_class_mask[idx] = False
            if np.any(same_class_mask):
                same_class_distances = distances.copy()
                same_class_distances[~same_class_mask] = np.inf
                nearest_hits_idx = np.argpartition(
                    same_class_distances, 
                    min(self.n_neighbors, np.sum(same_class_mask) - 1)
                )[:self.n_neighbors]
                nearest_hits = X_norm[nearest_hits_idx]
            else:
                nearest_hits = instance.reshape(1, -1)
            
            # Find nearest miss (different class)
            diff_class_mask = y != instance_class
            if np.any(diff_class_mask):
                diff_class_distances = distances.copy()
                diff_class_distances[~diff_class_mask] = np.inf
                nearest_misses_idx = np.argpartition(
                    diff_class_distances,
                    min(self.n_neighbors, np.sum(diff_class_mask) - 1)
                )[:self.n_neighbors]
                nearest_misses = X_norm[nearest_misses_idx]
            else:
                nearest_misses = instance.reshape(1, -1)
            
            # Update weights
            for f in range(n_features):
                # Difference with nearest hits
                diff_hits = np.mean(np.abs(instance[f] - nearest_hits[:, f]))
                
                # Difference with nearest misses
                diff_misses = np.mean(np.abs(instance[f] - nearest_misses[:, f]))
                
                # Update weight
                weights[f] += (diff_misses - diff_hits) / len(sample_indices)
        
        # Create result
        scores_series = pd.Series(weights, index=feature_names)
        sorted_features = scores_series.sort_values(ascending=False).index.tolist()
        
        return FilterResult(
            scores=scores_series,
            features=sorted_features,
            method="relief",
            task_type=task.task_type,
            params={
                'n_neighbors': self.n_neighbors,
                'n_samples': self.n_samples
            }
        )


@mlpy_filters.register("relieff")
class ReliefF(Relief):
    """
    ReliefF - Extension of Relief for multi-class problems.
    
    ReliefF extends Relief to handle multi-class problems and
    incomplete data more robustly. It considers multiple nearest
    neighbors and weights contributions by class probabilities.
    
    Parameters
    ----------
    n_neighbors : int, default=10
        Number of neighbors to consider
    n_samples : int or float, optional
        Number of samples to use
    weight_by_distance : bool, default=True
        Weight neighbor contributions by distance
    random_state : int, optional
        Random state for reproducibility
        
    References
    ----------
    Kononenko, I. (1994). Estimating attributes: Analysis and extensions
    of RELIEF. In European conference on machine learning (pp. 171-182).
    """
    
    def __init__(
        self,
        n_neighbors: int = 10,
        n_samples: Optional[Union[int, float]] = None,
        weight_by_distance: bool = True,
        random_state: Optional[int] = None
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            n_samples=n_samples,
            random_state=random_state
        )
        self.id = "relieff"
        self.weight_by_distance = weight_by_distance
    
    def calculate(self, task: Task) -> FilterResult:
        """Calculate ReliefF scores for features."""
        self.check_task_compatibility(task)
        
        if not isinstance(task, TaskClassif):
            raise ValueError("ReliefF only supports classification tasks")
        
        # Get data
        X = task.data(cols=task.feature_names, data_format="array")
        y = task.truth()
        feature_names = task.feature_names
        n_features = X.shape[1]
        n_instances = X.shape[0]
        
        # Normalize features
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1
        X_norm = (X - X_min) / X_range
        
        # Determine samples to use
        if self.n_samples is None:
            sample_indices = np.arange(n_instances)
        elif isinstance(self.n_samples, float):
            n_samples_use = int(n_instances * self.n_samples)
            np.random.seed(self.random_state)
            sample_indices = np.random.choice(n_instances, n_samples_use, replace=False)
        else:
            n_samples_use = min(self.n_samples, n_instances)
            np.random.seed(self.random_state)
            sample_indices = np.random.choice(n_instances, n_samples_use, replace=False)
        
        # Initialize weights
        weights = np.zeros(n_features)
        
        # Get class probabilities
        classes, class_counts = np.unique(y, return_counts=True)
        class_probs = class_counts / n_instances
        n_classes = len(classes)
        
        # Build nearest neighbor model
        nn = NearestNeighbors(n_neighbors=min(self.n_neighbors + 1, n_instances))
        nn.fit(X_norm)
        
        # For each sampled instance
        for idx in sample_indices:
            instance = X_norm[idx]
            instance_class = y[idx]
            
            # Find k nearest neighbors
            distances, neighbors = nn.kneighbors(instance.reshape(1, -1))
            distances = distances[0][1:]  # Exclude self
            neighbors = neighbors[0][1:]  # Exclude self
            
            # Weight by distance if requested
            if self.weight_by_distance and len(distances) > 0:
                distance_weights = 1.0 / (distances + 1e-10)
                distance_weights /= distance_weights.sum()
            else:
                distance_weights = np.ones(len(neighbors)) / len(neighbors)
            
            # For each feature
            for f in range(n_features):
                diff_same = 0
                diff_other = 0
                
                # For each neighbor
                for n_idx, n_weight in zip(neighbors, distance_weights):
                    neighbor_class = y[n_idx]
                    diff = np.abs(instance[f] - X_norm[n_idx, f])
                    
                    if neighbor_class == instance_class:
                        diff_same += n_weight * diff
                    else:
                        # Weight by class probability for multi-class
                        class_idx = np.where(classes == neighbor_class)[0][0]
                        diff_other += n_weight * diff * class_probs[class_idx]
                
                # Update weight
                weights[f] += (diff_other - diff_same) / len(sample_indices)
        
        # Create result
        scores_series = pd.Series(weights, index=feature_names)
        sorted_features = scores_series.sort_values(ascending=False).index.tolist()
        
        return FilterResult(
            scores=scores_series,
            features=sorted_features,
            method="relieff",
            task_type=task.task_type,
            params={
                'n_neighbors': self.n_neighbors,
                'n_samples': self.n_samples,
                'weight_by_distance': self.weight_by_distance
            }
        )


@mlpy_filters.register("disr")
class DISR(Filter):
    """
    Double Input Symmetrical Relevance (DISR) feature selection.
    
    DISR uses normalized mutual information to measure both feature
    relevance (with target) and redundancy (between features).
    
    Parameters
    ----------
    threshold : float, default=0.1
        Threshold for feature selection
    normalize : bool, default=True
        Whether to normalize scores
        
    References
    ----------
    Meyer, P. E., & Bontempi, G. (2006). On the use of variable
    complementarity for feature selection in cancer classification.
    In Workshops on applications of evolutionary computation.
    
    Examples
    --------
    >>> from mlpy.filters import DISR
    >>> filter = DISR(threshold=0.1)
    >>> result = filter.calculate(task)
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        normalize: bool = True
    ):
        super().__init__(
            id="disr",
            task_types={'classif', 'regr'},
            feature_types={'numeric'}
        )
        self.threshold = threshold
        self.normalize = normalize
    
    def calculate(self, task: Task) -> FilterResult:
        """Calculate DISR scores for features."""
        self.check_task_compatibility(task)
        
        # Get data
        X = task.data(cols=task.feature_names, data_format="array")
        y = task.truth()
        feature_names = task.feature_names
        n_features = len(feature_names)
        
        # Import MI functions
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        from sklearn.metrics import normalized_mutual_info_score
        
        # Determine task type
        is_classification = isinstance(task, TaskClassif)
        
        # Compute relevance (MI with target)
        if is_classification:
            relevance = mutual_info_classif(X, y, random_state=42)
        else:
            relevance = mutual_info_regression(X, y, random_state=42)
        
        # Normalize relevance if requested
        if self.normalize:
            max_rel = relevance.max()
            if max_rel > 0:
                relevance = relevance / max_rel
        
        # Compute redundancy matrix (MI between features)
        redundancy_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Discretize continuous features for NMI
                x_i = pd.qcut(X[:, i], 10, labels=False, duplicates='drop')
                x_j = pd.qcut(X[:, j], 10, labels=False, duplicates='drop')
                
                mi = normalized_mutual_info_score(x_i, x_j)
                redundancy_matrix[i, j] = mi
                redundancy_matrix[j, i] = mi
        
        # Compute DISR scores
        disr_scores = np.zeros(n_features)
        
        for i in range(n_features):
            # Relevance term
            rel = relevance[i]
            
            # Average redundancy with other features
            red = np.mean([redundancy_matrix[i, j] for j in range(n_features) if j != i])
            
            # DISR score: relevance minus redundancy
            disr_scores[i] = rel - red
        
        # Apply threshold
        disr_scores[disr_scores < self.threshold] = 0
        
        # Create result
        scores_series = pd.Series(disr_scores, index=feature_names)
        sorted_features = scores_series.sort_values(ascending=False).index.tolist()
        
        # Filter by threshold
        selected_features = [f for f in sorted_features if scores_series[f] >= self.threshold]
        
        return FilterResult(
            scores=scores_series,
            features=selected_features if selected_features else sorted_features,
            method="disr",
            task_type=task.task_type,
            params={
                'threshold': self.threshold,
                'normalize': self.normalize
            }
        )


@mlpy_filters.register("anova")
class ANOVA(Filter):
    """
    ANOVA F-statistic feature selection.
    
    Computes ANOVA F-value for each feature to measure the linear
    dependency between feature and target.
    
    Parameters
    ----------
    use_fdr : bool, default=False
        Use False Discovery Rate correction
    alpha : float, default=0.05
        Significance level
        
    Examples
    --------
    >>> from mlpy.filters import ANOVA
    >>> filter = ANOVA(alpha=0.05)
    >>> result = filter.calculate(task)
    """
    
    def __init__(
        self,
        use_fdr: bool = False,
        alpha: float = 0.05
    ):
        super().__init__(
            id="anova",
            task_types={'classif'},
            feature_types={'numeric'}
        )
        self.use_fdr = use_fdr
        self.alpha = alpha
    
    def calculate(self, task: Task) -> FilterResult:
        """Calculate ANOVA F-scores for features."""
        self.check_task_compatibility(task)
        
        if not isinstance(task, TaskClassif):
            raise ValueError("ANOVA only supports classification tasks")
        
        # Get data
        X = task.data(cols=task.feature_names, data_format="array")
        y = task.truth()
        feature_names = task.feature_names
        n_features = len(feature_names)
        
        # Get unique classes
        classes = np.unique(y)
        n_classes = len(classes)
        
        if n_classes < 2:
            raise ValueError("ANOVA requires at least 2 classes")
        
        # Compute F-statistic for each feature
        f_scores = np.zeros(n_features)
        p_values = np.zeros(n_features)
        
        for i in range(n_features):
            # Get feature values for each class
            groups = [X[y == c, i] for c in classes]
            
            # Compute F-statistic
            f_stat, p_val = stats.f_oneway(*groups)
            f_scores[i] = f_stat
            p_values[i] = p_val
        
        # Apply FDR correction if requested
        if self.use_fdr:
            from statsmodels.stats.multitest import fdrcorrection
            _, p_values = fdrcorrection(p_values, alpha=self.alpha)
        
        # Select features based on p-value
        selected_mask = p_values < self.alpha
        
        # Create result
        scores_series = pd.Series(f_scores, index=feature_names)
        sorted_features = scores_series.sort_values(ascending=False).index.tolist()
        
        # Filter by significance
        selected_features = [f for f in sorted_features if selected_mask[feature_names.index(f)]]
        
        return FilterResult(
            scores=scores_series,
            features=selected_features if selected_features else sorted_features,
            method="anova",
            task_type=task.task_type,
            params={
                'use_fdr': self.use_fdr,
                'alpha': self.alpha,
                'p_values': dict(zip(feature_names, p_values))
            }
        )


@mlpy_filters.register("variance")
class VarianceThreshold(Filter):
    """
    Variance threshold feature selection.
    
    Removes features with variance below a threshold. This is useful
    for removing constant or quasi-constant features.
    
    Parameters
    ----------
    threshold : float, default=0.0
        Variance threshold
    normalize : bool, default=False
        Whether to normalize by mean (coefficient of variation)
        
    Examples
    --------
    >>> from mlpy.filters import VarianceThreshold
    >>> filter = VarianceThreshold(threshold=0.01)
    >>> result = filter.calculate(task)
    """
    
    def __init__(
        self,
        threshold: float = 0.0,
        normalize: bool = False
    ):
        super().__init__(
            id="variance",
            task_types={'any'},
            feature_types={'numeric'}
        )
        self.threshold = threshold
        self.normalize = normalize
    
    def calculate(self, task: Task) -> FilterResult:
        """Calculate variance scores for features."""
        self.check_task_compatibility(task)
        
        # Get data
        X = task.data(cols=task.feature_names, data_format="array")
        feature_names = task.feature_names
        
        # Compute variance
        variances = np.var(X, axis=0)
        
        # Normalize by mean if requested (coefficient of variation)
        if self.normalize:
            means = np.mean(X, axis=0)
            # Avoid division by zero
            means[means == 0] = 1
            variances = variances / np.abs(means)
        
        # Create result
        scores_series = pd.Series(variances, index=feature_names)
        sorted_features = scores_series.sort_values(ascending=False).index.tolist()
        
        # Filter by threshold
        selected_features = [f for f in sorted_features if scores_series[f] >= self.threshold]
        
        return FilterResult(
            scores=scores_series,
            features=selected_features if selected_features else sorted_features,
            method="variance",
            task_type=task.task_type,
            params={
                'threshold': self.threshold,
                'normalize': self.normalize
            }
        )