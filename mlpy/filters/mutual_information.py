"""
Mutual Information based feature selection methods for MLPY.

These methods use information theory to measure feature relevance
and redundancy, implementing state-of-the-art algorithms like
MRMR, CMIM, JMI, and others commonly used in machine learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score
import warnings

from .base import Filter, FilterResult
from ..tasks.base import Task
from ..tasks.supervised import TaskClassif, TaskRegr
from ..utils.registry import mlpy_filters


class MutualInformationFilter(Filter):
    """Base class for mutual information based filters."""
    
    def __init__(
        self,
        id: str,
        task_types: Optional[set] = None,
        n_neighbors: int = 3,
        random_state: Optional[int] = None
    ):
        super().__init__(
            id=id,
            task_types=task_types or {'classif', 'regr'},
            feature_types={'numeric'}
        )
        self.n_neighbors = n_neighbors
        self.random_state = random_state
    
    def _compute_mi(self, X: np.ndarray, y: np.ndarray, is_classification: bool) -> np.ndarray:
        """Compute mutual information between features and target."""
        if is_classification:
            return mutual_info_classif(
                X, y,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state
            )
        else:
            return mutual_info_regression(
                X, y,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state
            )
    
    def _compute_mi_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise mutual information between features."""
        n_features = X.shape[1]
        mi_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i, n_features):
                if i == j:
                    mi_matrix[i, j] = 1.0  # Perfect correlation with itself
                else:
                    # Use normalized MI for feature-feature relationships
                    mi = normalized_mutual_info_score(
                        self._discretize(X[:, i]),
                        self._discretize(X[:, j])
                    )
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi
        
        return mi_matrix
    
    def _discretize(self, x: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """Discretize continuous variable for MI calculation."""
        if len(np.unique(x)) <= n_bins:
            return x
        return pd.qcut(x, n_bins, labels=False, duplicates='drop')
    
    def _compute_conditional_mi(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_idx: int, 
        condition_idx: int
    ) -> float:
        """Compute conditional mutual information I(X_i; Y | X_j)."""
        # Simplified implementation using residuals
        from sklearn.linear_model import LinearRegression
        
        # Regress out the conditioning variable
        lr = LinearRegression()
        X_cond = X[:, condition_idx].reshape(-1, 1)
        
        # Residualize feature
        lr.fit(X_cond, X[:, feature_idx])
        X_resid = X[:, feature_idx] - lr.predict(X_cond)
        
        # Residualize target
        if y.dtype.kind in 'fi':  # Numeric target
            lr.fit(X_cond, y)
            y_resid = y - lr.predict(X_cond)
        else:  # Categorical target
            y_resid = y  # Keep as is for classification
        
        # Compute MI on residuals
        is_classif = y.dtype.kind not in 'fi'
        if is_classif:
            mi = mutual_info_classif(
                X_resid.reshape(-1, 1), y_resid,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state
            )[0]
        else:
            mi = mutual_info_regression(
                X_resid.reshape(-1, 1), y_resid,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state
            )[0]
        
        return mi


@mlpy_filters.register("mrmr")
class MRMR(MutualInformationFilter):
    """
    Minimum Redundancy Maximum Relevance (MRMR) feature selection.
    
    MRMR selects features that have high mutual information with the target
    (relevance) while having low mutual information with already selected
    features (redundancy).
    
    Parameters
    ----------
    n_features : int, optional
        Number of features to select
    relevance_weight : float, default=1.0
        Weight for relevance term
    redundancy_weight : float, default=1.0
        Weight for redundancy term
    n_neighbors : int, default=3
        Number of neighbors for MI estimation
    random_state : int, optional
        Random state for reproducibility
        
    References
    ----------
    Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual
    information criteria of max-dependency, max-relevance, and min-redundancy.
    IEEE Transactions on pattern analysis and machine intelligence, 27(8).
    
    Examples
    --------
    >>> from mlpy.filters import MRMR
    >>> filter = MRMR(n_features=10)
    >>> result = filter.calculate(task)
    >>> top_features = result.select_top_k(5)
    """
    
    def __init__(
        self,
        n_features: Optional[int] = None,
        relevance_weight: float = 1.0,
        redundancy_weight: float = 1.0,
        n_neighbors: int = 3,
        random_state: Optional[int] = None
    ):
        super().__init__(
            id="mrmr",
            n_neighbors=n_neighbors,
            random_state=random_state
        )
        self.n_features = n_features
        self.relevance_weight = relevance_weight
        self.redundancy_weight = redundancy_weight
    
    def calculate(self, task: Task) -> FilterResult:
        """Calculate MRMR scores for features."""
        self.check_task_compatibility(task)
        
        # Get data
        X = task.data(cols=task.feature_names, data_format="array")
        y = task.truth()
        feature_names = task.feature_names
        n_features = len(feature_names)
        
        # Determine if classification or regression
        is_classification = isinstance(task, TaskClassif)
        
        # Compute relevance (MI with target)
        relevance = self._compute_mi(X, y, is_classification)
        
        # Compute redundancy (MI between features)
        redundancy_matrix = self._compute_mi_matrix(X)
        
        # MRMR selection process
        selected_features = []
        selected_indices = []
        remaining_indices = list(range(n_features))
        
        # Select first feature (highest relevance)
        first_idx = np.argmax(relevance)
        selected_features.append(feature_names[first_idx])
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select remaining features
        n_to_select = self.n_features or n_features
        
        while len(selected_features) < n_to_select and remaining_indices:
            scores = []
            
            for idx in remaining_indices:
                # Relevance term
                rel_score = self.relevance_weight * relevance[idx]
                
                # Redundancy term (average MI with selected features)
                if selected_indices:
                    red_score = self.redundancy_weight * np.mean([
                        redundancy_matrix[idx, s_idx] for s_idx in selected_indices
                    ])
                else:
                    red_score = 0
                
                # MRMR score
                mrmr_score = rel_score - red_score
                scores.append(mrmr_score)
            
            # Select feature with highest MRMR score
            best_idx = remaining_indices[np.argmax(scores)]
            selected_features.append(feature_names[best_idx])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Create scores for all features
        final_scores = np.zeros(n_features)
        for i, idx in enumerate(selected_indices):
            final_scores[idx] = n_features - i  # Higher score for earlier selection
        
        # Create result
        scores_series = pd.Series(final_scores, index=feature_names)
        
        return FilterResult(
            scores=scores_series,
            features=selected_features,
            method="mrmr",
            task_type=task.task_type,
            params={
                'relevance_weight': self.relevance_weight,
                'redundancy_weight': self.redundancy_weight,
                'n_neighbors': self.n_neighbors
            }
        )


@mlpy_filters.register("cmim")
class CMIM(MutualInformationFilter):
    """
    Conditional Mutual Information Maximization (CMIM) feature selection.
    
    CMIM selects features by maximizing the conditional mutual information
    between features and the target given the already selected features.
    
    Parameters
    ----------
    n_features : int, optional
        Number of features to select
    n_neighbors : int, default=3
        Number of neighbors for MI estimation
    random_state : int, optional
        Random state for reproducibility
        
    References
    ----------
    Fleuret, F. (2004). Fast binary feature selection with conditional mutual
    information. Journal of Machine Learning Research, 5(Nov), 1531-1555.
    
    Examples
    --------
    >>> from mlpy.filters import CMIM
    >>> filter = CMIM(n_features=10)
    >>> result = filter.calculate(task)
    """
    
    def __init__(
        self,
        n_features: Optional[int] = None,
        n_neighbors: int = 3,
        random_state: Optional[int] = None
    ):
        super().__init__(
            id="cmim",
            n_neighbors=n_neighbors,
            random_state=random_state
        )
        self.n_features = n_features
    
    def calculate(self, task: Task) -> FilterResult:
        """Calculate CMIM scores for features."""
        self.check_task_compatibility(task)
        
        # Get data
        X = task.data(cols=task.feature_names, data_format="array")
        y = task.truth()
        feature_names = task.feature_names
        n_features = len(feature_names)
        
        # Determine task type
        is_classification = isinstance(task, TaskClassif)
        
        # Compute unconditional MI with target
        mi_scores = self._compute_mi(X, y, is_classification)
        
        # CMIM selection process
        selected_features = []
        selected_indices = []
        remaining_indices = list(range(n_features))
        
        # Select first feature (highest MI)
        first_idx = np.argmax(mi_scores)
        selected_features.append(feature_names[first_idx])
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Track minimum conditional MI for each feature
        min_cmi = mi_scores.copy()
        
        # Select remaining features
        n_to_select = self.n_features or n_features
        
        while len(selected_features) < n_to_select and remaining_indices:
            # For each remaining feature, compute conditional MI
            for idx in remaining_indices:
                for selected_idx in selected_indices:
                    # Compute I(X_i; Y | X_selected)
                    cmi = self._compute_conditional_mi(X, y, idx, selected_idx)
                    min_cmi[idx] = min(min_cmi[idx], cmi)
            
            # Select feature with maximum minimum CMI
            best_remaining_idx = np.argmax([min_cmi[i] for i in remaining_indices])
            best_idx = remaining_indices[best_remaining_idx]
            
            selected_features.append(feature_names[best_idx])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Create scores
        final_scores = np.zeros(n_features)
        for i, idx in enumerate(selected_indices):
            final_scores[idx] = n_features - i
        
        scores_series = pd.Series(final_scores, index=feature_names)
        
        return FilterResult(
            scores=scores_series,
            features=selected_features,
            method="cmim",
            task_type=task.task_type,
            params={'n_neighbors': self.n_neighbors}
        )


@mlpy_filters.register("jmi")
class JMI(MutualInformationFilter):
    """
    Joint Mutual Information (JMI) feature selection.
    
    JMI selects features by maximizing the joint mutual information
    between feature pairs and the target.
    
    Parameters
    ----------
    n_features : int, optional
        Number of features to select
    n_neighbors : int, default=3
        Number of neighbors for MI estimation
    random_state : int, optional
        Random state for reproducibility
        
    References
    ----------
    Yang, H., & Moody, J. (1999). Feature selection based on joint mutual
    information. In Proceedings of international ICSC symposium on advances
    in intelligent data analysis.
    """
    
    def __init__(
        self,
        n_features: Optional[int] = None,
        n_neighbors: int = 3,
        random_state: Optional[int] = None
    ):
        super().__init__(
            id="jmi",
            n_neighbors=n_neighbors,
            random_state=random_state
        )
        self.n_features = n_features
    
    def calculate(self, task: Task) -> FilterResult:
        """Calculate JMI scores for features."""
        self.check_task_compatibility(task)
        
        # Get data
        X = task.data(cols=task.feature_names, data_format="array")
        y = task.truth()
        feature_names = task.feature_names
        n_features = len(feature_names)
        
        # Determine task type
        is_classification = isinstance(task, TaskClassif)
        
        # Compute MI with target
        mi_scores = self._compute_mi(X, y, is_classification)
        
        # JMI selection process
        selected_features = []
        selected_indices = []
        remaining_indices = list(range(n_features))
        
        # Select first feature
        first_idx = np.argmax(mi_scores)
        selected_features.append(feature_names[first_idx])
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select remaining features
        n_to_select = self.n_features or n_features
        
        while len(selected_features) < n_to_select and remaining_indices:
            jmi_scores = []
            
            for idx in remaining_indices:
                # Compute joint MI with each selected feature
                if selected_indices:
                    joint_mi = 0
                    for selected_idx in selected_indices:
                        # Approximate joint MI as sum of individual MIs
                        # I(X_i, X_j; Y) ≈ I(X_i; Y) + I(X_j; Y | X_i)
                        joint_mi += mi_scores[idx] + self._compute_conditional_mi(
                            X, y, selected_idx, idx
                        )
                    jmi_score = joint_mi / len(selected_indices)
                else:
                    jmi_score = mi_scores[idx]
                
                jmi_scores.append(jmi_score)
            
            # Select feature with highest JMI
            best_idx = remaining_indices[np.argmax(jmi_scores)]
            selected_features.append(feature_names[best_idx])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Create scores
        final_scores = np.zeros(n_features)
        for i, idx in enumerate(selected_indices):
            final_scores[idx] = n_features - i
        
        scores_series = pd.Series(final_scores, index=feature_names)
        
        return FilterResult(
            scores=scores_series,
            features=selected_features,
            method="jmi",
            task_type=task.task_type,
            params={'n_neighbors': self.n_neighbors}
        )


@mlpy_filters.register("jmim")
class JMIM(MutualInformationFilter):
    """
    Joint Mutual Information Maximization (JMIM) feature selection.
    
    JMIM selects features by maximizing the minimum joint mutual information
    with previously selected features.
    
    Parameters
    ----------
    n_features : int, optional
        Number of features to select
    n_neighbors : int, default=3
        Number of neighbors for MI estimation
    random_state : int, optional
        Random state for reproducibility
        
    References
    ----------
    Bennasar, M., Hicks, Y., & Setchi, R. (2015). Feature selection using
    joint mutual information maximisation. Expert Systems with Applications.
    """
    
    def __init__(
        self,
        n_features: Optional[int] = None,
        n_neighbors: int = 3,
        random_state: Optional[int] = None
    ):
        super().__init__(
            id="jmim",
            n_neighbors=n_neighbors,
            random_state=random_state
        )
        self.n_features = n_features
    
    def calculate(self, task: Task) -> FilterResult:
        """Calculate JMIM scores for features."""
        self.check_task_compatibility(task)
        
        # Get data
        X = task.data(cols=task.feature_names, data_format="array")
        y = task.truth()
        feature_names = task.feature_names
        n_features = len(feature_names)
        
        # Determine task type
        is_classification = isinstance(task, TaskClassif)
        
        # Compute MI with target
        mi_scores = self._compute_mi(X, y, is_classification)
        
        # JMIM selection process
        selected_features = []
        selected_indices = []
        remaining_indices = list(range(n_features))
        
        # Select first feature
        first_idx = np.argmax(mi_scores)
        selected_features.append(feature_names[first_idx])
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Track minimum joint MI for each feature
        min_joint_mi = mi_scores.copy()
        
        # Select remaining features
        n_to_select = self.n_features or n_features
        
        while len(selected_features) < n_to_select and remaining_indices:
            for idx in remaining_indices:
                for selected_idx in selected_indices:
                    # Compute joint MI: I(X_i, X_j; Y)
                    # Approximation: use sum of individual and conditional MI
                    joint_mi = (mi_scores[idx] + mi_scores[selected_idx] + 
                               self._compute_conditional_mi(X, y, idx, selected_idx))
                    min_joint_mi[idx] = min(min_joint_mi[idx], joint_mi)
            
            # Select feature with maximum of minimum joint MI
            best_remaining_idx = np.argmax([min_joint_mi[i] for i in remaining_indices])
            best_idx = remaining_indices[best_remaining_idx]
            
            selected_features.append(feature_names[best_idx])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Create scores
        final_scores = np.zeros(n_features)
        for i, idx in enumerate(selected_indices):
            final_scores[idx] = n_features - i
        
        scores_series = pd.Series(final_scores, index=feature_names)
        
        return FilterResult(
            scores=scores_series,
            features=selected_features,
            method="jmim",
            task_type=task.task_type,
            params={'n_neighbors': self.n_neighbors}
        )


@mlpy_filters.register("mim")
class MIM(MutualInformationFilter):
    """
    Mutual Information Maximization (MIM) feature selection.
    
    Simple filter that ranks features by their mutual information with
    the target variable. This is the baseline MI method.
    
    Parameters
    ----------
    n_neighbors : int, default=3
        Number of neighbors for MI estimation
    random_state : int, optional
        Random state for reproducibility
        
    Examples
    --------
    >>> from mlpy.filters import MIM
    >>> filter = MIM()
    >>> result = filter.calculate(task)
    >>> top_features = result.select_top_k(10)
    """
    
    def __init__(
        self,
        n_neighbors: int = 3,
        random_state: Optional[int] = None
    ):
        super().__init__(
            id="mim",
            n_neighbors=n_neighbors,
            random_state=random_state
        )
    
    def calculate(self, task: Task) -> FilterResult:
        """Calculate MI scores for features."""
        self.check_task_compatibility(task)
        
        # Get data
        X = task.data(cols=task.feature_names, data_format="array")
        y = task.truth()
        feature_names = task.feature_names
        
        # Determine task type
        is_classification = isinstance(task, TaskClassif)
        
        # Compute MI with target
        mi_scores = self._compute_mi(X, y, is_classification)
        
        # Create scores series
        scores_series = pd.Series(mi_scores, index=feature_names)
        
        # Sort features by score
        sorted_features = scores_series.sort_values(ascending=False).index.tolist()
        
        return FilterResult(
            scores=scores_series,
            features=sorted_features,
            method="mim",
            task_type=task.task_type,
            params={'n_neighbors': self.n_neighbors}
        )


@mlpy_filters.register("information_gain")
class InformationGain(MutualInformationFilter):
    """
    Information Gain feature selection (alias for MIM).
    
    Information gain is equivalent to mutual information for discrete variables.
    This is an alias for the MIM filter for compatibility.
    
    Parameters
    ----------
    n_neighbors : int, default=3
        Number of neighbors for MI estimation
    random_state : int, optional
        Random state for reproducibility
    """
    
    def __init__(
        self,
        n_neighbors: int = 3,
        random_state: Optional[int] = None
    ):
        super().__init__(
            id="information_gain",
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        self._mim = MIM(n_neighbors=n_neighbors, random_state=random_state)
    
    def calculate(self, task: Task) -> FilterResult:
        """Calculate information gain scores."""
        result = self._mim.calculate(task)
        result.method = "information_gain"
        return result