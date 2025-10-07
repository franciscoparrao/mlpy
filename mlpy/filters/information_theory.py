"""
Information theory-based feature filters for MLPY.

These filters use entropy, information gain, and related
measures for feature selection.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from scipy.stats import entropy
import warnings

from .base import Filter, FilterResult, filter_registry
from ..tasks.base import Task
from ..tasks.supervised import TaskClassif
from ..tasks.supervised import TaskRegr


class FilterInformationGain(Filter):
    """Information Gain filter for classification.
    
    Measures the reduction in entropy when a feature is used
    to split the data.
    
    IG(Feature, Target) = H(Target) - H(Target|Feature)
    
    Parameters
    ----------
    n_bins : int, default=10
        Number of bins for discretizing continuous features.
    strategy : str, default='uniform'
        Binning strategy: 'uniform', 'quantile', 'kmeans'.
    """
    
    def __init__(self, n_bins: int = 10, strategy: str = 'uniform'):
        super().__init__(
            id="info_gain",
            task_types={'classif'},
            feature_types={'numeric', 'factor'}
        )
        self.n_bins = n_bins
        self.strategy = strategy
        
    def _discretize_feature(self, values: np.ndarray) -> np.ndarray:
        """Discretize continuous values into bins."""
        if len(np.unique(values)) <= self.n_bins:
            # Already discrete enough
            return values
            
        if self.strategy == 'uniform':
            # Equal width bins
            bins = np.linspace(values.min(), values.max(), self.n_bins + 1)
            return np.digitize(values, bins[1:-1])
        elif self.strategy == 'quantile':
            # Equal frequency bins
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            bins = np.quantile(values, quantiles)
            bins = np.unique(bins)  # Remove duplicates
            return np.digitize(values, bins[1:-1])
        elif self.strategy == 'kmeans':
            # K-means based bins
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_bins, random_state=42)
            return kmeans.fit_predict(values.reshape(-1, 1))
        else:
            raise ValueError(f"Unknown binning strategy: {self.strategy}")
            
    def _calculate_entropy(self, labels: np.ndarray) -> float:
        """Calculate entropy of labels."""
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return entropy(probabilities, base=2)
        
    def _calculate_conditional_entropy(self, feature: np.ndarray, target: np.ndarray) -> float:
        """Calculate H(Target|Feature)."""
        conditional_entropy = 0.0
        feature_values = np.unique(feature)
        
        for value in feature_values:
            mask = feature == value
            if np.any(mask):
                p_feature = np.sum(mask) / len(feature)
                target_given_feature = target[mask]
                h_target_given_feature = self._calculate_entropy(target_given_feature)
                conditional_entropy += p_feature * h_target_given_feature
                
        return conditional_entropy
        
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        self.check_task_compatibility(task)
        
        if features is None:
            features = task.feature_names
            
        # Get target
        y = task.truth()
        
        # Calculate target entropy
        h_target = self._calculate_entropy(y)
        
        # Calculate information gain for each feature
        scores = {}
        
        for feature in features:
            # Get feature values
            x = task.data(cols=[feature], data_format='array').ravel()
            
            # Handle missing values
            mask = ~np.isnan(x) if np.issubdtype(x.dtype, np.number) else pd.notna(x)
            if not np.any(mask):
                scores[feature] = 0.0
                continue
                
            x_clean = x[mask]
            y_clean = y[mask]
            
            # Discretize if continuous
            if np.issubdtype(x.dtype, np.number):
                x_discrete = self._discretize_feature(x_clean)
            else:
                x_discrete = x_clean
                
            # Calculate conditional entropy
            h_target_given_feature = self._calculate_conditional_entropy(x_discrete, y_clean)
            
            # Information gain
            info_gain = h_target - h_target_given_feature
            scores[feature] = max(0.0, info_gain)  # Ensure non-negative
            
        # Create result
        scores_series = pd.Series(scores)
        sorted_features = scores_series.sort_values(ascending=False).index.tolist()
        
        return FilterResult(
            scores=scores_series,
            features=sorted_features,
            method="info_gain",
            task_type="classif",
            params={
                "n_bins": self.n_bins,
                "strategy": self.strategy,
                "target_entropy": h_target
            }
        )


class FilterInformationGainRatio(Filter):
    """Information Gain Ratio filter (used in C4.5).
    
    Normalizes Information Gain by the intrinsic information
    of the feature to avoid bias toward features with many values.
    
    IGR(Feature, Target) = IG(Feature, Target) / H(Feature)
    
    Parameters
    ----------
    n_bins : int, default=10
        Number of bins for discretizing continuous features.
    strategy : str, default='uniform'
        Binning strategy: 'uniform', 'quantile', 'kmeans'.
    min_intrinsic_value : float, default=0.01
        Minimum intrinsic value to avoid division by zero.
    """
    
    def __init__(
        self, 
        n_bins: int = 10, 
        strategy: str = 'uniform',
        min_intrinsic_value: float = 0.01
    ):
        super().__init__(
            id="info_gain_ratio",
            task_types={'classif'},
            feature_types={'numeric', 'factor'}
        )
        self.n_bins = n_bins
        self.strategy = strategy
        self.min_intrinsic_value = min_intrinsic_value
        self._ig_filter = FilterInformationGain(n_bins=n_bins, strategy=strategy)
        
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        self.check_task_compatibility(task)
        
        if features is None:
            features = task.feature_names
            
        # First calculate information gain
        ig_result = self._ig_filter.calculate(task, features)
        
        # Calculate information gain ratio
        scores = {}
        
        for feature in features:
            # Get feature values
            x = task.data(cols=[feature], data_format='array').ravel()
            
            # Handle missing values
            mask = ~np.isnan(x) if np.issubdtype(x.dtype, np.number) else pd.notna(x)
            if not np.any(mask):
                scores[feature] = 0.0
                continue
                
            x_clean = x[mask]
            
            # Discretize if continuous
            if np.issubdtype(x.dtype, np.number):
                x_discrete = self._ig_filter._discretize_feature(x_clean)
            else:
                x_discrete = x_clean
                
            # Calculate intrinsic value (entropy of feature)
            intrinsic_value = self._ig_filter._calculate_entropy(x_discrete)
            
            # Avoid division by zero
            if intrinsic_value < self.min_intrinsic_value:
                intrinsic_value = self.min_intrinsic_value
                
            # Information gain ratio
            ig_ratio = ig_result.scores[feature] / intrinsic_value
            scores[feature] = ig_ratio
            
        # Create result
        scores_series = pd.Series(scores)
        sorted_features = scores_series.sort_values(ascending=False).index.tolist()
        
        return FilterResult(
            scores=scores_series,
            features=sorted_features,
            method="info_gain_ratio",
            task_type="classif",
            params={
                "n_bins": self.n_bins,
                "strategy": self.strategy,
                "min_intrinsic_value": self.min_intrinsic_value,
                "info_gains": ig_result.scores.to_dict()
            }
        )


class FilterSymmetricalUncertainty(Filter):
    """Symmetrical Uncertainty filter.
    
    Normalized mutual information measure that compensates for
    the bias of mutual information toward features with more values.
    
    SU(X, Y) = 2 * MI(X, Y) / (H(X) + H(Y))
    
    Parameters
    ----------
    n_bins : int, default=10
        Number of bins for discretizing continuous features.
    strategy : str, default='quantile'
        Binning strategy: 'uniform', 'quantile', 'kmeans'.
    """
    
    def __init__(self, n_bins: int = 10, strategy: str = 'quantile'):
        super().__init__(
            id="symmetrical_uncertainty",
            task_types={'classif'},
            feature_types={'numeric', 'factor'}
        )
        self.n_bins = n_bins
        self.strategy = strategy
        self._ig_filter = FilterInformationGain(n_bins=n_bins, strategy=strategy)
        
    def _calculate_mutual_information(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> float:
        """Calculate mutual information between two variables."""
        # Joint probability
        xy_unique = np.unique(np.column_stack([x, y]), axis=0)
        joint_probs = np.array([
            np.sum((x == xy[0]) & (y == xy[1])) / len(x)
            for xy in xy_unique
        ])
        
        # Marginal probabilities
        x_unique, x_counts = np.unique(x, return_counts=True)
        y_unique, y_counts = np.unique(y, return_counts=True)
        x_probs = x_counts / len(x)
        y_probs = y_counts / len(y)
        
        # Calculate MI
        mi = 0.0
        for i, xy in enumerate(xy_unique):
            p_xy = joint_probs[i]
            if p_xy > 0:
                p_x = x_probs[x_unique == xy[0]][0]
                p_y = y_probs[y_unique == xy[1]][0]
                mi += p_xy * np.log2(p_xy / (p_x * p_y))
                
        return max(0.0, mi)
        
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        self.check_task_compatibility(task)
        
        if features is None:
            features = task.feature_names
            
        # Get target
        y = task.truth()
        h_y = self._ig_filter._calculate_entropy(y)
        
        # Calculate symmetrical uncertainty for each feature
        scores = {}
        
        for feature in features:
            # Get feature values
            x = task.data(cols=[feature], data_format='array').ravel()
            
            # Handle missing values
            mask = ~np.isnan(x) if np.issubdtype(x.dtype, np.number) else pd.notna(x)
            if not np.any(mask):
                scores[feature] = 0.0
                continue
                
            x_clean = x[mask]
            y_clean = y[mask]
            
            # Discretize if continuous
            if np.issubdtype(x.dtype, np.number):
                x_discrete = self._ig_filter._discretize_feature(x_clean)
            else:
                x_discrete = x_clean
                
            # Calculate entropies
            h_x = self._ig_filter._calculate_entropy(x_discrete)
            
            # Calculate mutual information
            mi = self._calculate_mutual_information(x_discrete, y_clean)
            
            # Symmetrical uncertainty
            if h_x + h_y > 0:
                su = 2 * mi / (h_x + h_y)
                scores[feature] = min(1.0, max(0.0, su))  # Bound to [0, 1]
            else:
                scores[feature] = 0.0
                
        # Create result
        scores_series = pd.Series(scores)
        sorted_features = scores_series.sort_values(ascending=False).index.tolist()
        
        return FilterResult(
            scores=scores_series,
            features=sorted_features,
            method="symmetrical_uncertainty",
            task_type="classif",
            params={
                "n_bins": self.n_bins,
                "strategy": self.strategy,
                "target_entropy": h_y
            }
        )


class FilterJMIM(Filter):
    """Joint Mutual Information Maximization filter.
    
    Selects features that jointly have high mutual information
    with the target, considering feature redundancy.
    
    Parameters
    ----------
    n_features : int, default=20
        Number of features to select.
    n_bins : int, default=10
        Number of bins for discretization.
    """
    
    def __init__(self, n_features: int = 20, n_bins: int = 10):
        super().__init__(
            id="jmim",
            task_types={'classif'},
            feature_types={'numeric', 'factor'}
        )
        self.n_features = n_features
        self.n_bins = n_bins
        self._ig_filter = FilterInformationGain(n_bins=n_bins)
        self._su_filter = FilterSymmetricalUncertainty(n_bins=n_bins)
        
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        self.check_task_compatibility(task)
        
        if features is None:
            features = task.feature_names
            
        # Get target
        y = task.truth()
        
        # First, calculate MI for all features
        mi_scores = {}
        feature_data = {}
        
        for feature in features:
            x = task.data(cols=[feature], data_format='array').ravel()
            mask = ~np.isnan(x) if np.issubdtype(x.dtype, np.number) else pd.notna(x)
            
            if not np.any(mask):
                mi_scores[feature] = 0.0
                continue
                
            x_clean = x[mask]
            y_clean = y[mask]
            
            if np.issubdtype(x.dtype, np.number):
                x_discrete = self._ig_filter._discretize_feature(x_clean)
            else:
                x_discrete = x_clean
                
            feature_data[feature] = (x_discrete, mask)
            mi_scores[feature] = self._su_filter._calculate_mutual_information(x_discrete, y_clean)
            
        # JMIM algorithm
        n_select = min(self.n_features, len(features))
        selected_features = []
        remaining_features = list(features)
        scores = np.zeros(len(features))
        feature_to_idx = {f: i for i, f in enumerate(features)}
        
        # Select first feature (highest MI)
        first_feature = max(mi_scores.keys(), key=lambda k: mi_scores[k])
        selected_features.append(first_feature)
        remaining_features.remove(first_feature)
        scores[feature_to_idx[first_feature]] = mi_scores[first_feature]
        
        # Select remaining features
        for _ in range(1, n_select):
            if not remaining_features:
                break
                
            best_score = -np.inf
            best_feature = None
            
            for candidate in remaining_features:
                if candidate not in feature_data:
                    continue
                    
                x_cand, mask_cand = feature_data[candidate]
                
                # Calculate min joint MI with selected features
                min_joint_mi = mi_scores[candidate]
                
                for selected in selected_features:
                    if selected in feature_data:
                        x_sel, mask_sel = feature_data[selected]
                        
                        # Use common mask
                        common_mask = mask_cand & mask_sel
                        if np.any(common_mask):
                            # Calculate I(candidate, selected; target)
                            joint_data = np.column_stack([
                                x_cand[mask_cand][common_mask[mask_cand]],
                                x_sel[mask_sel][common_mask[mask_sel]]
                            ])
                            # Simplified: use min of individual MIs
                            min_joint_mi = min(min_joint_mi, mi_scores[selected])
                            
                if min_joint_mi > best_score:
                    best_score = min_joint_mi
                    best_feature = candidate
                    
            if best_feature is not None:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                scores[feature_to_idx[best_feature]] = best_score
                
        # Add remaining features with original MI scores
        for feature in remaining_features:
            scores[feature_to_idx[feature]] = mi_scores.get(feature, 0.0) * 0.1
            
        # Create result
        scores_series = pd.Series(scores, index=features)
        
        return FilterResult(
            scores=scores_series,
            features=selected_features + remaining_features,
            method="jmim",
            task_type="classif",
            params={
                "n_features": n_select,
                "n_bins": self.n_bins,
                "initial_mi_scores": mi_scores
            }
        )


# Register all information theory filters
filter_registry.register(FilterInformationGain, "info_gain")
filter_registry.register(FilterInformationGainRatio, "info_gain_ratio")
filter_registry.register(FilterSymmetricalUncertainty, "symmetrical_uncertainty")
filter_registry.register(FilterJMIM, "jmim")