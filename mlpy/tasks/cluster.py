"""
Clustering tasks for MLPY.

This module provides task types for unsupervised clustering analysis.
"""

from abc import ABC
from typing import Any, Dict, List, Optional, Union
import warnings
import numpy as np
import pandas as pd

from .base import Task
from mlpy.backends.base import DataBackend
from mlpy.backends.pandas_backend import DataBackendPandas
from mlpy.utils.registry import mlpy_tasks


class TaskCluster(Task):
    """
    Clustering (unsupervised learning) task.
    
    For grouping observations into clusters based on feature similarity
    without knowing the true cluster labels.
    
    Parameters
    ----------
    data : pd.DataFrame, DataBackend, or dict
        The data for the task
    id : str, optional
        Task identifier
    label : str, optional
        Task label
    n_clusters : int, optional
        Expected number of clusters (if known)
    exclude : List[str], optional
        Column names to exclude from clustering features
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, DataBackend, Dict[str, Any]],
        id: Optional[str] = None,
        label: Optional[str] = None,
        n_clusters: Optional[int] = None,
        exclude: Optional[List[str]] = None,
        **kwargs
    ):
        # Convert data to backend if needed
        if isinstance(data, pd.DataFrame):
            backend = DataBackendPandas(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
            backend = DataBackendPandas(df)
        elif isinstance(data, DataBackend):
            backend = data
        else:
            raise TypeError(
                f"data must be DataFrame, DataBackend, or dict, got {type(data)}"
            )
        
        super().__init__(backend=backend, id=id, label=label, **kwargs)
        
        self._n_clusters = n_clusters
        
        # Set column roles
        all_cols = set(backend.colnames)
        excluded_cols = set(exclude or [])
        
        # All non-excluded columns are features by default
        feature_cols = all_cols - excluded_cols
        
        # Warn about non-numeric features
        col_info = backend.col_info()
        numeric_features = set()
        non_numeric_features = set()
        
        for col in feature_cols:
            col_type = col_info[col]["type"]
            if col_type in ("numeric", "integer"):
                numeric_features.add(col)
            else:
                non_numeric_features.add(col)
        
        if non_numeric_features:
            warnings.warn(
                f"Non-numeric features found: {non_numeric_features}. "
                f"Consider preprocessing or excluding them."
            )
        
        # Set roles - clustering has no target, only features
        self.set_col_roles({
            "feature": list(feature_cols),
            "name": list(excluded_cols)  # Excluded columns become name/ID columns
        })
        
        self._validate_task()
    
    @property
    def task_type(self) -> str:
        """Task type identifier."""
        return "cluster"
    
    @property
    def n_clusters(self) -> Optional[int]:
        """Expected number of clusters."""
        return self._n_clusters
    
    @property
    def n_numeric_features(self) -> int:
        """Number of numeric features."""
        col_info = self._backend.col_info()
        return sum(
            1 for col in self.feature_names
            if col_info[col]["type"] in ("numeric", "integer")
        )
    
    @property
    def n_categorical_features(self) -> int:
        """Number of categorical features."""
        return len(self.feature_names) - self.n_numeric_features
    
    def _validate_task(self) -> None:
        """Validate clustering task requirements."""
        # Must have at least one feature
        if not self.feature_names:
            raise ValueError("Clustering task requires at least one feature")
        
        # Should have multiple observations
        if self.nrow < 2:
            raise ValueError(
                f"Clustering requires at least 2 observations, got {self.nrow}"
            )
        
        # Warn if very few observations relative to features
        if self.nrow < len(self.feature_names) * 3:
            warnings.warn(
                f"Few observations ({self.nrow}) relative to features "
                f"({len(self.feature_names)}). Consider feature selection."
            )
        
        # Validate n_clusters if provided
        if self._n_clusters is not None:
            if self._n_clusters < 2:
                raise ValueError("Number of clusters must be >= 2")
            if self._n_clusters >= self.nrow:
                raise ValueError(
                    f"Number of clusters ({self._n_clusters}) must be less "
                    f"than observations ({self.nrow})"
                )
    
    def get_numeric_features(self) -> List[str]:
        """Get list of numeric feature names."""
        col_info = self._backend.col_info()
        return [
            col for col in self.feature_names
            if col_info[col]["type"] in ("numeric", "integer")
        ]
    
    def get_categorical_features(self) -> List[str]:
        """Get list of categorical feature names."""
        numeric_features = set(self.get_numeric_features())
        return [col for col in self.feature_names if col not in numeric_features]
    
    def suggest_n_clusters(self, max_k: int = 10) -> Dict[str, Union[int, List[int]]]:
        """
        Suggest number of clusters using simple heuristics.
        
        Parameters
        ----------
        max_k : int
            Maximum number of clusters to consider
            
        Returns
        -------
        dict
            Dictionary with suggested number of clusters and reasoning
        """
        n_obs = self.nrow
        n_features = len(self.feature_names)
        
        # Rule of thumb heuristics
        sqrt_rule = max(2, int(np.sqrt(n_obs / 2)))
        elbow_rule = min(max_k, max(2, n_obs // 10))  # Rough elbow approximation
        feature_rule = min(max_k, max(2, n_features + 1))
        
        # Conservative range
        min_clusters = max(2, min(sqrt_rule, elbow_rule, feature_rule) - 1)
        max_clusters = min(max_k, max(sqrt_rule, elbow_rule, feature_rule) + 2)
        
        suggested_range = list(range(min_clusters, max_clusters + 1))
        
        # Pick middle value as primary suggestion
        primary_suggestion = suggested_range[len(suggested_range) // 2]
        
        return {
            "primary": primary_suggestion,
            "range": suggested_range,
            "heuristics": {
                "sqrt_rule": sqrt_rule,
                "elbow_rule": elbow_rule,
                "feature_rule": feature_rule
            },
            "reasoning": (
                f"Based on {n_obs} observations and {n_features} features, "
                f"suggested range is {min_clusters}-{max_clusters} clusters. "
                f"Primary suggestion: {primary_suggestion} clusters."
            )
        }
    
    def preprocess_features(self, 
                          standardize: bool = True,
                          handle_missing: str = "drop") -> pd.DataFrame:
        """
        Preprocess features for clustering.
        
        Parameters
        ----------
        standardize : bool
            Whether to standardize numeric features (recommended for clustering)
        handle_missing : str
            How to handle missing values: "drop", "mean", "median", "mode"
            
        Returns
        -------
        pd.DataFrame
            Preprocessed feature data
        """
        # Get feature data
        data = self.data(cols=self.feature_names)
        
        # Handle missing values
        if handle_missing == "drop":
            data = data.dropna()
        elif handle_missing == "mean":
            # Only fill numeric columns with mean
            numeric_cols = self.get_numeric_features()
            for col in numeric_cols:
                if col in data.columns:
                    data[col].fillna(data[col].mean(), inplace=True)
        elif handle_missing == "median":
            numeric_cols = self.get_numeric_features()
            for col in numeric_cols:
                if col in data.columns:
                    data[col].fillna(data[col].median(), inplace=True)
        elif handle_missing == "mode":
            for col in data.columns:
                mode_val = data[col].mode()
                if not mode_val.empty:
                    data[col].fillna(mode_val.iloc[0], inplace=True)
        
        # Standardize numeric features if requested
        if standardize:
            numeric_cols = self.get_numeric_features()
            for col in numeric_cols:
                if col in data.columns and data[col].std() > 0:
                    data[col] = (data[col] - data[col].mean()) / data[col].std()
        
        return data
    
    def distance_matrix(self, 
                       metric: str = "euclidean",
                       standardize: bool = True) -> np.ndarray:
        """
        Compute pairwise distance matrix.
        
        Parameters
        ----------
        metric : str
            Distance metric: "euclidean", "manhattan", "cosine"
        standardize : bool
            Whether to standardize features first
            
        Returns
        -------
        np.ndarray
            Pairwise distance matrix
        """
        # Get preprocessed numeric data only
        numeric_features = self.get_numeric_features()
        if not numeric_features:
            raise ValueError("Distance matrix requires numeric features")
        
        data = self.preprocess_features(standardize=standardize)
        numeric_data = data[numeric_features].values
        
        n = numeric_data.shape[0]
        distances = np.zeros((n, n))
        
        if metric == "euclidean":
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt(np.sum((numeric_data[i] - numeric_data[j]) ** 2))
                    distances[i, j] = distances[j, i] = dist
        elif metric == "manhattan":
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sum(np.abs(numeric_data[i] - numeric_data[j]))
                    distances[i, j] = distances[j, i] = dist
        elif metric == "cosine":
            for i in range(n):
                for j in range(i+1, n):
                    dot_product = np.dot(numeric_data[i], numeric_data[j])
                    norm_i = np.linalg.norm(numeric_data[i])
                    norm_j = np.linalg.norm(numeric_data[j])
                    if norm_i == 0 or norm_j == 0:
                        dist = 1.0
                    else:
                        dist = 1 - (dot_product / (norm_i * norm_j))
                    distances[i, j] = distances[j, i] = dist
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return distances
    
    def evaluate_clustering(self, 
                          cluster_labels: np.ndarray,
                          metric: str = "silhouette") -> float:
        """
        Evaluate clustering quality.
        
        Parameters
        ----------
        cluster_labels : np.ndarray
            Predicted cluster labels
        metric : str
            Evaluation metric: "silhouette", "inertia"
            
        Returns
        -------
        float
            Clustering quality score
        """
        if len(cluster_labels) != self.nrow:
            raise ValueError(
                f"cluster_labels length ({len(cluster_labels)}) "
                f"must match task rows ({self.nrow})"
            )
        
        # Get numeric data for evaluation
        numeric_features = self.get_numeric_features()
        if not numeric_features:
            raise ValueError("Clustering evaluation requires numeric features")
        
        data = self.preprocess_features()
        X = data[numeric_features].values
        
        if metric == "silhouette":
            # Simple silhouette calculation
            n = len(cluster_labels)
            silhouette_scores = []
            
            for i in range(n):
                # Same cluster distances (a)
                same_cluster = cluster_labels == cluster_labels[i]
                if np.sum(same_cluster) == 1:
                    # Only one point in cluster
                    silhouette_scores.append(0)
                    continue
                
                a = np.mean([
                    np.linalg.norm(X[i] - X[j])
                    for j in range(n)
                    if same_cluster[j] and j != i
                ])
                
                # Nearest cluster distances (b)
                unique_clusters = np.unique(cluster_labels)
                other_clusters = unique_clusters[unique_clusters != cluster_labels[i]]
                
                if len(other_clusters) == 0:
                    silhouette_scores.append(0)
                    continue
                
                b = min([
                    np.mean([
                        np.linalg.norm(X[i] - X[j])
                        for j in range(n)
                        if cluster_labels[j] == cluster
                    ])
                    for cluster in other_clusters
                ])
                
                # Silhouette score for point i
                if max(a, b) == 0:
                    silhouette_scores.append(0)
                else:
                    silhouette_scores.append((b - a) / max(a, b))
            
            return np.mean(silhouette_scores)
        
        elif metric == "inertia":
            # Within-cluster sum of squares
            inertia = 0
            unique_clusters = np.unique(cluster_labels)
            
            for cluster in unique_clusters:
                cluster_points = X[cluster_labels == cluster]
                if len(cluster_points) > 0:
                    cluster_center = np.mean(cluster_points, axis=0)
                    inertia += np.sum((cluster_points - cluster_center) ** 2)
            
            return -inertia  # Return negative (higher is better)
        
        else:
            raise ValueError(f"Unknown evaluation metric: {metric}")
    
    @property
    def _properties(self) -> set[str]:
        """Task properties."""
        props = super()._properties
        props.add("unsupervised")
        
        if self.n_numeric_features > 0:
            props.add("numeric")
        
        if self.n_categorical_features > 0:
            props.add("categorical")
        
        if self._n_clusters:
            props.add(f"k_{self._n_clusters}")
        
        # Add size-based properties
        if self.nrow < 1000:
            props.add("small")
        elif self.nrow > 100000:
            props.add("large")
        
        return props


# Register task type
mlpy_tasks.register("cluster", TaskCluster, aliases=["clustering"])


__all__ = ["TaskCluster"]