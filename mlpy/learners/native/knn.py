"""
Native K-Nearest Neighbors implementation for MLPY.

This is a pure Python/NumPy implementation of KNN
for both classification and regression.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Literal, Union
from collections import Counter
import warnings

from ...tasks import TaskClassif, TaskRegr
from ..classification import LearnerClassif
from ..regression import LearnerRegr
from ...predictions import PredictionClassif, PredictionRegr


class KNNBase:
    """Base class for K-Nearest Neighbors implementations."""
    
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal['uniform', 'distance'] = 'uniform',
        metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean',
        p: int = 2,  # Parameter for minkowski metric
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        
        # Stored training data
        self.X_train = None
        self.y_train = None
        self.n_features_ = None
        
    def _compute_distances(self, X: np.ndarray, X_query: np.ndarray) -> np.ndarray:
        """Compute distances between query points and training data.
        
        Returns distance matrix of shape (n_query, n_train).
        """
        n_query = X_query.shape[0]
        n_train = X.shape[0]
        
        if self.metric == 'euclidean':
            # Efficient euclidean distance computation
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            X_norm = np.sum(X**2, axis=1)
            X_query_norm = np.sum(X_query**2, axis=1)
            distances = X_query_norm[:, np.newaxis] + X_norm - 2 * X_query @ X.T
            # Avoid negative distances due to numerical errors
            distances = np.maximum(distances, 0)
            distances = np.sqrt(distances)
        
        elif self.metric == 'manhattan':
            # Manhattan distance (L1 norm)
            distances = np.zeros((n_query, n_train))
            for i in range(n_query):
                distances[i] = np.sum(np.abs(X - X_query[i]), axis=1)
                
        elif self.metric == 'minkowski':
            # Minkowski distance (Lp norm)
            distances = np.zeros((n_query, n_train))
            for i in range(n_query):
                distances[i] = np.sum(np.abs(X - X_query[i])**self.p, axis=1)**(1/self.p)
                
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
            
        return distances
        
    def _get_neighbors(self, distances: np.ndarray) -> tuple:
        """Get indices and distances of k nearest neighbors.
        
        Returns:
            neighbor_indices: shape (n_query, k)
            neighbor_distances: shape (n_query, k)
        """
        n_query = distances.shape[0]
        k = min(self.n_neighbors, distances.shape[1])
        
        # Get indices of k smallest distances for each query point
        neighbor_indices = np.argpartition(distances, k-1, axis=1)[:, :k]
        
        # Get the actual distances for these neighbors
        neighbor_distances = np.zeros((n_query, k))
        for i in range(n_query):
            neighbor_distances[i] = distances[i, neighbor_indices[i]]
            # Sort neighbors by distance
            sort_idx = np.argsort(neighbor_distances[i])
            neighbor_indices[i] = neighbor_indices[i][sort_idx]
            neighbor_distances[i] = neighbor_distances[i][sort_idx]
            
        return neighbor_indices, neighbor_distances
        
    def _get_weights(self, distances: np.ndarray) -> np.ndarray:
        """Calculate weights for neighbors based on distances."""
        if self.weights == 'uniform':
            return np.ones_like(distances)
        else:  # distance
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                weights = 1 / distances
                weights[distances == 0] = 1e10  # Very large weight for exact matches
            return weights


class LearnerKNN(LearnerClassif, KNNBase):
    """Native K-Nearest Neighbors classifier for MLPY.
    
    Parameters
    ----------
    id : str, optional
        Identifier for the learner.
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction:
        - 'uniform': All neighbors have equal weight
        - 'distance': Weight neighbors by inverse of their distance
    metric : {'euclidean', 'manhattan', 'minkowski'}, default='euclidean'
        Distance metric to use.
    p : int, default=2
        Parameter for the Minkowski metric.
    predict_type : str, default='response'
        Type of prediction to make.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        n_neighbors: int = 5,
        weights: Literal['uniform', 'distance'] = 'uniform',
        metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean',
        p: int = 2,
        predict_type: str = "response",
        **kwargs
    ):
        # Initialize learner
        super().__init__(
            id=id or "knn_native",
            predict_type=predict_type,
            predict_types=["response", "prob"],
            properties=["twoclass", "multiclass"],
            feature_types=["numeric"],
            **kwargs
        )
        
        # Initialize KNN base
        KNNBase.__init__(
            self,
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p
        )
        
        self.classes_ = None
        
    def _train(self, task: TaskClassif, row_ids: Optional[List[int]] = None) -> "LearnerKNN":
        """Train KNN classifier (store training data)."""
        # Get training data
        X = task.data(rows=row_ids, cols=task.feature_names)
        y = task.truth(rows=row_ids)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Store training data
        self.X_train = X
        self.y_train = y
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        
        # Validate n_neighbors
        if self.n_neighbors > len(self.X_train):
            warnings.warn(
                f"n_neighbors ({self.n_neighbors}) is greater than "
                f"the number of training samples ({len(self.X_train)}). "
                f"Using {len(self.X_train)} neighbors."
            )
            
        return self
        
    def _predict(self, task: TaskClassif, row_ids: Optional[List[int]] = None) -> PredictionClassif:
        """Make predictions with KNN classifier."""
        # Get prediction data
        X = task.data(rows=row_ids, cols=task.feature_names)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        n_samples = X.shape[0]
        
        # Compute distances to all training points
        distances = self._compute_distances(self.X_train, X)
        
        # Get k nearest neighbors
        neighbor_indices, neighbor_distances = self._get_neighbors(distances)
        
        if self.predict_type == "response":
            # Predict classes
            predictions = np.empty(n_samples, dtype=self.y_train.dtype)
            
            for i in range(n_samples):
                # Get labels of neighbors
                neighbor_labels = self.y_train[neighbor_indices[i]]
                
                if self.weights == 'uniform':
                    # Simple majority vote
                    predictions[i] = Counter(neighbor_labels).most_common(1)[0][0]
                else:
                    # Weighted vote
                    weights = self._get_weights(neighbor_distances[i])
                    # Calculate weighted counts for each class
                    weighted_counts = {}
                    for label, weight in zip(neighbor_labels, weights):
                        weighted_counts[label] = weighted_counts.get(label, 0) + weight
                    # Get class with highest weighted count
                    predictions[i] = max(weighted_counts, key=weighted_counts.get)
                    
            return PredictionClassif(
                task=task,
                learner_id=self.id,
                row_ids=row_ids,
                response=predictions,
                truth=task.truth(rows=row_ids) if task.col_roles.get("target") else None
            )
            
        else:  # prob
            # Predict probabilities
            prob_matrix = np.zeros((n_samples, len(self.classes_)))
            
            for i in range(n_samples):
                # Get labels and weights of neighbors
                neighbor_labels = self.y_train[neighbor_indices[i]]
                weights = self._get_weights(neighbor_distances[i])
                
                # Calculate weighted probability for each class
                for j, cls in enumerate(self.classes_):
                    mask = neighbor_labels == cls
                    prob_matrix[i, j] = np.sum(weights[mask]) / np.sum(weights)
                    
            return PredictionClassif(
                task=task,
                learner_id=self.id,
                row_ids=row_ids,
                prob=prob_matrix,
                truth=task.truth(rows=row_ids) if task.col_roles.get("target") else None
            )


class LearnerKNNRegressor(LearnerRegr, KNNBase):
    """Native K-Nearest Neighbors regressor for MLPY.
    
    Parameters
    ----------
    id : str, optional
        Identifier for the learner.
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction.
    metric : {'euclidean', 'manhattan', 'minkowski'}, default='euclidean'
        Distance metric to use.
    p : int, default=2
        Parameter for the Minkowski metric.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        n_neighbors: int = 5,
        weights: Literal['uniform', 'distance'] = 'uniform',
        metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean',
        p: int = 2,
        **kwargs
    ):
        # Initialize learner
        super().__init__(
            id=id or "knn_regressor_native",
            predict_types=["response", "se"],
            properties=[],
            feature_types=["numeric"],
            **kwargs
        )
        
        # Initialize KNN base
        KNNBase.__init__(
            self,
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p
        )
        
    def _train(self, task: TaskRegr, row_ids: Optional[List[int]] = None) -> "LearnerKNNRegressor":
        """Train KNN regressor (store training data)."""
        # Get training data
        X = task.data(rows=row_ids, cols=task.feature_names)
        y = task.truth(rows=row_ids)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Store training data
        self.X_train = X
        self.y_train = y
        self.n_features_ = X.shape[1]
        
        # Validate n_neighbors
        if self.n_neighbors > len(self.X_train):
            warnings.warn(
                f"n_neighbors ({self.n_neighbors}) is greater than "
                f"the number of training samples ({len(self.X_train)}). "
                f"Using {len(self.X_train)} neighbors."
            )
            
        return self
        
    def _predict(self, task: TaskRegr, row_ids: Optional[List[int]] = None) -> PredictionRegr:
        """Make predictions with KNN regressor."""
        # Get prediction data
        X = task.data(rows=row_ids, cols=task.feature_names)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        n_samples = X.shape[0]
        
        # Compute distances to all training points
        distances = self._compute_distances(self.X_train, X)
        
        # Get k nearest neighbors
        neighbor_indices, neighbor_distances = self._get_neighbors(distances)
        
        # Predict values
        predictions = np.zeros(n_samples)
        se = None
        
        if self.predict_type == "se":
            se = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Get values of neighbors
            neighbor_values = self.y_train[neighbor_indices[i]]
            
            if self.weights == 'uniform':
                # Simple average
                predictions[i] = np.mean(neighbor_values)
                if self.predict_type == "se":
                    # Standard error based on neighbor variance
                    se[i] = np.std(neighbor_values) / np.sqrt(len(neighbor_values))
            else:
                # Weighted average
                weights = self._get_weights(neighbor_distances[i])
                predictions[i] = np.sum(weights * neighbor_values) / np.sum(weights)
                if self.predict_type == "se":
                    # Weighted standard error
                    weighted_var = np.sum(weights * (neighbor_values - predictions[i])**2) / np.sum(weights)
                    se[i] = np.sqrt(weighted_var / len(neighbor_values))
                    
        return PredictionRegr(
            task=task,
            learner_id=self.id,
            row_ids=row_ids,
            response=predictions,
            se=se,
            truth=task.truth(rows=row_ids) if task.col_roles.get("target") else None
        )