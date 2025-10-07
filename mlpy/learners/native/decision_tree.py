"""
Native Decision Tree implementation for MLPY.

This is a pure Python/NumPy implementation of decision trees
for both classification and regression.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass

from ...tasks import Task, TaskClassif, TaskRegr
from ..classification import LearnerClassif
from ..regression import LearnerRegr
from ...predictions import PredictionClassif, PredictionRegr


@dataclass
class TreeNode:
    """Node in a decision tree."""
    
    # Split information
    feature: Optional[int] = None  # Feature index for split
    threshold: Optional[float] = None  # Threshold for split
    
    # Child nodes
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    
    # Leaf information
    is_leaf: bool = False
    value: Optional[Union[float, str, int]] = None  # Prediction value
    n_samples: int = 0  # Number of samples in node
    
    # For classification
    class_counts: Optional[Dict[Any, int]] = None  # Class distribution
    
    # For regression
    mean: Optional[float] = None  # Mean value (regression)
    std: Optional[float] = None  # Standard deviation (regression)


class DecisionTreeBase:
    """Base class for decision tree implementations."""
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        
        self.tree_: Optional[TreeNode] = None
        self.n_features_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.rng = np.random.RandomState(random_state)
        
    def _get_n_features_split(self, n_features: int) -> int:
        """Calculate number of features to consider for split."""
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        else:
            return n_features
            
    def _gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity for classification."""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
        
    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy for classification."""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        # Avoid log(0)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
        
    def _mse(self, y: np.ndarray) -> float:
        """Calculate MSE for regression."""
        if len(y) == 0:
            return 0
        return np.var(y)
        
    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        features: List[int]
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Find best split for given features."""
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        n_samples = len(y)
        if n_samples < self.min_samples_split:
            return None, None, 0
            
        # Calculate impurity of current node
        if self._task_type == 'classif':
            parent_impurity = self._gini_impurity(y)
        else:
            parent_impurity = self._mse(y)
            
        # Try each feature
        for feature_idx in features:
            feature_values = X[:, feature_idx]
            
            # Get unique values for thresholds
            unique_values = np.unique(feature_values)
            if len(unique_values) <= 1:
                continue
                
            # Try thresholds between unique values
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                # Split data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                # Check minimum samples per leaf
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                    
                # Calculate impurity of splits
                if self._task_type == 'classif':
                    left_impurity = self._gini_impurity(y[left_mask])
                    right_impurity = self._gini_impurity(y[right_mask])
                else:
                    left_impurity = self._mse(y[left_mask])
                    right_impurity = self._mse(y[right_mask])
                    
                # Calculate information gain
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                gain = parent_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_feature, best_threshold, best_gain
        
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        """Recursively build decision tree."""
        n_samples, n_features = X.shape
        
        # Create node
        node = TreeNode(n_samples=n_samples)
        
        # For classification, store class distribution
        if self._task_type == 'classif':
            unique_classes, counts = np.unique(y, return_counts=True)
            node.class_counts = dict(zip(unique_classes, counts))
            node.value = unique_classes[np.argmax(counts)]
        else:
            # For regression, store mean and std
            node.mean = np.mean(y)
            node.std = np.std(y)
            node.value = node.mean
            
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            node.is_leaf = True
            return node
            
        # Select features to consider
        n_features_split = self._get_n_features_split(n_features)
        if n_features_split < n_features:
            features = self.rng.choice(n_features, n_features_split, replace=False)
        else:
            features = list(range(n_features))
            
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y, features)
        
        if best_feature is None:
            node.is_leaf = True
            return node
            
        # Store split information
        node.feature = best_feature
        node.threshold = best_threshold
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build child nodes
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
        
    def _predict_sample(self, x: np.ndarray, node: TreeNode) -> Any:
        """Predict single sample by traversing tree."""
        if node.is_leaf:
            return node.value
            
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
            
    def _predict_proba_sample(self, x: np.ndarray, node: TreeNode) -> Dict[Any, float]:
        """Get probability distribution for single sample."""
        if node.is_leaf:
            if node.class_counts:
                total = sum(node.class_counts.values())
                return {k: v/total for k, v in node.class_counts.items()}
            else:
                return {node.value: 1.0}
                
        if x[node.feature] <= node.threshold:
            return self._predict_proba_sample(x, node.left)
        else:
            return self._predict_proba_sample(x, node.right)
            
    def _calculate_feature_importances(self, node: TreeNode, importances: np.ndarray, n_samples_total: int):
        """Calculate feature importances recursively."""
        if node.is_leaf:
            return
            
        # Calculate importance for this split
        n_samples = node.n_samples
        if node.left and node.right:
            # Calculate impurity decrease
            if self._task_type == 'classif':
                parent_impurity = self._gini_impurity_from_counts(node.class_counts)
                left_impurity = self._gini_impurity_from_counts(node.left.class_counts)
                right_impurity = self._gini_impurity_from_counts(node.right.class_counts)
            else:
                parent_impurity = node.std ** 2 if node.std else 0
                left_impurity = node.left.std ** 2 if node.left.std else 0
                right_impurity = node.right.std ** 2 if node.right.std else 0
                
            weighted_impurity = (
                node.left.n_samples * left_impurity + 
                node.right.n_samples * right_impurity
            ) / n_samples
            
            importance = (n_samples / n_samples_total) * (parent_impurity - weighted_impurity)
            importances[node.feature] += importance
            
        # Recurse on children
        if node.left:
            self._calculate_feature_importances(node.left, importances, n_samples_total)
        if node.right:
            self._calculate_feature_importances(node.right, importances, n_samples_total)
            
    def _gini_impurity_from_counts(self, class_counts: Dict[Any, int]) -> float:
        """Calculate Gini impurity from class counts."""
        if not class_counts:
            return 0
        total = sum(class_counts.values())
        if total == 0:
            return 0
        probs = np.array(list(class_counts.values())) / total
        return 1 - np.sum(probs ** 2)


class LearnerDecisionTree(LearnerClassif, DecisionTreeBase):
    """Native Decision Tree classifier for MLPY."""
    
    def __init__(
        self,
        id: Optional[str] = None,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None,
        predict_type: str = "response",
        **kwargs
    ):
        # Initialize learner
        super().__init__(
            id=id or f"decision_tree_native",
            predict_type=predict_type,
            feature_types=["numeric"],
            predict_types=["response", "prob"],
            properties=["multiclass", "twoclass", "importance"],
            **kwargs
        )
        
        # Initialize tree base
        DecisionTreeBase.__init__(
            self,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )
        
        self._task_type = 'classif'
        self.classes_ = None
        
    def _train(self, task: TaskClassif, row_ids: Optional[List[int]] = None) -> "LearnerDecisionTree":
        """Train decision tree classifier."""
        # Get training data
        X = task.data(rows=row_ids, cols=task.feature_names)
        y = task.truth(rows=row_ids)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Store metadata
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        
        # Build tree
        self.tree_ = self._build_tree(X, y)
        
        # Calculate feature importances
        self.feature_importances_ = np.zeros(self.n_features_)
        self._calculate_feature_importances(self.tree_, self.feature_importances_, len(y))
        
        # Normalize importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
            
        return self
        
    def _predict(self, task: TaskClassif, row_ids: Optional[List[int]] = None) -> PredictionClassif:
        """Make predictions with decision tree."""
        # Get prediction data
        X = task.data(rows=row_ids, cols=task.feature_names)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        n_samples = X.shape[0]
        
        # Get predictions based on predict_type
        if self.predict_type == "response":
            predictions = np.array([self._predict_sample(X[i], self.tree_) for i in range(n_samples)])
            return PredictionClassif(
                task=task,
                learner_id=self.id,
                row_ids=row_ids,
                response=predictions,
                truth=task.truth(rows=row_ids) if task.col_roles.get("target") else None
            )
        else:  # prob
            # Get probability distributions
            prob_dicts = [self._predict_proba_sample(X[i], self.tree_) for i in range(n_samples)]
            
            # Convert to probability matrix
            prob_matrix = np.zeros((n_samples, len(self.classes_)))
            for i, prob_dict in enumerate(prob_dicts):
                for j, cls in enumerate(self.classes_):
                    prob_matrix[i, j] = prob_dict.get(cls, 0.0)
                    
            return PredictionClassif(
                task=task,
                learner_id=self.id,
                row_ids=row_ids,
                prob=prob_matrix,
                truth=task.truth(rows=row_ids) if task.col_roles.get("target") else None
            )
            
    def importance(self) -> Optional[np.ndarray]:
        """Get feature importances."""
        return self.feature_importances_


class LearnerDecisionTreeRegressor(LearnerRegr, DecisionTreeBase):
    """Native Decision Tree regressor for MLPY."""
    
    def __init__(
        self,
        id: Optional[str] = None,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None,
        **kwargs
    ):
        # Initialize learner
        super().__init__(
            id=id or f"decision_tree_regressor_native",
            feature_types=["numeric"],
            predict_types=["response", "se"],
            properties=["importance"],
            **kwargs
        )
        
        # Initialize tree base
        DecisionTreeBase.__init__(
            self,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )
        
        self._task_type = 'regr'
        
    def _train(self, task: TaskRegr, row_ids: Optional[List[int]] = None) -> "LearnerDecisionTreeRegressor":
        """Train decision tree regressor."""
        # Get training data
        X = task.data(rows=row_ids, cols=task.feature_names)
        y = task.truth(rows=row_ids)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Store metadata
        self.n_features_ = X.shape[1]
        
        # Build tree
        self.tree_ = self._build_tree(X, y)
        
        # Calculate feature importances
        self.feature_importances_ = np.zeros(self.n_features_)
        self._calculate_feature_importances(self.tree_, self.feature_importances_, len(y))
        
        # Normalize importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
            
        return self
        
    def _predict(self, task: TaskRegr, row_ids: Optional[List[int]] = None) -> PredictionRegr:
        """Make predictions with decision tree regressor."""
        # Get prediction data
        X = task.data(rows=row_ids, cols=task.feature_names)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        n_samples = X.shape[0]
        
        # Get predictions
        predictions = np.array([self._predict_sample(X[i], self.tree_) for i in range(n_samples)])
        
        # For SE, use the standard deviation from the leaf nodes
        se = None
        if self.predict_type == "se":
            se = np.zeros(n_samples)
            for i in range(n_samples):
                # Traverse to leaf and get std
                node = self.tree_
                x = X[i]
                while not node.is_leaf:
                    if x[node.feature] <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
                se[i] = node.std if node.std is not None else 0.0
                
        return PredictionRegr(
            task=task,
            learner_id=self.id,
            row_ids=row_ids,
            response=predictions,
            se=se,
            truth=task.truth(rows=row_ids) if task.col_roles.get("target") else None
        )
        
    def importance(self) -> Optional[np.ndarray]:
        """Get feature importances."""
        return self.feature_importances_