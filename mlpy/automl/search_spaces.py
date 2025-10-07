"""
Search Spaces for AutoML
=========================

Define hyperparameter search spaces for different models.
"""

import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

logger = logging.getLogger(__name__)


class SearchSpace(ABC):
    """Abstract base class for search spaces."""
    
    @abstractmethod
    def sample(self, random_state: Optional[int] = None) -> Any:
        """Sample a value from the search space."""
        pass
    
    @abstractmethod
    def get_bounds(self) -> Tuple[Any, Any]:
        """Get bounds of the search space."""
        pass
    
    @abstractmethod
    def contains(self, value: Any) -> bool:
        """Check if value is in search space."""
        pass


class CategoricalSpace(SearchSpace):
    """Categorical search space."""
    
    def __init__(self, choices: List[Any]):
        """
        Initialize categorical space.
        
        Args:
            choices: List of possible values
        """
        self.choices = choices
    
    def sample(self, random_state: Optional[int] = None) -> Any:
        """Sample a value."""
        rng = np.random.RandomState(random_state)
        return rng.choice(self.choices)
    
    def get_bounds(self) -> Tuple[List[Any], None]:
        """Get choices."""
        return self.choices, None
    
    def contains(self, value: Any) -> bool:
        """Check if value is in choices."""
        return value in self.choices


class NumericSpace(SearchSpace):
    """Numeric search space."""
    
    def __init__(
        self,
        low: float,
        high: float,
        distribution: str = "uniform",
        log_scale: bool = False,
        dtype: type = float
    ):
        """
        Initialize numeric space.
        
        Args:
            low: Lower bound
            high: Upper bound
            distribution: "uniform" or "normal"
            log_scale: Whether to use log scale
            dtype: int or float
        """
        self.low = low
        self.high = high
        self.distribution = distribution
        self.log_scale = log_scale
        self.dtype = dtype
    
    def sample(self, random_state: Optional[int] = None) -> Union[int, float]:
        """Sample a value."""
        rng = np.random.RandomState(random_state)
        
        if self.log_scale:
            value = np.exp(rng.uniform(np.log(self.low), np.log(self.high)))
        elif self.distribution == "uniform":
            value = rng.uniform(self.low, self.high)
        else:  # normal
            mean = (self.low + self.high) / 2
            std = (self.high - self.low) / 6  # 99.7% within bounds
            value = np.clip(rng.normal(mean, std), self.low, self.high)
        
        if self.dtype == int:
            value = int(np.round(value))
        
        return value
    
    def get_bounds(self) -> Tuple[float, float]:
        """Get bounds."""
        return self.low, self.high
    
    def contains(self, value: Union[int, float]) -> bool:
        """Check if value is in bounds."""
        return self.low <= value <= self.high


@dataclass
class ModelSearchSpace:
    """Search space for a model."""
    
    model_class: type
    parameters: Dict[str, SearchSpace]
    
    def sample_config(self, random_state: Optional[int] = None) -> Dict[str, Any]:
        """Sample a configuration."""
        config = {}
        rng = np.random.RandomState(random_state)
        
        for param_name, space in self.parameters.items():
            # Use different seed for each parameter
            seed = None if random_state is None else rng.randint(0, 2**31 - 1)
            config[param_name] = space.sample(seed)
        
        return config
    
    def create_model(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create model with configuration."""
        if config is None:
            config = self.sample_config()
        
        return self.model_class(**config)


def get_classification_search_spaces() -> Dict[str, ModelSearchSpace]:
    """Get search spaces for classification models."""
    spaces = {}
    
    # Random Forest
    spaces['RandomForest'] = ModelSearchSpace(
        model_class=RandomForestClassifier,
        parameters={
            'n_estimators': NumericSpace(50, 500, dtype=int),
            'max_depth': NumericSpace(3, 20, dtype=int),
            'min_samples_split': NumericSpace(2, 20, dtype=int),
            'min_samples_leaf': NumericSpace(1, 10, dtype=int),
            'max_features': CategoricalSpace(['sqrt', 'log2', None]),
            'bootstrap': CategoricalSpace([True, False]),
            'criterion': CategoricalSpace(['gini', 'entropy'])
        }
    )
    
    # Gradient Boosting
    spaces['GradientBoosting'] = ModelSearchSpace(
        model_class=GradientBoostingClassifier,
        parameters={
            'n_estimators': NumericSpace(50, 300, dtype=int),
            'learning_rate': NumericSpace(0.01, 0.3, log_scale=True),
            'max_depth': NumericSpace(3, 10, dtype=int),
            'min_samples_split': NumericSpace(2, 20, dtype=int),
            'min_samples_leaf': NumericSpace(1, 10, dtype=int),
            'subsample': NumericSpace(0.5, 1.0),
            'max_features': CategoricalSpace(['sqrt', 'log2', None])
        }
    )
    
    # Extra Trees
    spaces['ExtraTrees'] = ModelSearchSpace(
        model_class=ExtraTreesClassifier,
        parameters={
            'n_estimators': NumericSpace(50, 500, dtype=int),
            'max_depth': NumericSpace(3, 20, dtype=int),
            'min_samples_split': NumericSpace(2, 20, dtype=int),
            'min_samples_leaf': NumericSpace(1, 10, dtype=int),
            'max_features': CategoricalSpace(['sqrt', 'log2', None]),
            'bootstrap': CategoricalSpace([True, False])
        }
    )
    
    # Logistic Regression
    spaces['LogisticRegression'] = ModelSearchSpace(
        model_class=LogisticRegression,
        parameters={
            'C': NumericSpace(0.001, 10, log_scale=True),
            'penalty': CategoricalSpace(['l1', 'l2']),
            'solver': CategoricalSpace(['liblinear', 'saga']),
            'max_iter': NumericSpace(100, 1000, dtype=int)
        }
    )
    
    # SVM
    spaces['SVM'] = ModelSearchSpace(
        model_class=SVC,
        parameters={
            'C': NumericSpace(0.001, 100, log_scale=True),
            'kernel': CategoricalSpace(['linear', 'poly', 'rbf', 'sigmoid']),
            'gamma': CategoricalSpace(['scale', 'auto']),
            'degree': NumericSpace(2, 5, dtype=int),
            'probability': CategoricalSpace([True])
        }
    )
    
    # K-Nearest Neighbors
    spaces['KNN'] = ModelSearchSpace(
        model_class=KNeighborsClassifier,
        parameters={
            'n_neighbors': NumericSpace(3, 50, dtype=int),
            'weights': CategoricalSpace(['uniform', 'distance']),
            'algorithm': CategoricalSpace(['auto', 'ball_tree', 'kd_tree', 'brute']),
            'p': NumericSpace(1, 3, dtype=int)
        }
    )
    
    # Decision Tree
    spaces['DecisionTree'] = ModelSearchSpace(
        model_class=DecisionTreeClassifier,
        parameters={
            'max_depth': NumericSpace(3, 20, dtype=int),
            'min_samples_split': NumericSpace(2, 20, dtype=int),
            'min_samples_leaf': NumericSpace(1, 10, dtype=int),
            'criterion': CategoricalSpace(['gini', 'entropy']),
            'splitter': CategoricalSpace(['best', 'random'])
        }
    )
    
    # AdaBoost
    spaces['AdaBoost'] = ModelSearchSpace(
        model_class=AdaBoostClassifier,
        parameters={
            'n_estimators': NumericSpace(50, 300, dtype=int),
            'learning_rate': NumericSpace(0.01, 2.0, log_scale=True),
            'algorithm': CategoricalSpace(['SAMME', 'SAMME.R'])
        }
    )
    
    # XGBoost
    if HAS_XGBOOST:
        spaces['XGBoost'] = ModelSearchSpace(
            model_class=XGBClassifier,
            parameters={
                'n_estimators': NumericSpace(50, 500, dtype=int),
                'max_depth': NumericSpace(3, 15, dtype=int),
                'learning_rate': NumericSpace(0.01, 0.3, log_scale=True),
                'subsample': NumericSpace(0.5, 1.0),
                'colsample_bytree': NumericSpace(0.5, 1.0),
                'min_child_weight': NumericSpace(1, 10, dtype=int),
                'gamma': NumericSpace(0, 0.5),
                'reg_alpha': NumericSpace(0, 1),
                'reg_lambda': NumericSpace(0, 1)
            }
        )
    
    # LightGBM
    if HAS_LIGHTGBM:
        spaces['LightGBM'] = ModelSearchSpace(
            model_class=LGBMClassifier,
            parameters={
                'n_estimators': NumericSpace(50, 500, dtype=int),
                'max_depth': NumericSpace(3, 15, dtype=int),
                'learning_rate': NumericSpace(0.01, 0.3, log_scale=True),
                'num_leaves': NumericSpace(20, 300, dtype=int),
                'subsample': NumericSpace(0.5, 1.0),
                'colsample_bytree': NumericSpace(0.5, 1.0),
                'min_child_samples': NumericSpace(5, 30, dtype=int),
                'reg_alpha': NumericSpace(0, 1),
                'reg_lambda': NumericSpace(0, 1)
            }
        )
    
    # CatBoost
    if HAS_CATBOOST:
        spaces['CatBoost'] = ModelSearchSpace(
            model_class=CatBoostClassifier,
            parameters={
                'iterations': NumericSpace(50, 500, dtype=int),
                'depth': NumericSpace(3, 10, dtype=int),
                'learning_rate': NumericSpace(0.01, 0.3, log_scale=True),
                'l2_leaf_reg': NumericSpace(1, 10),
                'border_count': NumericSpace(32, 255, dtype=int),
                'verbose': CategoricalSpace([False])
            }
        )
    
    return spaces


def get_regression_search_spaces() -> Dict[str, ModelSearchSpace]:
    """Get search spaces for regression models."""
    spaces = {}
    
    # Random Forest
    spaces['RandomForest'] = ModelSearchSpace(
        model_class=RandomForestRegressor,
        parameters={
            'n_estimators': NumericSpace(50, 500, dtype=int),
            'max_depth': NumericSpace(3, 20, dtype=int),
            'min_samples_split': NumericSpace(2, 20, dtype=int),
            'min_samples_leaf': NumericSpace(1, 10, dtype=int),
            'max_features': CategoricalSpace(['sqrt', 'log2', None]),
            'bootstrap': CategoricalSpace([True, False])
        }
    )
    
    # Gradient Boosting
    spaces['GradientBoosting'] = ModelSearchSpace(
        model_class=GradientBoostingRegressor,
        parameters={
            'n_estimators': NumericSpace(50, 300, dtype=int),
            'learning_rate': NumericSpace(0.01, 0.3, log_scale=True),
            'max_depth': NumericSpace(3, 10, dtype=int),
            'min_samples_split': NumericSpace(2, 20, dtype=int),
            'min_samples_leaf': NumericSpace(1, 10, dtype=int),
            'subsample': NumericSpace(0.5, 1.0),
            'max_features': CategoricalSpace(['sqrt', 'log2', None])
        }
    )
    
    # Ridge
    spaces['Ridge'] = ModelSearchSpace(
        model_class=Ridge,
        parameters={
            'alpha': NumericSpace(0.001, 10, log_scale=True),
            'solver': CategoricalSpace(['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'])
        }
    )
    
    # Lasso
    spaces['Lasso'] = ModelSearchSpace(
        model_class=Lasso,
        parameters={
            'alpha': NumericSpace(0.001, 10, log_scale=True),
            'selection': CategoricalSpace(['cyclic', 'random'])
        }
    )
    
    # ElasticNet
    spaces['ElasticNet'] = ModelSearchSpace(
        model_class=ElasticNet,
        parameters={
            'alpha': NumericSpace(0.001, 10, log_scale=True),
            'l1_ratio': NumericSpace(0.1, 0.9),
            'selection': CategoricalSpace(['cyclic', 'random'])
        }
    )
    
    # SVR
    spaces['SVR'] = ModelSearchSpace(
        model_class=SVR,
        parameters={
            'C': NumericSpace(0.001, 100, log_scale=True),
            'kernel': CategoricalSpace(['linear', 'poly', 'rbf', 'sigmoid']),
            'gamma': CategoricalSpace(['scale', 'auto']),
            'epsilon': NumericSpace(0.01, 1, log_scale=True)
        }
    )
    
    # XGBoost
    if HAS_XGBOOST:
        spaces['XGBoost'] = ModelSearchSpace(
            model_class=XGBRegressor,
            parameters={
                'n_estimators': NumericSpace(50, 500, dtype=int),
                'max_depth': NumericSpace(3, 15, dtype=int),
                'learning_rate': NumericSpace(0.01, 0.3, log_scale=True),
                'subsample': NumericSpace(0.5, 1.0),
                'colsample_bytree': NumericSpace(0.5, 1.0)
            }
        )
    
    return spaces


def get_default_search_space(
    task_type: str,
    n_features: int,
    n_samples: int,
    include_models: Optional[List[str]] = None,
    exclude_models: Optional[List[str]] = None
) -> Dict[str, ModelSearchSpace]:
    """
    Get default search space based on data characteristics.
    
    Args:
        task_type: "classification" or "regression"
        n_features: Number of features
        n_samples: Number of samples
        include_models: Models to include
        exclude_models: Models to exclude
        
    Returns:
        Dictionary of model search spaces
    """
    if task_type == "classification":
        all_spaces = get_classification_search_spaces()
    else:
        all_spaces = get_regression_search_spaces()
    
    # Filter based on data size
    if n_samples < 100:
        # Small data - avoid complex models
        exclude_models = exclude_models or []
        exclude_models.extend(['XGBoost', 'LightGBM', 'CatBoost'])
    
    if n_features > n_samples:
        # High dimensional - prefer regularized models
        if task_type == "regression":
            priority_models = ['Ridge', 'Lasso', 'ElasticNet']
        else:
            priority_models = ['LogisticRegression', 'SVM']
    else:
        # Normal case - use ensemble methods
        priority_models = ['RandomForest', 'GradientBoosting', 'XGBoost']
    
    # Apply filters
    if include_models:
        all_spaces = {k: v for k, v in all_spaces.items() if k in include_models}
    
    if exclude_models:
        all_spaces = {k: v for k, v in all_spaces.items() if k not in exclude_models}
    
    return all_spaces