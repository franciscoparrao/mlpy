"""
Univariate feature filters for MLPY.

These filters score each feature independently based on
statistical tests or information theory.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from scipy import stats
from sklearn.feature_selection import (
    f_classif, f_regression, mutual_info_classif, 
    mutual_info_regression, chi2
)
import warnings

from .base import Filter, FilterResult, filter_registry
from ..tasks.base import Task
from ..tasks.supervised import TaskClassif
from ..tasks.supervised import TaskRegr


class FilterANOVA(Filter):
    """ANOVA F-value filter for classification.
    
    Computes ANOVA F-value between each feature and target.
    Works only with numeric features and classification tasks.
    """
    
    def __init__(self):
        super().__init__(
            id="anova",
            task_types={'classif'},
            feature_types={'numeric'}
        )
        
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        self.check_task_compatibility(task)
        
        if features is None:
            features = task.feature_names
            
        # Get data
        X = task.data(cols=features, data_format='array')
        y = task.truth()
        
        # Calculate F-values
        f_values, p_values = f_classif(X, y)
        
        # Handle NaN values
        f_values = np.nan_to_num(f_values, nan=0.0)
        
        # Create result
        scores = pd.Series(f_values, index=features)
        sorted_features = scores.sort_values(ascending=False).index.tolist()
        
        return FilterResult(
            scores=scores,
            features=sorted_features,
            method="anova",
            task_type="classif",
            params={"p_values": pd.Series(p_values, index=features)}
        )


class FilterFRegression(Filter):
    """F-statistic filter for regression.
    
    Computes F-statistic for each feature with the target.
    Works only with numeric features and regression tasks.
    """
    
    def __init__(self):
        super().__init__(
            id="f_regression",
            task_types={'regr'},
            feature_types={'numeric'}
        )
        
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        self.check_task_compatibility(task)
        
        if features is None:
            features = task.feature_names
            
        # Get data
        X = task.data(cols=features, data_format='array')
        y = task.truth()
        
        # Calculate F-values
        f_values, p_values = f_regression(X, y)
        
        # Handle NaN values
        f_values = np.nan_to_num(f_values, nan=0.0)
        
        # Create result
        scores = pd.Series(f_values, index=features)
        sorted_features = scores.sort_values(ascending=False).index.tolist()
        
        return FilterResult(
            scores=scores,
            features=sorted_features,
            method="f_regression",
            task_type="regr",
            params={"p_values": pd.Series(p_values, index=features)}
        )


class FilterMutualInformation(Filter):
    """Mutual information filter.
    
    Estimates mutual information between features and target.
    Works with both classification and regression tasks.
    
    Parameters
    ----------
    n_neighbors : int, default=3
        Number of neighbors for MI estimation.
    random_state : int, optional
        Random state for reproducibility.
    """
    
    def __init__(self, n_neighbors: int = 3, random_state: Optional[int] = None):
        super().__init__(
            id="mutual_info",
            task_types={'classif', 'regr'},
            feature_types={'numeric'}
        )
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        self.check_task_compatibility(task)
        
        if features is None:
            features = task.feature_names
            
        # Get data
        X = task.data(cols=features, data_format='array')
        y = task.truth()
        
        # Choose appropriate MI function
        if isinstance(task, TaskClassif):
            mi_func = mutual_info_classif
            task_type = "classif"
        else:
            mi_func = mutual_info_regression
            task_type = "regr"
            
        # Calculate mutual information
        mi_scores = mi_func(
            X, y, 
            n_neighbors=self.n_neighbors,
            random_state=self.random_state
        )
        
        # Create result
        scores = pd.Series(mi_scores, index=features)
        sorted_features = scores.sort_values(ascending=False).index.tolist()
        
        return FilterResult(
            scores=scores,
            features=sorted_features,
            method="mutual_info",
            task_type=task_type,
            params={"n_neighbors": self.n_neighbors}
        )


class FilterChiSquared(Filter):
    """Chi-squared filter for classification.
    
    Computes chi-squared stats between non-negative features and target.
    Features must be non-negative.
    """
    
    def __init__(self):
        super().__init__(
            id="chi2",
            task_types={'classif'},
            feature_types={'numeric'}
        )
        
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        self.check_task_compatibility(task)
        
        if features is None:
            features = task.feature_names
            
        # Get data
        X = task.data(cols=features, data_format='array')
        y = task.truth()
        
        # Check for negative values
        if np.any(X < 0):
            warnings.warn("Chi-squared requires non-negative features. Negative values will be set to 0.")
            X = np.maximum(X, 0)
            
        # Calculate chi-squared
        chi2_scores, p_values = chi2(X, y)
        
        # Handle NaN/inf values
        chi2_scores = np.nan_to_num(chi2_scores, nan=0.0, posinf=np.finfo(float).max)
        
        # Create result
        scores = pd.Series(chi2_scores, index=features)
        sorted_features = scores.sort_values(ascending=False).index.tolist()
        
        return FilterResult(
            scores=scores,
            features=sorted_features,
            method="chi2",
            task_type="classif",
            params={"p_values": pd.Series(p_values, index=features)}
        )


class FilterCorrelation(Filter):
    """Correlation filter.
    
    Computes correlation between features and target.
    Works with numeric features for both classification and regression.
    
    Parameters
    ----------
    method : str, default='pearson'
        Correlation method: 'pearson', 'spearman', 'kendall'.
    """
    
    def __init__(self, method: str = 'pearson'):
        super().__init__(
            id="correlation",
            task_types={'classif', 'regr'},
            feature_types={'numeric'}
        )
        self.method = method
        
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        self.check_task_compatibility(task)
        
        if features is None:
            features = task.feature_names
            
        # Get data as DataFrame
        df = task.data(cols=features + task.target_names, data_format='dataframe')
        target_col = task.target_names[0]
        
        # Calculate correlations
        correlations = {}
        for feature in features:
            if self.method == 'pearson':
                corr, _ = stats.pearsonr(df[feature], df[target_col])
            elif self.method == 'spearman':
                corr, _ = stats.spearmanr(df[feature], df[target_col])
            elif self.method == 'kendall':
                corr, _ = stats.kendalltau(df[feature], df[target_col])
            else:
                raise ValueError(f"Unknown correlation method: {self.method}")
                
            correlations[feature] = abs(corr)  # Use absolute correlation
            
        # Create result
        scores = pd.Series(correlations)
        scores = scores.fillna(0.0)  # Handle NaN
        sorted_features = scores.sort_values(ascending=False).index.tolist()
        
        task_type = "classif" if isinstance(task, TaskClassif) else "regr"
        
        return FilterResult(
            scores=scores,
            features=sorted_features,
            method=f"correlation_{self.method}",
            task_type=task_type,
            params={"method": self.method}
        )


class FilterVariance(Filter):
    """Variance threshold filter.
    
    Removes features with low variance.
    Useful for removing constant or quasi-constant features.
    
    Parameters
    ----------
    threshold : float, default=0.0
        Features with variance below this are removed.
    """
    
    def __init__(self, threshold: float = 0.0):
        super().__init__(
            id="variance",
            task_types={'any'},
            feature_types={'numeric'}
        )
        self.threshold = threshold
        
    def calculate(self, task: Task, features: Optional[List[str]] = None) -> FilterResult:
        if features is None:
            features = task.feature_names
            
        # Get numeric features only
        data = task.data(cols=features, data_format='dataframe')
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_features:
            raise ValueError("No numeric features found")
            
        # Calculate variance
        variances = data[numeric_features].var()
        
        # Create result
        scores = pd.Series(variances, index=numeric_features)
        sorted_features = scores.sort_values(ascending=False).index.tolist()
        
        task_type = "classif" if isinstance(task, TaskClassif) else "regr"
        
        return FilterResult(
            scores=scores,
            features=sorted_features,
            method="variance",
            task_type=task_type,
            params={"threshold": self.threshold}
        )


# Register all filters
filter_registry.register(FilterANOVA, "anova")
filter_registry.register(FilterFRegression, "f_regression")
filter_registry.register(FilterMutualInformation, "mutual_info")
filter_registry.register(FilterChiSquared, "chi2")
filter_registry.register(FilterCorrelation, "correlation")
filter_registry.register(FilterVariance, "variance")