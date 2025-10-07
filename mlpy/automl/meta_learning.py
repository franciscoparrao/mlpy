"""
Meta-learning for MLPY AutoML.

This module implements meta-learning capabilities to automatically select
the best algorithms and hyperparameters based on dataset characteristics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
from scipy.stats import skew, kurtosis

from ..tasks import Task, TaskClassif, TaskRegr


@dataclass
class DatasetCharacteristics:
    """Dataset characteristics for meta-learning."""
    # Basic characteristics
    n_samples: int
    n_features: int
    n_numeric: int
    n_categorical: int
    
    # Task-specific
    task_type: str
    n_classes: Optional[int] = None
    class_balance: Optional[float] = None  # Entropy for classification
    
    # Statistical properties
    missing_ratio: float = 0.0
    numeric_skewness: float = 0.0  # Average skewness of numeric features
    numeric_kurtosis: float = 0.0  # Average kurtosis of numeric features
    feature_correlation: float = 0.0  # Average absolute correlation
    
    # Complexity measures
    dimensionality_ratio: float = 0.0  # n_features / n_samples
    categorical_ratio: float = 0.0     # n_categorical / n_features
    
    # Size categories
    size_category: str = "medium"  # small, medium, large
    complexity_category: str = "moderate"  # simple, moderate, complex


class MetaLearner:
    """
    Meta-learner for algorithm recommendation.
    
    Uses dataset characteristics to recommend algorithms and configurations
    based on empirical performance patterns.
    """
    
    def __init__(self):
        """Initialize meta-learner with algorithm recommendations."""
        self._initialize_algorithm_preferences()
        self._initialize_size_thresholds()
    
    def _initialize_algorithm_preferences(self):
        """Initialize algorithm preferences based on dataset characteristics."""
        
        # Classification algorithm preferences
        self.classification_preferences = {
            # Format: condition -> (algorithms, priority_weight)
            "small_datasets": {
                "condition": lambda chars: chars.n_samples < 1000,
                "algorithms": [
                    ("LogisticRegression", 0.9),
                    ("KNeighborsClassifier", 0.8),
                    ("DecisionTreeClassifier", 0.7),
                    ("GaussianNB", 0.6),
                    ("SVC", 0.5)
                ]
            },
            "high_dimensional": {
                "condition": lambda chars: chars.dimensionality_ratio > 0.1,
                "algorithms": [
                    ("LogisticRegression", 0.9),
                    ("RandomForestClassifier", 0.8),
                    ("LinearDiscriminantAnalysis", 0.7),
                    ("SGDClassifier", 0.6)
                ]
            },
            "many_classes": {
                "condition": lambda chars: chars.n_classes and chars.n_classes > 10,
                "algorithms": [
                    ("RandomForestClassifier", 0.9),
                    ("GradientBoostingClassifier", 0.8),
                    ("ExtraTreesClassifier", 0.7),
                    ("LogisticRegression", 0.6)
                ]
            },
            "imbalanced_classes": {
                "condition": lambda chars: chars.class_balance and chars.class_balance < 0.7,
                "algorithms": [
                    ("RandomForestClassifier", 0.9),
                    ("GradientBoostingClassifier", 0.8),
                    ("SVC", 0.7),
                    ("AdaBoostClassifier", 0.6)
                ]
            },
            "large_datasets": {
                "condition": lambda chars: chars.n_samples > 50000,
                "algorithms": [
                    ("SGDClassifier", 0.9),
                    ("LogisticRegression", 0.8),
                    ("RandomForestClassifier", 0.7),
                    ("GradientBoostingClassifier", 0.6)
                ]
            },
            "categorical_heavy": {
                "condition": lambda chars: chars.categorical_ratio > 0.5,
                "algorithms": [
                    ("DecisionTreeClassifier", 0.9),
                    ("RandomForestClassifier", 0.8),
                    ("GradientBoostingClassifier", 0.7),
                    ("ExtraTreesClassifier", 0.6)
                ]
            },
            "default": {
                "condition": lambda chars: True,  # Always applies
                "algorithms": [
                    ("RandomForestClassifier", 0.8),
                    ("LogisticRegression", 0.7),
                    ("GradientBoostingClassifier", 0.6),
                    ("SVC", 0.5),
                    ("KNeighborsClassifier", 0.4)
                ]
            }
        }
        
        # Regression algorithm preferences  
        self.regression_preferences = {
            "small_datasets": {
                "condition": lambda chars: chars.n_samples < 1000,
                "algorithms": [
                    ("LinearRegression", 0.9),
                    ("KNeighborsRegressor", 0.8),
                    ("DecisionTreeRegressor", 0.7),
                    ("Ridge", 0.6),
                    ("SVR", 0.5)
                ]
            },
            "high_dimensional": {
                "condition": lambda chars: chars.dimensionality_ratio > 0.1,
                "algorithms": [
                    ("Ridge", 0.9),
                    ("Lasso", 0.8),
                    ("ElasticNet", 0.7),
                    ("LinearRegression", 0.6)
                ]
            },
            "large_datasets": {
                "condition": lambda chars: chars.n_samples > 50000,
                "algorithms": [
                    ("SGDRegressor", 0.9),
                    ("LinearRegression", 0.8),
                    ("RandomForestRegressor", 0.7),
                    ("GradientBoostingRegressor", 0.6)
                ]
            },
            "non_linear": {
                "condition": lambda chars: chars.feature_correlation < 0.3,
                "algorithms": [
                    ("RandomForestRegressor", 0.9),
                    ("GradientBoostingRegressor", 0.8),
                    ("SVR", 0.7),
                    ("KNeighborsRegressor", 0.6)
                ]
            },
            "categorical_heavy": {
                "condition": lambda chars: chars.categorical_ratio > 0.5,
                "algorithms": [
                    ("DecisionTreeRegressor", 0.9),
                    ("RandomForestRegressor", 0.8),
                    ("GradientBoostingRegressor", 0.7),
                    ("ExtraTreesRegressor", 0.6)
                ]
            },
            "default": {
                "condition": lambda chars: True,
                "algorithms": [
                    ("RandomForestRegressor", 0.8),
                    ("LinearRegression", 0.7),
                    ("GradientBoostingRegressor", 0.6),
                    ("Ridge", 0.5),
                    ("SVR", 0.4)
                ]
            }
        }
        
    def _initialize_size_thresholds(self):
        """Initialize thresholds for dataset size categories."""
        self.size_thresholds = {
            "small": (0, 5000),
            "medium": (5000, 100000),
            "large": (100000, float('inf'))
        }
        
        self.complexity_thresholds = {
            "simple": 0.01,    # dimensionality_ratio
            "moderate": 0.1,
            "complex": float('inf')
        }
    
    def extract_characteristics(self, task: Task) -> DatasetCharacteristics:
        """
        Extract meta-learning characteristics from a task.
        
        Parameters
        ----------
        task : Task
            The task to analyze
            
        Returns
        -------
        DatasetCharacteristics
            Extracted characteristics
        """
        # Basic statistics
        n_samples = task.nrow
        n_features = len(task.feature_names)
        
        # Get data for analysis
        data = task.data()
        feature_data = data[task.feature_names]
        
        # Count feature types
        col_info = task._backend.col_info()
        n_numeric = sum(1 for col in task.feature_names 
                       if col_info[col]["type"] in ("numeric", "integer"))
        n_categorical = n_features - n_numeric
        
        # Missing values
        missing_ratio = feature_data.isnull().sum().sum() / (n_samples * n_features)
        
        # Statistical properties of numeric features
        numeric_features = [col for col in task.feature_names 
                          if col_info[col]["type"] in ("numeric", "integer")]
        
        numeric_skewness = 0.0
        numeric_kurtosis = 0.0
        feature_correlation = 0.0
        
        if numeric_features:
            numeric_data = feature_data[numeric_features].select_dtypes(include=[np.number])
            
            if not numeric_data.empty:
                # Skewness and kurtosis
                skewness_values = []
                kurtosis_values = []
                
                for col in numeric_data.columns:
                    col_data = numeric_data[col].dropna()
                    if len(col_data) > 1 and col_data.var() > 0:
                        skewness_values.append(abs(skew(col_data)))
                        kurtosis_values.append(abs(kurtosis(col_data)))
                
                if skewness_values:
                    numeric_skewness = np.mean(skewness_values)
                if kurtosis_values:
                    numeric_kurtosis = np.mean(kurtosis_values)
                
                # Feature correlation
                if len(numeric_data.columns) > 1:
                    corr_matrix = numeric_data.corr().abs()
                    # Average correlation (excluding diagonal)
                    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                    if mask.any():
                        feature_correlation = corr_matrix.values[mask].mean()
        
        # Task-specific characteristics
        task_type = task.task_type
        n_classes = None
        class_balance = None
        
        if isinstance(task, TaskClassif):
            n_classes = task.n_classes
            
            # Calculate class balance using entropy
            if hasattr(task, 'class_names') and task.target_names:
                target_data = data[task.target_names[0]]
                value_counts = target_data.value_counts()
                if len(value_counts) > 1:
                    proportions = value_counts / len(target_data)
                    entropy = -np.sum(proportions * np.log2(proportions))
                    max_entropy = np.log2(len(proportions))
                    class_balance = entropy / max_entropy if max_entropy > 0 else 1.0
        
        # Derived measures
        dimensionality_ratio = n_features / n_samples if n_samples > 0 else 0
        categorical_ratio = n_categorical / n_features if n_features > 0 else 0
        
        # Categories
        size_category = "medium"
        for category, (min_size, max_size) in self.size_thresholds.items():
            if min_size <= n_samples < max_size:
                size_category = category
                break
        
        complexity_category = "moderate"
        if dimensionality_ratio <= self.complexity_thresholds["simple"]:
            complexity_category = "simple"
        elif dimensionality_ratio >= self.complexity_thresholds["moderate"]:
            complexity_category = "complex"
        
        return DatasetCharacteristics(
            n_samples=n_samples,
            n_features=n_features,
            n_numeric=n_numeric,
            n_categorical=n_categorical,
            task_type=task_type,
            n_classes=n_classes,
            class_balance=class_balance,
            missing_ratio=missing_ratio,
            numeric_skewness=numeric_skewness,
            numeric_kurtosis=numeric_kurtosis,
            feature_correlation=feature_correlation,
            dimensionality_ratio=dimensionality_ratio,
            categorical_ratio=categorical_ratio,
            size_category=size_category,
            complexity_category=complexity_category
        )
    
    def recommend_algorithms(self, 
                           characteristics: DatasetCharacteristics, 
                           max_algorithms: int = 5) -> List[Tuple[str, float]]:
        """
        Recommend algorithms based on dataset characteristics.
        
        Parameters
        ----------
        characteristics : DatasetCharacteristics
            Dataset characteristics
        max_algorithms : int
            Maximum number of algorithms to recommend
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (algorithm_name, priority_score) tuples
        """
        # Choose preference set based on task type
        if characteristics.task_type == "classif":
            preferences = self.classification_preferences
        elif characteristics.task_type == "regr":
            preferences = self.regression_preferences
        else:
            warnings.warn(f"Unknown task type: {characteristics.task_type}")
            return []
        
        # Collect algorithm scores from all matching conditions
        algorithm_scores = {}
        
        for condition_name, preference in preferences.items():
            condition = preference["condition"]
            algorithms = preference["algorithms"]
            
            # Check if condition applies
            if condition(characteristics):
                for algorithm, weight in algorithms:
                    if algorithm not in algorithm_scores:
                        algorithm_scores[algorithm] = 0.0
                    algorithm_scores[algorithm] += weight
        
        # Sort algorithms by score and return top ones
        sorted_algorithms = sorted(algorithm_scores.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        return sorted_algorithms[:max_algorithms]
    
    def recommend_preprocessing(self, 
                              characteristics: DatasetCharacteristics) -> Dict[str, Any]:
        """
        Recommend preprocessing steps based on characteristics.
        
        Parameters
        ----------
        characteristics : DatasetCharacteristics
            Dataset characteristics
            
        Returns
        -------
        Dict[str, Any]
            Preprocessing recommendations
        """
        recommendations = {
            "scaling": False,
            "imputation": False,
            "feature_selection": False,
            "feature_engineering": False,
            "categorical_encoding": False
        }
        
        # Scaling recommendations
        if characteristics.n_numeric > 0:
            # Scale if numeric features have different ranges or high variance
            if (characteristics.numeric_skewness > 1.0 or 
                characteristics.numeric_kurtosis > 3.0):
                recommendations["scaling"] = True
        
        # Imputation recommendations
        if characteristics.missing_ratio > 0.01:  # >1% missing
            recommendations["imputation"] = True
        
        # Feature selection recommendations
        if (characteristics.dimensionality_ratio > 0.1 or 
            characteristics.n_features > 100):
            recommendations["feature_selection"] = True
        
        # Feature engineering recommendations
        if (characteristics.size_category in ["medium", "large"] and
            characteristics.n_numeric >= 2):
            recommendations["feature_engineering"] = True
        
        # Categorical encoding recommendations
        if characteristics.n_categorical > 0:
            recommendations["categorical_encoding"] = True
        
        return recommendations
    
    def recommend_cv_strategy(self, 
                            characteristics: DatasetCharacteristics) -> Dict[str, Any]:
        """
        Recommend cross-validation strategy.
        
        Parameters
        ----------
        characteristics : DatasetCharacteristics
            Dataset characteristics
            
        Returns
        -------
        Dict[str, Any]
            CV strategy recommendations
        """
        # Default strategy
        strategy = {
            "method": "cv",
            "folds": 5,
            "stratify": False,
            "shuffle": True
        }
        
        # Adjust based on dataset size
        if characteristics.n_samples < 500:
            strategy["method"] = "cv"
            strategy["folds"] = 3  # Fewer folds for small datasets
        elif characteristics.n_samples > 50000:
            strategy["method"] = "holdout"
            strategy["ratio"] = 0.2  # Holdout for large datasets
        else:
            strategy["folds"] = 5
        
        # Stratification for classification
        if (characteristics.task_type == "classif" and 
            characteristics.n_classes and characteristics.n_classes <= 10):
            strategy["stratify"] = True
        
        # Special handling for imbalanced classes
        if (characteristics.class_balance and 
            characteristics.class_balance < 0.7):
            strategy["stratify"] = True
            strategy["folds"] = min(strategy.get("folds", 5), 3)
        
        return strategy
    
    def get_meta_summary(self, 
                        characteristics: DatasetCharacteristics) -> Dict[str, Any]:
        """
        Get a comprehensive meta-learning summary.
        
        Parameters
        ----------
        characteristics : DatasetCharacteristics
            Dataset characteristics
            
        Returns
        -------
        Dict[str, Any]
            Complete meta-learning recommendations
        """
        return {
            "characteristics": characteristics,
            "algorithms": self.recommend_algorithms(characteristics),
            "preprocessing": self.recommend_preprocessing(characteristics),
            "cv_strategy": self.recommend_cv_strategy(characteristics),
            "insights": self._generate_insights(characteristics)
        }
    
    def _generate_insights(self, 
                          characteristics: DatasetCharacteristics) -> List[str]:
        """Generate human-readable insights about the dataset."""
        insights = []
        
        # Size insights
        insights.append(f"Dataset size: {characteristics.size_category} "
                       f"({characteristics.n_samples:,} samples)")
        
        # Complexity insights  
        insights.append(f"Complexity: {characteristics.complexity_category} "
                       f"({characteristics.n_features} features)")
        
        # Feature type insights
        if characteristics.n_categorical > 0:
            insights.append(f"Contains {characteristics.n_categorical} "
                           f"categorical features ({characteristics.categorical_ratio:.1%})")
        
        # Missing data insights
        if characteristics.missing_ratio > 0.05:
            insights.append(f"Significant missing data: {characteristics.missing_ratio:.1%}")
        
        # Class balance insights (classification)
        if characteristics.task_type == "classif":
            if characteristics.class_balance and characteristics.class_balance < 0.7:
                insights.append("Classes appear imbalanced")
            if characteristics.n_classes and characteristics.n_classes > 10:
                insights.append(f"Many classes ({characteristics.n_classes})")
        
        # Statistical insights
        if characteristics.numeric_skewness > 2.0:
            insights.append("Highly skewed numeric features detected")
        
        if characteristics.feature_correlation > 0.7:
            insights.append("High feature correlation detected")
        
        return insights


__all__ = ["MetaLearner", "DatasetCharacteristics"]