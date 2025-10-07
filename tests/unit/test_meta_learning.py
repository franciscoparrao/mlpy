"""
Tests for meta-learning functionality.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.automl.meta_learning import MetaLearner, DatasetCharacteristics


class TestMetaLearner:
    """Test MetaLearner functionality."""
    
    @pytest.fixture
    def small_classif_task(self):
        """Small classification task."""
        X, y = make_classification(
            n_samples=500, 
            n_features=10, 
            n_classes=3,
            n_informative=5,  # Increase informative features
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df['target'] = y
        return TaskClassif(df, target='target')
    
    @pytest.fixture
    def large_regr_task(self):
        """Large regression task."""
        X, y = make_regression(
            n_samples=80000,
            n_features=50,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(50)])
        df['target'] = y
        return TaskRegr(df, target='target')
    
    @pytest.fixture
    def mixed_data_task(self):
        """Task with mixed data types."""
        np.random.seed(42)
        n_samples = 1000
        
        # Numeric features
        X_numeric = np.random.randn(n_samples, 5)
        
        # Categorical features
        categories = ['A', 'B', 'C', 'D']
        X_categorical = np.random.choice(categories, (n_samples, 3))
        
        # Create DataFrame
        df = pd.DataFrame(X_numeric, columns=[f'num_{i}' for i in range(5)])
        for i in range(3):
            df[f'cat_{i}'] = X_categorical[:, i]
        
        # Target
        df['target'] = np.random.choice([0, 1], n_samples)
        
        return TaskClassif(df, target='target')
    
    def test_meta_learner_initialization(self):
        """Test MetaLearner initialization."""
        meta_learner = MetaLearner()
        
        assert hasattr(meta_learner, 'classification_preferences')
        assert hasattr(meta_learner, 'regression_preferences')
        assert 'default' in meta_learner.classification_preferences
        assert 'default' in meta_learner.regression_preferences
    
    def test_extract_characteristics_classification(self, small_classif_task):
        """Test characteristic extraction for classification."""
        meta_learner = MetaLearner()
        characteristics = meta_learner.extract_characteristics(small_classif_task)
        
        assert isinstance(characteristics, DatasetCharacteristics)
        assert characteristics.n_samples == 500
        assert characteristics.n_features == 10
        assert characteristics.task_type == "classif"
        assert characteristics.n_classes == 3
        assert characteristics.n_numeric == 10
        assert characteristics.n_categorical == 0
        assert characteristics.size_category == "small"
    
    def test_extract_characteristics_regression(self, large_regr_task):
        """Test characteristic extraction for regression."""
        meta_learner = MetaLearner()
        characteristics = meta_learner.extract_characteristics(large_regr_task)
        
        assert characteristics.n_samples == 80000
        assert characteristics.n_features == 50
        assert characteristics.task_type == "regr"
        assert characteristics.n_classes is None
        assert characteristics.size_category == "medium"  # 80000 is in medium range
        assert characteristics.complexity_category == "simple"  # 50/80000 < 0.01
    
    def test_extract_characteristics_mixed_data(self, mixed_data_task):
        """Test characteristic extraction with mixed data."""
        meta_learner = MetaLearner()
        characteristics = meta_learner.extract_characteristics(mixed_data_task)
        
        assert characteristics.n_numeric == 5
        assert characteristics.n_categorical == 3
        assert characteristics.categorical_ratio == 3/8  # 3 out of 8 features
        assert 0 <= characteristics.missing_ratio <= 1
        assert characteristics.size_category == "small"
    
    def test_recommend_algorithms_classification(self, small_classif_task):
        """Test algorithm recommendations for classification."""
        meta_learner = MetaLearner()
        characteristics = meta_learner.extract_characteristics(small_classif_task)
        
        algorithms = meta_learner.recommend_algorithms(characteristics, max_algorithms=5)
        
        assert len(algorithms) <= 5
        assert all(isinstance(alg, str) and isinstance(score, float) for alg, score in algorithms)
        assert all(score > 0 for _, score in algorithms)
        
        # Should be sorted by score (highest first)
        scores = [score for _, score in algorithms]
        assert scores == sorted(scores, reverse=True)
    
    def test_recommend_algorithms_regression(self, large_regr_task):
        """Test algorithm recommendations for regression."""
        meta_learner = MetaLearner()
        characteristics = meta_learner.extract_characteristics(large_regr_task)
        
        algorithms = meta_learner.recommend_algorithms(characteristics, max_algorithms=3)
        
        assert len(algorithms) <= 3
        # For large datasets, should recommend scalable algorithms
        algorithm_names = [alg for alg, _ in algorithms]
        scalable_algorithms = ['SGDRegressor', 'LinearRegression', 'RandomForestRegressor']
        assert any(alg in algorithm_names for alg in scalable_algorithms)
    
    def test_recommend_preprocessing(self, mixed_data_task):
        """Test preprocessing recommendations."""
        meta_learner = MetaLearner()
        characteristics = meta_learner.extract_characteristics(mixed_data_task)
        
        preprocessing = meta_learner.recommend_preprocessing(characteristics)
        
        assert isinstance(preprocessing, dict)
        required_keys = ['scaling', 'imputation', 'feature_selection', 
                        'feature_engineering', 'categorical_encoding']
        assert all(key in preprocessing for key in required_keys)
        assert all(isinstance(preprocessing[key], bool) for key in required_keys)
        
        # Should recommend categorical encoding for mixed data
        assert preprocessing['categorical_encoding'] == True
    
    def test_recommend_cv_strategy(self, small_classif_task):
        """Test CV strategy recommendations."""
        meta_learner = MetaLearner()
        characteristics = meta_learner.extract_characteristics(small_classif_task)
        
        cv_strategy = meta_learner.recommend_cv_strategy(characteristics)
        
        assert isinstance(cv_strategy, dict)
        assert 'method' in cv_strategy
        assert cv_strategy['method'] in ['cv', 'holdout']
        
        if cv_strategy['method'] == 'cv':
            assert 'folds' in cv_strategy
            assert cv_strategy['folds'] >= 3
        else:
            assert 'ratio' in cv_strategy
            assert 0 < cv_strategy['ratio'] < 1
    
    def test_get_meta_summary(self, small_classif_task):
        """Test comprehensive meta-learning summary."""
        meta_learner = MetaLearner()
        characteristics = meta_learner.extract_characteristics(small_classif_task)
        
        summary = meta_learner.get_meta_summary(characteristics)
        
        required_keys = ['characteristics', 'algorithms', 'preprocessing', 
                        'cv_strategy', 'insights']
        assert all(key in summary for key in required_keys)
        
        # Check types
        assert isinstance(summary['characteristics'], DatasetCharacteristics)
        assert isinstance(summary['algorithms'], list)
        assert isinstance(summary['preprocessing'], dict)
        assert isinstance(summary['cv_strategy'], dict)
        assert isinstance(summary['insights'], list)
        
        # Check insights are strings
        assert all(isinstance(insight, str) for insight in summary['insights'])
    
    def test_size_categorization(self):
        """Test dataset size categorization."""
        meta_learner = MetaLearner()
        
        # Test different sizes
        test_cases = [
            (100, "small"),
            (3000, "small"),
            (10000, "medium"),
            (50000, "medium"),
            (150000, "large")
        ]
        
        for n_samples, expected_category in test_cases:
            X, y = make_classification(n_samples=n_samples, n_features=5, random_state=42)
            df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(5)])
            df['target'] = y
            task = TaskClassif(df, target='target')
            
            characteristics = meta_learner.extract_characteristics(task)
            assert characteristics.size_category == expected_category
    
    def test_complexity_categorization(self):
        """Test dataset complexity categorization."""
        meta_learner = MetaLearner()
        
        # High dimensionality (features >> samples)
        X, y = make_classification(n_samples=100, n_features=50, random_state=42)
        df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(50)])
        df['target'] = y
        task = TaskClassif(df, target='target')
        
        characteristics = meta_learner.extract_characteristics(task)
        assert characteristics.complexity_category == "complex"
        assert characteristics.dimensionality_ratio == 0.5  # 50/100
    
    def test_class_balance_detection(self):
        """Test class balance detection."""
        meta_learner = MetaLearner()
        
        # Create imbalanced dataset
        X, _ = make_classification(n_samples=1000, n_features=10, random_state=42)
        df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(10)])
        
        # Highly imbalanced: 90% class 0, 10% class 1
        y_imbalanced = np.concatenate([np.zeros(900), np.ones(100)])
        df['target'] = y_imbalanced
        
        task = TaskClassif(df, target='target')
        characteristics = meta_learner.extract_characteristics(task)
        
        # Class balance should be low (high imbalance)
        assert characteristics.class_balance is not None
        assert characteristics.class_balance < 0.8  # Should detect imbalance
    
    def test_missing_data_detection(self):
        """Test missing data detection."""
        meta_learner = MetaLearner()
        
        # Create data with missing values
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(10)])
        df['target'] = y
        
        # Add missing values to some features
        df.loc[:100, 'f_0'] = np.nan  # ~10% missing in first feature
        df.loc[:50, 'f_1'] = np.nan   # ~5% missing in second feature
        
        task = TaskClassif(df, target='target')
        characteristics = meta_learner.extract_characteristics(task)
        
        # Should detect missing values
        assert characteristics.missing_ratio > 0
        # Approximately (100 + 50) / (1000 * 10) = 0.015
        assert 0.01 < characteristics.missing_ratio < 0.02
    
    def test_algorithm_priority_scoring(self, small_classif_task):
        """Test that algorithm priority scoring works correctly."""
        meta_learner = MetaLearner()
        characteristics = meta_learner.extract_characteristics(small_classif_task)
        
        # Get recommendations
        algorithms = meta_learner.recommend_algorithms(characteristics, max_algorithms=10)
        
        # Verify conditions are being applied correctly
        # For small datasets, certain algorithms should get high priority
        algorithm_names = [alg for alg, _ in algorithms]
        
        # Should include algorithms good for small datasets
        small_dataset_algorithms = ['LogisticRegression', 'KNeighborsClassifier', 'DecisionTreeClassifier']
        assert any(alg in algorithm_names for alg in small_dataset_algorithms)
    
    def test_unknown_task_type_warning(self):
        """Test warning for unknown task types."""
        meta_learner = MetaLearner()
        
        # Create characteristics with unknown task type
        characteristics = DatasetCharacteristics(
            n_samples=100,
            n_features=5,
            n_numeric=5,
            n_categorical=0,
            task_type="unknown_type"
        )
        
        with pytest.warns(UserWarning, match="Unknown task type"):
            algorithms = meta_learner.recommend_algorithms(characteristics)
            assert algorithms == []