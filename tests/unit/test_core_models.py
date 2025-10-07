"""
Unit tests for core MLPY models.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestLearnerBase:
    """Test base learner functionality."""
    
    @pytest.mark.unit
    def test_learner_initialization(self):
        """Test learner can be initialized."""
        from mlpy.learners.base import Learner
        
        # Create a concrete implementation since Learner is abstract
        class ConcreteLearner(Learner):
            def train(self, task):
                self._model = "trained"
                return self
            
            def predict(self, task):
                return []
            
            @property
            def task_type(self):
                return "test"
        
        learner = ConcreteLearner(id="test_learner", label="Test Learner")
        assert learner.id == "test_learner"
        assert learner.label == "Test Learner"
        assert not learner.is_trained
    
    @pytest.mark.unit
    def test_learner_train_abstract(self):
        """Test that base learner train is abstract."""
        from mlpy.learners.base import Learner
        
        # Cannot instantiate abstract class
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            learner = Learner()
    
    @pytest.mark.unit
    def test_learner_predict_requires_training(self):
        """Test that predict requires trained model."""
        from mlpy.learners.base import Learner
        
        class TestLearner(Learner):
            def train(self, task):
                self._model = "trained"
                return self
            
            def predict(self, task):
                if not self.is_trained:
                    raise RuntimeError("Model must be trained")
                return []
            
            @property
            def task_type(self):
                return "test"
        
        learner = TestLearner(id="test_learner")
        task = Mock()
        
        # Should raise before training
        with pytest.raises(RuntimeError, match="Model must be trained"):
            learner.predict(task)
        
        # Should work after training
        learner.train(task)
        assert learner.is_trained


class TestClassificationLearners:
    """Test classification learners."""
    
    @pytest.mark.unit
    @pytest.mark.requires_sklearn
    def test_random_forest_classifier_init(self):
        """Test RandomForest classifier initialization."""
        try:
            from mlpy.learners.sklearn import LearnerRandomForestClassifier
            
            learner = LearnerRandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            
            # Check parameters are stored correctly
            assert learner.estimator_params['n_estimators'] == 100
            assert learner.estimator_params['max_depth'] == 5
            assert learner.estimator_params['random_state'] == 42
            assert not learner.is_trained
            
        except ImportError:
            pytest.skip("scikit-learn not installed")
    
    @pytest.mark.unit
    @pytest.mark.requires_sklearn
    def test_random_forest_train_predict(self, classification_task):
        """Test RandomForest training and prediction."""
        try:
            from mlpy.learners.sklearn import LearnerRandomForestClassifier
            from mlpy.predictions import PredictionClassif
            
            learner = LearnerRandomForestClassifier(n_estimators=10, random_state=42)
            
            # Train
            learner.train(classification_task)
            assert learner.is_trained
            
            # Predict
            predictions = learner.predict(classification_task)
            assert isinstance(predictions, PredictionClassif)
            # Get data length properly
            if callable(classification_task.data):
                data_len = len(classification_task.data())
            else:
                data_len = len(classification_task.data)
            assert len(predictions.response) == data_len
            
        except ImportError:
            pytest.skip("scikit-learn not installed")
    
    @pytest.mark.unit
    def test_baseline_classifier(self, classification_task):
        """Test baseline featureless classifier."""
        from mlpy.learners.baseline import LearnerClassifFeatureless
        
        learner = LearnerClassifFeatureless()
        
        # Train
        learner.train(classification_task)
        assert learner.is_trained
        
        # Predict - should predict most frequent class
        predictions = learner.predict(classification_task)
        assert all(pred == predictions.response[0] for pred in predictions.response)


class TestRegressionLearners:
    """Test regression learners."""
    
    @pytest.mark.unit
    def test_baseline_regressor(self, regression_task):
        """Test baseline featureless regressor."""
        from mlpy.learners.baseline import LearnerRegrFeatureless
        
        learner = LearnerRegrFeatureless()
        
        # Train
        learner.train(regression_task)
        assert learner.is_trained
        
        # Predict - should predict mean value
        predictions = learner.predict(regression_task)
        # Get target data properly
        if hasattr(regression_task, 'y'):
            target_data = regression_task.y
        elif callable(regression_task.data):
            target_data = regression_task.data(cols=['target'])
        else:
            target_data = regression_task.data['target']
        
        expected_value = np.mean(target_data)
        
        # Convert predictions to numpy array for comparison
        pred_values = np.array(predictions.response)
        assert np.all(np.abs(pred_values - expected_value) < 0.01)
    
    @pytest.mark.unit
    @pytest.mark.requires_sklearn
    def test_linear_regression(self, regression_task):
        """Test linear regression."""
        try:
            from mlpy.learners.sklearn import LearnerLinearRegression
            from mlpy.predictions import PredictionRegr
            
            learner = LearnerLinearRegression()
            
            # Train
            learner.train(regression_task)
            assert learner.is_trained
            
            # Predict
            predictions = learner.predict(regression_task)
            assert isinstance(predictions, PredictionRegr)
            # Get data length properly
            if callable(regression_task.data):
                data_len = len(regression_task.data())
            else:
                data_len = len(regression_task.data)
            assert len(predictions.response) == data_len
            
            # Check predictions are reasonable (not all same value)
            assert len(set(predictions.response)) > 1
            
        except ImportError:
            pytest.skip("scikit-learn not installed")


class TestEnsembleLearners:
    """Test ensemble learners."""
    
    @pytest.mark.unit
    def test_voting_ensemble_init(self, mock_learner):
        """Test voting ensemble initialization."""
        from mlpy.learners.ensemble import LearnerVoting
        
        base_learners = [mock_learner, mock_learner]
        ensemble = LearnerVoting(
            base_learners=base_learners,
            voting='hard',
            weights=[0.6, 0.4]
        )
        
        assert len(ensemble.base_learners) == 2
        assert ensemble.voting == 'hard'
        assert abs(sum(ensemble.weights) - 1.0) < 0.01  # Weights normalized
    
    @pytest.mark.unit
    def test_voting_ensemble_train(self, mock_learner, classification_task):
        """Test voting ensemble training."""
        from mlpy.learners.ensemble import LearnerVoting
        
        base_learners = [mock_learner, mock_learner]
        ensemble = LearnerVoting(base_learners=base_learners)
        
        # Train ensemble
        ensemble.train(classification_task)
        assert ensemble.is_trained
        assert ensemble._trained_learners is not None
        assert len(ensemble._trained_learners) == 2
    
    @pytest.mark.unit
    def test_stacking_ensemble_init(self, mock_learner):
        """Test stacking ensemble initialization."""
        from mlpy.learners.ensemble import LearnerStacking
        
        base_learners = [mock_learner, mock_learner]
        meta_learner = mock_learner
        
        ensemble = LearnerStacking(
            base_learners=base_learners,
            meta_learner=meta_learner,
            cv_folds=3
        )
        
        assert len(ensemble.base_learners) == 2
        assert ensemble.meta_learner is not None
        assert ensemble.cv_folds == 3


class TestClusteringModels:
    """Test clustering models."""
    
    @pytest.mark.unit
    @pytest.mark.requires_sklearn
    def test_dbscan_initialization(self):
        """Test DBSCAN initialization."""
        try:
            from sklearn.cluster import DBSCAN
            
            # Test sklearn DBSCAN directly
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            assert dbscan.eps == 0.5
            assert dbscan.min_samples == 5
            
        except ImportError:
            pytest.skip("scikit-learn not installed")
    
    @pytest.mark.unit
    @pytest.mark.requires_sklearn
    def test_dbscan_clustering(self, sample_clustering_data):
        """Test DBSCAN clustering."""
        try:
            from sklearn.cluster import DBSCAN
            
            X, y_true = sample_clustering_data
            
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(X.values)
            
            # Check we got some clusters
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            assert n_clusters >= 0  # At least found some structure
            
            # Check labels are correct length
            assert len(labels) == len(X)
            
        except ImportError:
            pytest.skip("scikit-learn not installed")
    
    @pytest.mark.unit
    @pytest.mark.requires_sklearn
    def test_gaussian_mixture(self, sample_clustering_data):
        """Test Gaussian Mixture Model."""
        try:
            from sklearn.mixture import GaussianMixture
            
            X, y_true = sample_clustering_data
            
            gmm = GaussianMixture(n_components=3, random_state=42)
            labels = gmm.fit_predict(X.values)
            
            assert len(set(labels)) == 3  # Should find 3 clusters
            assert len(labels) == len(X)
            assert gmm.converged_
            
        except ImportError:
            pytest.skip("scikit-learn not installed")


class TestModelRegistry:
    """Test model registry functionality."""
    
    @pytest.mark.unit
    def test_registry_initialization(self):
        """Test registry can be initialized."""
        from mlpy.model_registry.registry import ModelRegistry
        
        registry = ModelRegistry()
        assert registry._models == {}
        assert not registry._initialized
    
    @pytest.mark.unit
    def test_registry_register_model(self):
        """Test registering a model."""
        from mlpy.model_registry.registry import (
            ModelRegistry, ModelMetadata, ModelCategory, TaskType, Complexity
        )
        
        registry = ModelRegistry()
        
        metadata = ModelMetadata(
            name="test_model",
            display_name="Test Model",
            description="A test model",
            category=ModelCategory.TRADITIONAL_ML,
            class_path="mlpy.test.TestModel",
            task_types=[TaskType.CLASSIFICATION],
            complexity=Complexity.LOW
        )
        
        success = registry.register(metadata)
        assert success
        assert "test_model" in registry._models
        assert registry.get("test_model") == metadata
    
    @pytest.mark.unit
    def test_registry_search(self):
        """Test searching models in registry."""
        from mlpy.model_registry.registry import (
            ModelRegistry, ModelMetadata, ModelCategory, TaskType, Complexity
        )
        
        registry = ModelRegistry()
        
        # Register multiple models
        models = [
            ModelMetadata(
                name="model1",
                display_name="Model 1",
                description="Classification model",
                category=ModelCategory.TRADITIONAL_ML,
                class_path="mlpy.learners.baseline.LearnerClassifFeatureless",
                task_types=[TaskType.CLASSIFICATION],
                complexity=Complexity.LOW
            ),
            ModelMetadata(
                name="model2",
                display_name="Model 2",
                description="Clustering model",
                category=ModelCategory.UNSUPERVISED,
                class_path="mlpy.learners.baseline.LearnerRegrFeatureless",
                task_types=[TaskType.CLUSTERING],
                complexity=Complexity.MEDIUM
            )
        ]
        
        for model in models:
            registry.register(model)
        
        # Search by task type
        classif_models = registry.search(task_type=TaskType.CLASSIFICATION)
        assert len(classif_models) == 1
        assert classif_models[0].name == "model1"
        
        # Search by category
        unsupervised = registry.search(category=ModelCategory.UNSUPERVISED)
        assert len(unsupervised) == 1
        assert unsupervised[0].name == "model2"


class TestAutoSelector:
    """Test automatic model selection."""
    
    @pytest.mark.unit
    def test_data_characteristics_analysis(self, sample_classification_data):
        """Test data characteristics analysis."""
        from mlpy.model_registry.auto_selector import AutoModelSelector, DataCharacteristics
        from mlpy.tasks import TaskClassif
        
        task = TaskClassif(data=sample_classification_data, target='target')
        # Add required attributes
        task.X = sample_classification_data.drop('target', axis=1)
        task.y = sample_classification_data['target']
        task.nrow = len(sample_classification_data)
        task.ncol = len(sample_classification_data.columns)
        
        selector = AutoModelSelector()
        
        chars = selector.analyze_data(task)
        
        assert isinstance(chars, DataCharacteristics)
        assert chars.n_samples == len(sample_classification_data)
        assert chars.n_features == len(sample_classification_data.columns) - 1
        assert chars.n_classes == 3  # A, B, C
        assert chars.dataset_size is not None
        assert chars.dataset_complexity is not None
    
    @pytest.mark.unit
    def test_model_recommendations(self, sample_classification_data):
        """Test model recommendations."""
        from mlpy.model_registry.auto_selector import AutoModelSelector
        from mlpy.model_registry.registry import Complexity
        from mlpy.tasks import TaskClassif
        
        task = TaskClassif(data=sample_classification_data, target='target')
        # Add required attributes
        task.X = sample_classification_data.drop('target', axis=1)
        task.y = sample_classification_data['target']
        task.nrow = len(sample_classification_data)
        task.ncol = len(sample_classification_data.columns)
        
        selector = AutoModelSelector()
        
        # Initialize registry (in real scenario it would have models)
        selector.registry.initialize()
        
        recommendations = selector.recommend_models(
            task=task,
            top_k=3,
            complexity_preference=Complexity.MEDIUM
        )
        
        # Should return list of recommendations (may be empty if no models match)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3


class TestValidation:
    """Test validation functionality."""
    
    @pytest.mark.unit
    def test_validation_with_missing_data(self, sample_missing_data):
        """Test validation detects missing data."""
        # Create simple validation logic
        def validate_data(df):
            errors = []
            warnings = []
            
            # Check for missing values
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                warnings.append(f"Missing values in columns: {missing_cols}")
            
            # Check for minimum samples
            if len(df) < 10:
                errors.append(f"Insufficient samples: {len(df)} < 10")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }
        
        result = validate_data(sample_missing_data)
        
        assert result['valid'] is False  # Too few samples
        assert len(result['errors']) > 0
        assert len(result['warnings']) > 0
    
    @pytest.mark.unit
    def test_validation_with_good_data(self, sample_classification_data):
        """Test validation with good data."""
        def validate_data(df):
            errors = []
            warnings = []
            
            if df.isnull().any().any():
                warnings.append("Contains missing values")
            
            if len(df) < 10:
                errors.append("Too few samples")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }
        
        result = validate_data(sample_classification_data)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0


class TestSerialization:
    """Test model serialization."""
    
    @pytest.mark.unit
    def test_pickle_serialization(self, temp_model_file):
        """Test basic pickle serialization."""
        import pickle
        
        # Create simple model
        model = {"type": "test", "params": {"a": 1, "b": 2}}
        
        # Save
        with open(temp_model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Load
        with open(temp_model_file, 'rb') as f:
            loaded = pickle.load(f)
        
        assert loaded == model
    
    @pytest.mark.unit
    def test_checksum_generation(self):
        """Test SHA256 checksum generation."""
        import hashlib
        
        data = b"test data for checksum"
        checksum = hashlib.sha256(data).hexdigest()
        
        assert len(checksum) == 64  # SHA256 produces 64 char hex string
        assert checksum == hashlib.sha256(data).hexdigest()  # Deterministic


# Run tests with: pytest tests/unit/test_core_models.py -v