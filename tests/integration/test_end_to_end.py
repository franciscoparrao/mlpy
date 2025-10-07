"""
Integration tests for end-to-end workflows in MLPY.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestEndToEndWorkflows:
    """Test complete ML workflows."""
    
    @pytest.mark.integration
    @pytest.mark.requires_sklearn
    def test_classification_workflow(self, sample_classification_data):
        """Test complete classification workflow."""
        from mlpy.tasks import TaskClassif
        from mlpy.resamplings import ResamplingHoldout
        from mlpy.measures import MeasureClassifAccuracy
        
        try:
            from mlpy.learners.sklearn import LearnerRandomForestClassifier
        except ImportError:
            pytest.skip("scikit-learn required")
        
        # 1. Create task
        task = TaskClassif(data=sample_classification_data, target='target')
        assert task.nrow == len(sample_classification_data)
        
        # 2. Create resampling strategy
        resampling = ResamplingHoldout(ratio=0.3, stratify=True)
        resampling_instance = resampling.instantiate(task)
        
        train_idx = resampling_instance.train_set(0)
        test_idx = resampling_instance.test_set(0)
        
        assert len(train_idx) + len(test_idx) == task.nrow
        
        # 3. Create train/test tasks
        train_task = task.filter(train_idx)
        test_task = task.filter(test_idx)
        
        # 4. Train model
        learner = LearnerRandomForestClassifier(n_estimators=10, random_state=42)
        learner.train(train_task)
        
        # 5. Make predictions
        predictions = learner.predict(test_task)
        
        # 6. Evaluate
        measure = MeasureClassifAccuracy()
        score = measure.score(predictions.truth, predictions.response)
        
        assert 0 <= score <= 1
        assert score > 0.3  # Should be better than random
    
    @pytest.mark.integration
    @pytest.mark.requires_sklearn
    def test_regression_workflow(self, sample_regression_data):
        """Test complete regression workflow."""
        from mlpy.tasks import TaskRegr
        from mlpy.resamplings import ResamplingCV
        from mlpy.measures import MeasureRegrMSE
        
        try:
            from mlpy.learners.sklearn import LearnerLinearRegression
        except ImportError:
            pytest.skip("scikit-learn required")
        
        # 1. Create task
        task = TaskRegr(data=sample_regression_data, target='target')
        
        # 2. Cross-validation
        cv = ResamplingCV(folds=3)
        cv_instance = cv.instantiate(task)
        
        scores = []
        learner = LearnerLinearRegression()
        measure = MeasureRegrMSE()
        
        for fold in range(3):
            train_idx = cv_instance.train_set(fold)
            test_idx = cv_instance.test_set(fold)
            
            train_task = task.filter(train_idx)
            test_task = task.filter(test_idx)
            
            # Train
            fold_learner = LearnerLinearRegression()
            fold_learner.train(train_task)
            
            # Predict
            predictions = fold_learner.predict(test_task)
            
            # Score
            score = measure.score(predictions.truth, predictions.response)
            scores.append(score)
        
        # Average score should be reasonable
        avg_score = np.mean(scores)
        assert avg_score < 10  # MSE should be reasonable
    
    @pytest.mark.integration
    def test_ensemble_workflow(self, sample_classification_data):
        """Test ensemble learning workflow."""
        from mlpy.tasks import TaskClassif
        from mlpy.learners.ensemble import LearnerVoting
        from mlpy.learners.baseline import LearnerClassifFeatureless
        
        # Create multiple base learners
        base_learners = [
            LearnerClassifFeatureless(),
            LearnerClassifFeatureless(),
            LearnerClassifFeatureless()
        ]
        
        # Create ensemble
        ensemble = LearnerVoting(
            base_learners=base_learners,
            voting='hard'
        )
        
        # Create task and train
        task = TaskClassif(data=sample_classification_data, target='target')
        ensemble.train(task)
        
        # Predict
        predictions = ensemble.predict(task)
        
        assert len(predictions.response) == len(sample_classification_data)
        assert ensemble.is_trained
    
    @pytest.mark.integration
    @pytest.mark.requires_sklearn
    def test_clustering_workflow(self, sample_clustering_data):
        """Test clustering workflow."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            pytest.skip("scikit-learn required")
        
        X, y_true = sample_clustering_data
        
        # Cluster
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Evaluate
        score = silhouette_score(X, labels)
        
        assert -1 <= score <= 1
        assert score > 0  # Should have decent clustering
    
    @pytest.mark.integration
    def test_model_persistence_workflow(self, temp_dir):
        """Test saving and loading models."""
        import pickle
        from mlpy.learners.baseline import LearnerClassifFeatureless
        from mlpy.tasks import TaskClassif
        
        # Train model
        df = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5],
            'f2': [2, 3, 4, 5, 6],
            'target': ['A', 'B', 'A', 'B', 'A']
        })
        
        task = TaskClassif(data=df, target='target')
        learner = LearnerClassifFeatureless()
        learner.train(task)
        
        # Save
        model_path = temp_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(learner, f)
        
        # Load
        with open(model_path, 'rb') as f:
            loaded_learner = pickle.load(f)
        
        # Verify
        assert loaded_learner.is_trained
        predictions = loaded_learner.predict(task)
        assert len(predictions.response) == len(df)


class TestModelRegistry:
    """Integration tests for model registry."""
    
    @pytest.mark.integration
    def test_registry_full_workflow(self):
        """Test complete registry workflow."""
        from mlpy.model_registry.registry import (
            ModelRegistry, ModelMetadata, ModelCategory, 
            TaskType, Complexity
        )
        from mlpy.model_registry.auto_selector import AutoModelSelector
        from mlpy.tasks import TaskClassif
        
        # 1. Create and populate registry
        registry = ModelRegistry()
        
        models = [
            ModelMetadata(
                name="rf_classifier",
                display_name="Random Forest",
                description="Random Forest Classifier",
                category=ModelCategory.TRADITIONAL_ML,
                class_path="mlpy.learners.sklearn.LearnerRandomForestClassifier",
                task_types=[TaskType.CLASSIFICATION],
                complexity=Complexity.MEDIUM,
                min_samples=50,
                supports_probabilities=True
            ),
            ModelMetadata(
                name="simple_classifier",
                display_name="Simple Classifier",
                description="Simple baseline classifier",
                category=ModelCategory.TRADITIONAL_ML,
                class_path="mlpy.learners.baseline.LearnerClassifFeatureless",
                task_types=[TaskType.CLASSIFICATION],
                complexity=Complexity.LOW,
                min_samples=10
            )
        ]
        
        for model in models:
            registry.register(model)
        
        # 2. Search models
        classif_models = registry.search(task_type=TaskType.CLASSIFICATION)
        assert len(classif_models) >= 2
        
        # 3. Auto-select best model
        selector = AutoModelSelector(registry)
        
        # Create sample task
        df = pd.DataFrame({
            'f1': np.random.randn(100),
            'f2': np.random.randn(100),
            'target': np.random.choice(['A', 'B'], 100)
        })
        task = TaskClassif(data=df, target='target')
        
        # Get recommendations
        recommendations = selector.recommend_models(
            task=task,
            top_k=2,
            complexity_preference=Complexity.LOW
        )
        
        assert len(recommendations) <= 2
        if recommendations:
            assert recommendations[0].confidence_score >= 0
    
    @pytest.mark.integration
    def test_registry_with_auto_initialization(self):
        """Test registry with automatic initialization."""
        from mlpy.model_registry import list_models, search_models
        from mlpy.model_registry.registry import TaskType
        
        # List all models (triggers initialization)
        all_models = list_models()
        assert isinstance(all_models, list)
        
        # Search specific models
        classif_models = search_models(task_type=TaskType.CLASSIFICATION)
        assert isinstance(classif_models, list)


class TestPipelines:
    """Integration tests for ML pipelines."""
    
    @pytest.mark.integration
    def test_basic_pipeline(self, sample_classification_data):
        """Test basic ML pipeline."""
        from mlpy.tasks import TaskClassif
        from mlpy.learners.baseline import LearnerClassifFeatureless
        
        # Complete pipeline
        def ml_pipeline(data, target_col):
            # 1. Create task
            task = TaskClassif(data=data, target=target_col)
            
            # 2. Train model
            learner = LearnerClassifFeatureless()
            learner.train(task)
            
            # 3. Predict
            predictions = learner.predict(task)
            
            # 4. Return results
            return {
                'task': task,
                'learner': learner,
                'predictions': predictions
            }
        
        results = ml_pipeline(sample_classification_data, 'target')
        
        assert results['learner'].is_trained
        assert len(results['predictions'].response) == len(sample_classification_data)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complex_pipeline_with_validation(self):
        """Test complex pipeline with validation and optimization."""
        from mlpy.tasks import TaskClassif
        from mlpy.learners.ensemble import LearnerVoting
        from mlpy.learners.baseline import LearnerClassifFeatureless
        from mlpy.resamplings import ResamplingCV
        from mlpy.measures import MeasureClassifAccuracy
        
        # Generate larger dataset
        np.random.seed(42)
        df = pd.DataFrame({
            'f1': np.random.randn(200),
            'f2': np.random.randn(200),
            'f3': np.random.randn(200),
            'target': np.random.choice(['A', 'B', 'C'], 200)
        })
        
        # 1. Validation step
        def validate_data(df):
            if df.isnull().any().any():
                raise ValueError("Data contains missing values")
            if len(df) < 50:
                raise ValueError("Insufficient data")
            return True
        
        assert validate_data(df)
        
        # 2. Create task
        task = TaskClassif(data=df, target='target')
        
        # 3. Create ensemble
        base_learners = [
            LearnerClassifFeatureless(),
            LearnerClassifFeatureless()
        ]
        ensemble = LearnerVoting(base_learners=base_learners)
        
        # 4. Cross-validation
        cv = ResamplingCV(folds=3)
        cv_instance = cv.instantiate(task)
        measure = MeasureClassifAccuracy()
        
        scores = []
        for fold in range(3):
            train_idx = cv_instance.train_set(fold)
            test_idx = cv_instance.test_set(fold)
            
            train_task = task.filter(train_idx)
            test_task = task.filter(test_idx)
            
            # Train new ensemble for each fold
            fold_ensemble = LearnerVoting(base_learners=[
                LearnerClassifFeatureless(),
                LearnerClassifFeatureless()
            ])
            fold_ensemble.train(train_task)
            
            # Predict and score
            predictions = fold_ensemble.predict(test_task)
            score = measure.score(predictions.truth, predictions.response)
            scores.append(score)
        
        avg_cv_score = np.mean(scores)
        assert 0 <= avg_cv_score <= 1
        assert len(scores) == 3


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.integration
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        from mlpy.tasks import TaskClassif
        
        empty_df = pd.DataFrame()
        
        with pytest.raises(Exception):
            task = TaskClassif(data=empty_df, target='target')
    
    @pytest.mark.integration
    def test_missing_target_column(self):
        """Test handling of missing target column."""
        from mlpy.tasks import TaskClassif
        
        df = pd.DataFrame({
            'f1': [1, 2, 3],
            'f2': [4, 5, 6]
        })
        
        with pytest.raises(Exception):
            task = TaskClassif(data=df, target='nonexistent')
    
    @pytest.mark.integration
    def test_untrained_model_prediction(self):
        """Test prediction with untrained model."""
        from mlpy.learners.baseline import LearnerClassifFeatureless
        from mlpy.tasks import TaskClassif
        
        df = pd.DataFrame({
            'f1': [1, 2, 3],
            'target': ['A', 'B', 'A']
        })
        
        task = TaskClassif(data=df, target='target')
        learner = LearnerClassifFeatureless()
        
        with pytest.raises(RuntimeError, match="Model must be trained"):
            learner.predict(task)


# Run with: pytest tests/integration/test_end_to_end.py -v