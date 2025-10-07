"""
Tests for ensemble learners.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier

from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners import (
    LearnerVoting,
    LearnerStacking,
    LearnerBlending,
    create_ensemble,
    LearnerClassifSklearn,
    LearnerRegrSklearn
)
from mlpy.predictions import PredictionClassif, PredictionRegr


class TestLearnerVoting:
    """Test voting ensemble learner."""
    
    @pytest.fixture
    def classif_task(self):
        """Create classification task."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=3,
            n_informative=5,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df['target'] = y
        return TaskClassif(df, target='target')
    
    @pytest.fixture
    def regr_task(self):
        """Create regression task."""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=5,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df['target'] = y
        return TaskRegr(df, target='target')
    
    @pytest.fixture
    def base_classifiers(self):
        """Create base classifiers."""
        return [
            LearnerClassifSklearn(estimator=LogisticRegression(random_state=42, max_iter=200)),
            LearnerClassifSklearn(estimator=DecisionTreeClassifier(random_state=42)),
            LearnerClassifSklearn(estimator=KNeighborsClassifier(n_neighbors=5))
        ]
    
    @pytest.fixture
    def base_regressors(self):
        """Create base regressors."""
        return [
            LearnerRegrSklearn(estimator=LinearRegression()),
            LearnerRegrSklearn(estimator=DecisionTreeRegressor(random_state=42)),
            LearnerRegrSklearn(estimator=KNeighborsRegressor(n_neighbors=5))
        ]
    
    def test_voting_hard_classification(self, classif_task, base_classifiers):
        """Test hard voting for classification."""
        voting = LearnerVoting(
            base_learners=base_classifiers,
            voting='hard'
        )
        
        # Train
        voting.train(classif_task)
        assert voting._trained_learners is not None
        assert len(voting._trained_learners) == 3
        
        # Predict
        predictions = voting.predict(classif_task)
        assert isinstance(predictions, PredictionClassif)
        assert len(predictions.response) == classif_task.nrow
        
        # Check predictions are valid classes
        valid_classes = classif_task.class_names
        for pred in predictions.response:
            assert str(pred) in valid_classes
    
    def test_voting_soft_classification(self, classif_task, base_classifiers):
        """Test soft voting for classification."""
        voting = LearnerVoting(
            base_learners=base_classifiers,
            voting='soft'
        )
        
        voting.train(classif_task)
        predictions = voting.predict(classif_task)
        
        assert isinstance(predictions, PredictionClassif)
        assert hasattr(predictions, 'prob')
        assert predictions.prob.shape == (classif_task.nrow, len(classif_task.class_names))
        
        # Probabilities should sum to 1
        prob_sums = predictions.prob.sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sums, np.ones(len(prob_sums)))
    
    def test_voting_weighted(self, classif_task, base_classifiers):
        """Test weighted voting."""
        weights = [0.5, 0.3, 0.2]
        voting = LearnerVoting(
            base_learners=base_classifiers,
            voting='hard',
            weights=weights
        )
        
        voting.train(classif_task)
        predictions = voting.predict(classif_task)
        
        assert isinstance(predictions, PredictionClassif)
        # Weights should be normalized
        assert np.isclose(voting.weights.sum(), 1.0)
    
    def test_voting_regression(self, regr_task, base_regressors):
        """Test voting for regression (averaging)."""
        voting = LearnerVoting(
            base_learners=base_regressors,
            voting='hard'  # Ignored for regression
        )
        
        voting.train(regr_task)
        predictions = voting.predict(regr_task)
        
        assert isinstance(predictions, PredictionRegr)
        assert len(predictions.response) == regr_task.nrow
        assert all(isinstance(p, (int, float)) for p in predictions.response)
    
    def test_voting_invalid_params(self, base_classifiers):
        """Test invalid parameters."""
        # Invalid voting type
        with pytest.raises(ValueError, match="voting must be"):
            LearnerVoting(base_classifiers, voting='invalid')
        
        # Wrong number of weights
        with pytest.raises(ValueError, match="Number of weights"):
            LearnerVoting(base_classifiers, weights=[0.5, 0.5])
        
        # No base learners
        with pytest.raises(ValueError, match="At least one base learner"):
            LearnerVoting([])


class TestLearnerStacking:
    """Test stacking ensemble learner."""
    
    @pytest.fixture
    def classif_task(self):
        """Create classification task."""
        X, y = make_classification(
            n_samples=150,
            n_features=10,
            n_classes=2,
            n_informative=5,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df['target'] = y
        return TaskClassif(df, target='target')
    
    @pytest.fixture
    def base_classifiers(self):
        """Create base classifiers."""
        return [
            LearnerClassifSklearn(estimator=LogisticRegression(random_state=42, max_iter=200)),
            LearnerClassifSklearn(estimator=DecisionTreeClassifier(random_state=42))
        ]
    
    @pytest.fixture
    def meta_classifier(self):
        """Create meta-classifier."""
        return LearnerClassifSklearn(estimator=LogisticRegression(random_state=42, max_iter=200))
    
    def test_stacking_basic(self, classif_task, base_classifiers, meta_classifier):
        """Test basic stacking."""
        stacking = LearnerStacking(
            base_learners=base_classifiers,
            meta_learner=meta_classifier,
            cv_folds=3
        )
        
        # Train
        stacking.train(classif_task)
        assert stacking._trained_learners is not None
        assert stacking._trained_meta is not None
        assert stacking._meta_task is not None
        
        # Predict
        predictions = stacking.predict(classif_task)
        assert isinstance(predictions, PredictionClassif)
        assert len(predictions.response) == classif_task.nrow
    
    def test_stacking_with_proba(self, classif_task, base_classifiers, meta_classifier):
        """Test stacking with probabilities as meta-features."""
        stacking = LearnerStacking(
            base_learners=base_classifiers,
            meta_learner=meta_classifier,
            use_proba=True,
            cv_folds=3
        )
        
        stacking.train(classif_task)
        
        # Meta-task should have probability features
        n_classes = len(classif_task.class_names)
        expected_features = len(base_classifiers) * n_classes
        assert len(stacking._meta_task.feature_names) == expected_features
        
        predictions = stacking.predict(classif_task)
        assert isinstance(predictions, PredictionClassif)
    
    def test_stacking_regression(self):
        """Test stacking for regression."""
        # Create regression task
        X, y = make_regression(n_samples=150, n_features=10, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df['target'] = y
        task = TaskRegr(df, target='target')
        
        # Base learners
        base_learners = [
            LearnerRegrSklearn(estimator=LinearRegression()),
            LearnerRegrSklearn(estimator=DecisionTreeRegressor(random_state=42))
        ]
        
        # Meta learner
        meta_learner = LearnerRegrSklearn(estimator=LinearRegression())
        
        stacking = LearnerStacking(
            base_learners=base_learners,
            meta_learner=meta_learner,
            cv_folds=3
        )
        
        stacking.train(task)
        predictions = stacking.predict(task)
        
        assert isinstance(predictions, PredictionRegr)
        assert len(predictions.response) == task.nrow
    
    def test_stacking_meta_features_generation(self, classif_task, base_classifiers, meta_classifier):
        """Test meta-features generation process."""
        stacking = LearnerStacking(
            base_learners=base_classifiers,
            meta_learner=meta_classifier,
            cv_folds=5
        )
        
        # Generate meta-features
        meta_features = stacking._generate_meta_features(classif_task)
        
        assert meta_features.shape[0] == classif_task.nrow
        assert meta_features.shape[1] == len(base_classifiers)
        
        # Should have no NaN values (all samples covered by CV)
        assert not np.isnan(meta_features).any()


class TestLearnerBlending:
    """Test blending ensemble learner."""
    
    @pytest.fixture
    def classif_task(self):
        """Create classification task."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            n_informative=5,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df['target'] = y
        return TaskClassif(df, target='target')
    
    def test_blending_basic(self, classif_task):
        """Test basic blending."""
        base_learners = [
            LearnerClassifSklearn(estimator=LogisticRegression(random_state=42, max_iter=200)),
            LearnerClassifSklearn(estimator=DecisionTreeClassifier(random_state=42))
        ]
        meta_learner = LearnerClassifSklearn(estimator=LogisticRegression(random_state=42, max_iter=200))
        
        blending = LearnerBlending(
            base_learners=base_learners,
            meta_learner=meta_learner,
            blend_ratio=0.3
        )
        
        # Train
        blending.train(classif_task)
        assert blending._trained_learners is not None
        assert blending._trained_meta is not None
        
        # Predict
        predictions = blending.predict(classif_task)
        assert isinstance(predictions, PredictionClassif)
        assert len(predictions.response) == classif_task.nrow
    
    def test_blending_with_proba(self, classif_task):
        """Test blending with probabilities."""
        base_learners = [
            LearnerClassifSklearn(estimator=LogisticRegression(random_state=42, max_iter=200)),
            LearnerClassifSklearn(estimator=RandomForestClassifier(random_state=42, n_estimators=10))
        ]
        meta_learner = LearnerClassifSklearn(estimator=LogisticRegression(random_state=42, max_iter=200))
        
        blending = LearnerBlending(
            base_learners=base_learners,
            meta_learner=meta_learner,
            blend_ratio=0.25,
            use_proba=True
        )
        
        blending.train(classif_task)
        predictions = blending.predict(classif_task)
        
        assert isinstance(predictions, PredictionClassif)
        # Meta features should include probabilities
        n_classes = len(classif_task.class_names)
        expected_features = len(base_learners) * n_classes
        assert len(blending._meta_feature_names) == expected_features
    
    def test_blending_invalid_ratio(self):
        """Test invalid blend ratio."""
        base_learners = [LearnerClassifSklearn(estimator=LogisticRegression(random_state=42, max_iter=200))]
        meta_learner = LearnerClassifSklearn(estimator=LogisticRegression(random_state=42, max_iter=200))
        
        with pytest.raises(ValueError, match="blend_ratio must be between"):
            LearnerBlending(base_learners, meta_learner, blend_ratio=0)
        
        with pytest.raises(ValueError, match="blend_ratio must be between"):
            LearnerBlending(base_learners, meta_learner, blend_ratio=1.5)


class TestCreateEnsemble:
    """Test ensemble creation helper."""
    
    def test_create_voting(self):
        """Test creating voting ensemble."""
        base_learners = [
            LearnerClassifSklearn(estimator=LogisticRegression(random_state=42, max_iter=200)),
            LearnerClassifSklearn(estimator=DecisionTreeClassifier(random_state=42))
        ]
        
        ensemble = create_ensemble('voting', base_learners, voting='hard')
        assert isinstance(ensemble, LearnerVoting)
        assert ensemble.voting == 'hard'
    
    def test_create_stacking(self):
        """Test creating stacking ensemble."""
        base_learners = [
            LearnerClassifSklearn(estimator=LogisticRegression(random_state=42, max_iter=200)),
            LearnerClassifSklearn(estimator=DecisionTreeClassifier(random_state=42))
        ]
        meta_learner = LearnerClassifSklearn(estimator=LogisticRegression(random_state=42, max_iter=200))
        
        ensemble = create_ensemble('stacking', base_learners, meta_learner=meta_learner)
        assert isinstance(ensemble, LearnerStacking)
    
    def test_create_blending(self):
        """Test creating blending ensemble."""
        base_learners = [
            LearnerClassifSklearn(estimator=LogisticRegression(random_state=42, max_iter=200)),
            LearnerClassifSklearn(estimator=DecisionTreeClassifier(random_state=42))
        ]
        meta_learner = LearnerClassifSklearn(estimator=LogisticRegression(random_state=42, max_iter=200))
        
        ensemble = create_ensemble('blending', base_learners, 
                                 meta_learner=meta_learner, blend_ratio=0.3)
        assert isinstance(ensemble, LearnerBlending)
        assert ensemble.blend_ratio == 0.3
    
    def test_create_invalid_method(self):
        """Test invalid ensemble method."""
        base_learners = [LearnerClassifSklearn(estimator=LogisticRegression(random_state=42, max_iter=200))]
        
        with pytest.raises(ValueError, match="Unknown ensemble method"):
            create_ensemble('invalid', base_learners)