"""Unit tests for scikit-learn integration."""

import pytest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.base import BaseEstimator

from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners import (
    LearnerSklearn, LearnerClassifSklearn, LearnerRegrSklearn,
    learner_sklearn
)
from mlpy.predictions import PredictionClassif, PredictionRegr
from mlpy.measures import MeasureClassifAccuracy, MeasureRegrMSE
from mlpy.resamplings import ResamplingCV
from mlpy import resample


class TestLearnerSklearn:
    """Test base sklearn learner functionality."""
    
    @pytest.fixture
    def iris_task(self):
        """Create Iris classification task."""
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target_names[iris.target]
        return TaskClassif(data=df, target='species', id='iris')
        
    @pytest.fixture
    def boston_task(self):
        """Create Boston regression task."""
        # Create synthetic regression data
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        y = X[:, 0] * 2 + X[:, 1] * -1.5 + np.random.randn(n) * 0.5
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        return TaskRegr(data=df, target='target', id='regression')
        
    def test_property_detection(self):
        """Test automatic property detection."""
        # Decision tree - should have feature_importances
        dt = DecisionTreeClassifier()
        learner = LearnerClassifSklearn(dt)
        assert 'tree_based' in learner.properties
        
        # Logistic regression - should have linear property
        lr = LogisticRegression()
        learner = LearnerClassifSklearn(lr)
        assert 'linear' in learner.properties
        assert 'prob' in learner.properties  # Has predict_proba
        
        # SVM - should have kernel property
        svm = SVC()
        learner = LearnerClassifSklearn(svm)
        assert 'kernel' in learner.properties
        assert 'decision' in learner.properties  # Has decision_function
        
        # Random Forest - should have ensemble property
        rf = RandomForestClassifier()
        learner = LearnerClassifSklearn(rf)
        assert 'ensemble' in learner.properties
        assert 'tree_based' in learner.properties
        
    def test_auto_id_generation(self):
        """Test automatic ID generation from estimator class name."""
        dt = DecisionTreeClassifier()
        learner = LearnerClassifSklearn(dt)
        assert learner.id == 'sklearn.decisiontreeclassifier'
        
        rf = RandomForestRegressor()
        learner = LearnerRegrSklearn(rf)
        assert learner.id == 'sklearn.randomforestregressor'
        
    def test_task_type_detection(self):
        """Test correct task type detection."""
        # Classifiers
        clf = DecisionTreeClassifier()
        learner = LearnerSklearn(clf)
        assert learner.task_type == 'classif'
        
        # Regressors
        reg = DecisionTreeRegressor()
        learner = LearnerSklearn(reg)
        assert learner.task_type == 'regr'
        
    def test_train_predict_classifier(self, iris_task):
        """Test training and prediction with classifier."""
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        learner = LearnerClassifSklearn(clf)
        
        # Train
        learner.train(iris_task)
        assert learner.is_trained
        assert learner.model is not None
        
        # Predict
        pred = learner.predict(iris_task)
        assert isinstance(pred, PredictionClassif)
        assert len(pred.response) == len(iris_task.row_roles['use'])
        
        # Check accuracy is reasonable
        accuracy = MeasureClassifAccuracy()
        score = accuracy.score(pred, iris_task)
        assert score > 0.8  # Should get decent accuracy on iris
        
    def test_train_predict_regressor(self, boston_task):
        """Test training and prediction with regressor."""
        reg = LinearRegression()
        learner = LearnerRegrSklearn(reg)
        
        # Train
        learner.train(boston_task)
        assert learner.is_trained
        
        # Predict
        pred = learner.predict(boston_task)
        assert isinstance(pred, PredictionRegr)
        assert len(pred.response) == len(boston_task.row_roles['use'])
        
        # Check MSE is reasonable
        mse = MeasureRegrMSE()
        score = mse.score(pred, boston_task)
        assert score < 5.0  # Should get reasonable MSE
        
    def test_probability_predictions(self, iris_task):
        """Test probability predictions for classifiers."""
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        learner = LearnerClassifSklearn(clf, predict_type='prob')
        
        learner.train(iris_task)
        pred = learner.predict(iris_task)
        
        # Should have probabilities
        assert pred.prob is not None
        assert isinstance(pred.prob, pd.DataFrame)
        assert list(pred.prob.columns) == sorted(iris_task.class_names)
        
        # Probabilities should sum to 1
        prob_sums = pred.prob.sum(axis=1)
        assert np.allclose(prob_sums, 1.0)
        
    def test_feature_importances(self, iris_task):
        """Test feature importance extraction."""
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        learner = LearnerClassifSklearn(rf)
        
        # No importances before training
        assert learner.feature_importances is None
        
        # Train
        learner.train(iris_task)
        
        # Should have importances after training
        importances = learner.feature_importances
        assert importances is not None
        assert len(importances) == len(iris_task.feature_names)
        assert np.sum(importances) > 0
        
    def test_clone(self, iris_task):
        """Test learner cloning."""
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        learner = LearnerClassifSklearn(clf)
        
        # Train original
        learner.train(iris_task)
        
        # Clone
        cloned = learner.clone()
        
        # Clone should not be trained
        assert not cloned.is_trained
        assert cloned.id == learner.id
        assert cloned.predict_type == learner.predict_type
        
        # Original should still be trained
        assert learner.is_trained
        
    def test_subset_training(self, iris_task):
        """Test training on subset of data."""
        clf = DecisionTreeClassifier(random_state=42)
        learner = LearnerClassifSklearn(clf)
        
        # Train on first 100 samples
        row_ids = list(range(100))
        learner.train(iris_task, row_ids=row_ids)
        
        # Predict on different subset
        test_ids = list(range(100, 150))
        pred = learner.predict(iris_task, row_ids=test_ids)
        
        assert len(pred.response) == 50
        
    def test_error_handling(self, iris_task, boston_task):
        """Test error handling for wrong task types."""
        # Classifier on regression task
        clf = DecisionTreeClassifier()
        learner = LearnerClassifSklearn(clf)
        
        with pytest.raises(TypeError, match="Classifier requires TaskClassif"):
            learner.train(boston_task)
            
        # Regressor on classification task
        reg = DecisionTreeRegressor()
        learner = LearnerRegrSklearn(reg)
        
        with pytest.raises(TypeError, match="Regressor requires TaskRegr"):
            learner.train(iris_task)
            
    def test_not_trained_error(self, iris_task):
        """Test error when predicting without training."""
        clf = DecisionTreeClassifier()
        learner = LearnerClassifSklearn(clf)
        
        with pytest.raises(RuntimeError, match="Model must be trained"):
            learner.predict(iris_task)


class TestLearnerClassifSklearn:
    """Test sklearn classifier wrapper."""
    
    def test_invalid_estimator(self):
        """Test error for non-classifier estimator."""
        reg = LinearRegression()
        
        with pytest.raises(TypeError, match="must be a scikit-learn classifier"):
            LearnerClassifSklearn(reg)
            
    def test_task_type(self):
        """Test task type is always classif."""
        clf = LogisticRegression()
        learner = LearnerClassifSklearn(clf)
        assert learner.task_type == 'classif'


class TestLearnerRegrSklearn:
    """Test sklearn regressor wrapper."""
    
    def test_invalid_estimator(self):
        """Test error for non-regressor estimator."""
        clf = LogisticRegression()
        
        with pytest.raises(TypeError, match="appears to be a classifier"):
            LearnerRegrSklearn(clf)
            
    def test_task_type(self):
        """Test task type is always regr."""
        reg = LinearRegression()
        learner = LearnerRegrSklearn(reg)
        assert learner.task_type == 'regr'
        
    def test_predict_type_forced(self):
        """Test predict_type is always response for regressors."""
        reg = LinearRegression()
        learner = LearnerRegrSklearn(reg, predict_type='prob')
        assert learner.predict_type == 'response'  # Should be forced to response


class TestLearnerSklearnConvenience:
    """Test convenience function learner_sklearn."""
    
    def test_auto_classifier_detection(self):
        """Test automatic detection of classifiers."""
        clf = RandomForestClassifier()
        learner = learner_sklearn(clf)
        assert isinstance(learner, LearnerClassifSklearn)
        assert learner.task_type == 'classif'
        
    def test_auto_regressor_detection(self):
        """Test automatic detection of regressors."""
        reg = RandomForestRegressor()
        learner = learner_sklearn(reg)
        assert isinstance(learner, LearnerRegrSklearn)
        assert learner.task_type == 'regr'
        
    def test_with_custom_params(self):
        """Test passing custom parameters."""
        clf = DecisionTreeClassifier()
        learner = learner_sklearn(clf, id='my_tree', predict_type='prob')
        assert learner.id == 'my_tree'
        assert learner.predict_type == 'prob'


class TestSklearnIntegration:
    """Integration tests with full MLPY workflow."""
    
    @pytest.fixture
    def make_classification_task(self):
        """Create synthetic classification task."""
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5,
            n_classes=3, random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
        df['target'] = y
        return TaskClassif(data=df, target='target')
        
    @pytest.fixture
    def make_regression_task(self):
        """Create synthetic regression task."""
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=200, n_features=10, n_informative=5,
            noise=0.1, random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
        df['target'] = y
        return TaskRegr(data=df, target='target')
        
    def test_resample_classification(self, make_classification_task):
        """Test resampling with sklearn classifier."""
        # Multiple sklearn algorithms
        learners = [
            learner_sklearn(DecisionTreeClassifier(max_depth=5, random_state=42)),
            learner_sklearn(RandomForestClassifier(n_estimators=10, random_state=42)),
            learner_sklearn(LogisticRegression(max_iter=200, random_state=42))
        ]
        
        for learner in learners:
            result = resample(
                task=make_classification_task,
                learner=learner,
                resampling=ResamplingCV(folds=3),
                measures=MeasureClassifAccuracy()
            )
            
            # Should complete without errors
            assert result.n_iters == 3
            assert result.n_errors == 0
            
            # Should get reasonable accuracy
            mean_acc = result.score()
            assert 0.5 < mean_acc < 1.0
            
    def test_resample_regression(self, make_regression_task):
        """Test resampling with sklearn regressor."""
        # Multiple sklearn algorithms
        learners = [
            learner_sklearn(DecisionTreeRegressor(max_depth=5, random_state=42)),
            learner_sklearn(RandomForestRegressor(n_estimators=10, random_state=42)),
            learner_sklearn(LinearRegression())
        ]
        
        for learner in learners:
            result = resample(
                task=make_regression_task,
                learner=learner,
                resampling=ResamplingCV(folds=3),
                measures=MeasureRegrMSE()
            )
            
            # Should complete without errors
            assert result.n_iters == 3
            assert result.n_errors == 0
            
            # Should get reasonable MSE
            mean_mse = result.score()
            assert mean_mse > 0
            
    def test_with_preprocessing(self, make_classification_task):
        """Test sklearn pipeline integration."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        # Create pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42))
        ])
        
        learner = learner_sklearn(pipe)
        
        # Should work with pipeline
        result = resample(
            task=make_classification_task,
            learner=learner,
            resampling=ResamplingCV(folds=3),
            measures=MeasureClassifAccuracy()
        )
        
        assert result.n_errors == 0
        assert result.score() > 0.5