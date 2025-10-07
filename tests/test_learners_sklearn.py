"""Test scikit-learn learner wrappers."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import warnings

from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.predictions import PredictionClassif, PredictionRegr


# Test if sklearn is available
try:
    import sklearn
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    sklearn = None


# Import MLPY sklearn wrappers
try:
    from mlpy.learners.sklearn import (
        LearnerSKLearn,
        LearnerLogisticRegression,
        LearnerDecisionTree,
        LearnerRandomForest,
        LearnerLinearRegression,
        LearnerRidge,
        LearnerLasso,
        auto_sklearn
    )
    from mlpy.learners.sklearn.base import LearnerClassifSKLearn, LearnerRegrSKLearn
    _HAS_MLPY_SKLEARN = True
except ImportError:
    _HAS_MLPY_SKLEARN = False


@pytest.fixture
def sample_task_classif():
    """Create a sample classification task."""
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n),
        'y': np.random.choice(['A', 'B', 'C'], n)
    })
    return TaskClassif(data=data, target='y', id='test_task')


@pytest.fixture
def sample_task_regr():
    """Create a sample regression task."""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 3)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1
    data = pd.DataFrame({
        'x1': X[:, 0],
        'x2': X[:, 1],
        'x3': X[:, 2],
        'y': y
    })
    return TaskRegr(data=data, target='y', id='test_task')


@pytest.mark.skipif(not _HAS_MLPY_SKLEARN, reason="MLPY sklearn wrappers not available")
class TestLearnerSKLearnBase:
    """Test base sklearn wrapper functionality."""
    
    def test_task_type_inference(self):
        """Test automatic task type inference."""
        # Mock classifier
        mock_classifier = Mock()
        mock_classifier.__name__ = "TestClassifier"
        mock_classifier._estimator_type = "classifier"
        
        assert LearnerSKLearn._infer_task_type(mock_classifier) == "classif"
        
        # Mock regressor
        mock_regressor = Mock()
        mock_regressor.__name__ = "TestRegressor"
        mock_regressor._estimator_type = "regressor"
        
        assert LearnerSKLearn._infer_task_type(mock_regressor) == "regr"
        
        # By name
        mock_model = Mock()
        mock_model.__name__ = "LogisticRegression"
        
        assert LearnerSKLearn._infer_task_type(mock_model) == "classif"
        
    def test_wrapper_creation(self):
        """Test creating wrapper with mock estimator."""
        mock_estimator = Mock()
        mock_estimator.__name__ = "MockEstimator"
        mock_estimator._estimator_type = "classifier"
        
        learner = LearnerSKLearn(
            estimator_class=mock_estimator,
            id="test_learner",
            param1=10,
            param2="value"
        )
        
        assert learner.id == "test_learner"
        assert learner.estimator_class == mock_estimator
        assert learner.estimator_params == {"param1": 10, "param2": "value"}
        assert learner.task_type == "classif"
        
    @pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
    def test_train_predict_classif(self, sample_task_classif):
        """Test training and prediction for classification."""
        learner = LearnerLogisticRegression(random_state=42)
        
        # Train
        learner.train(sample_task_classif)
        assert learner.is_trained
        assert learner.estimator is not None
        
        # Predict response
        pred = learner.predict(sample_task_classif)
        assert isinstance(pred, PredictionClassif)
        assert len(pred.response) == len(sample_task_classif.y)
        assert pred.prob is None  # Default is response only
        
        # Predict probabilities
        learner_prob = LearnerLogisticRegression(
            predict_type="prob",
            random_state=42
        )
        learner_prob.train(sample_task_classif)
        pred_prob = learner_prob.predict(sample_task_classif)
        
        assert pred_prob.prob is not None
        assert isinstance(pred_prob.prob, pd.DataFrame)
        assert pred_prob.prob.shape[0] == len(sample_task_classif.y)
        
    @pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
    def test_train_predict_regr(self, sample_task_regr):
        """Test training and prediction for regression."""
        learner = LearnerLinearRegression()
        
        # Train
        learner.train(sample_task_regr)
        assert learner.is_trained
        assert learner.estimator is not None
        
        # Predict
        pred = learner.predict(sample_task_regr)
        assert isinstance(pred, PredictionRegr)
        assert len(pred.response) == len(sample_task_regr.y)
        
    def test_clone_reset(self):
        """Test cloning and resetting learners."""
        mock_estimator = Mock()
        mock_estimator.__name__ = "MockEstimator"
        mock_estimator._estimator_type = "classifier"
        
        learner = LearnerSKLearn(
            estimator_class=mock_estimator,
            id="test",
            param1=10
        )
        
        # Clone
        learner2 = learner.clone()
        assert learner2.id == learner.id
        assert learner2.estimator_params == learner.estimator_params
        assert learner2 is not learner
        
        # Reset
        learner.is_trained = True
        learner.estimator = "dummy"
        learner.reset()
        
        assert not learner.is_trained
        assert learner.estimator is None
        
    def test_get_set_params(self):
        """Test parameter getting and setting."""
        mock_estimator = Mock()
        mock_estimator.__name__ = "MockEstimator"
        
        learner = LearnerSKLearn(
            estimator_class=mock_estimator,
            id="test",
            param1=10,
            param2="value"
        )
        
        # Get params
        params = learner.get_params()
        assert params["id"] == "test"
        assert params["param1"] == 10
        assert params["param2"] == "value"
        
        # Set params
        learner.set_params(param1=20, param3="new")
        assert learner.estimator_params["param1"] == 20
        assert learner.estimator_params["param3"] == "new"


@pytest.mark.skipif(not _HAS_MLPY_SKLEARN, reason="MLPY sklearn wrappers not available")
@pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
class TestClassificationWrappers:
    """Test specific classification model wrappers."""
    
    def test_logistic_regression(self, sample_task_classif):
        """Test LogisticRegression wrapper."""
        learner = LearnerLogisticRegression(
            C=0.5,
            penalty='l2',
            random_state=42
        )
        
        assert learner.id == "logreg"
        learner.train(sample_task_classif)
        pred = learner.predict(sample_task_classif)
        
        assert isinstance(pred, PredictionClassif)
        assert hasattr(learner.estimator, 'coef_')
        
    def test_decision_tree(self, sample_task_classif):
        """Test DecisionTree wrapper."""
        learner = LearnerDecisionTree(
            max_depth=3,
            random_state=42
        )
        
        assert learner.id == "decision_tree"
        learner.train(sample_task_classif)
        pred = learner.predict(sample_task_classif)
        
        assert isinstance(pred, PredictionClassif)
        assert hasattr(learner.estimator, 'tree_')
        
    def test_random_forest(self, sample_task_classif):
        """Test RandomForest wrapper."""
        learner = LearnerRandomForest(
            n_estimators=10,
            max_depth=3,
            random_state=42
        )
        
        assert learner.id == "random_forest"
        learner.train(sample_task_classif)
        
        # Test probability prediction
        learner.predict_type = "prob"
        pred = learner.predict(sample_task_classif)
        
        assert isinstance(pred, PredictionClassif)
        assert pred.prob is not None
        assert hasattr(learner.estimator, 'estimators_')


@pytest.mark.skipif(not _HAS_MLPY_SKLEARN, reason="MLPY sklearn wrappers not available")
@pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
class TestRegressionWrappers:
    """Test specific regression model wrappers."""
    
    def test_linear_regression(self, sample_task_regr):
        """Test LinearRegression wrapper."""
        learner = LearnerLinearRegression()
        
        assert learner.id == "linear_regression"
        learner.train(sample_task_regr)
        pred = learner.predict(sample_task_regr)
        
        assert isinstance(pred, PredictionRegr)
        assert hasattr(learner.estimator, 'coef_')
        
    def test_ridge(self, sample_task_regr):
        """Test Ridge wrapper."""
        learner = LearnerRidge(alpha=0.5)
        
        assert learner.id == "ridge"
        learner.train(sample_task_regr)
        pred = learner.predict(sample_task_regr)
        
        assert isinstance(pred, PredictionRegr)
        assert learner.estimator.alpha == 0.5
        
    def test_lasso(self, sample_task_regr):
        """Test Lasso wrapper."""
        learner = LearnerLasso(alpha=0.1, random_state=42)
        
        assert learner.id == "lasso"
        learner.train(sample_task_regr)
        pred = learner.predict(sample_task_regr)
        
        assert isinstance(pred, PredictionRegr)
        # Check that Lasso performs feature selection (some coefs should be 0)
        assert np.any(np.abs(learner.estimator.coef_) < 1e-10)


@pytest.mark.skipif(not _HAS_MLPY_SKLEARN, reason="MLPY sklearn wrappers not available")
class TestAutoWrap:
    """Test auto-wrapper functionality."""
    
    def test_auto_wrap_class(self):
        """Test auto-wrapping a class."""
        # Mock estimator class
        mock_class = Mock()
        mock_class.__name__ = "TestClassifier"
        mock_class._estimator_type = "classifier"
        
        learner = auto_sklearn(mock_class, n_estimators=100)
        
        assert learner.id == "test"
        assert learner.estimator_params["n_estimators"] == 100
        assert isinstance(learner, LearnerClassifSKLearn)
        
    def test_auto_wrap_instance(self):
        """Test auto-wrapping an instance."""
        # Mock fitted instance
        mock_instance = Mock()
        mock_instance.__class__.__name__ = "TestRegressor"
        mock_instance.__class__._estimator_type = "regressor"
        mock_instance.get_params.return_value = {"alpha": 1.0}
        mock_instance.classes_ = None  # Not fitted
        
        learner = auto_sklearn(mock_instance)
        
        assert learner.id == "test"
        assert isinstance(learner, LearnerRegrSKLearn)
        
    def test_auto_wrap_explicit_type(self):
        """Test auto-wrap with explicit task type."""
        # Mock ambiguous estimator
        mock_class = Mock()
        mock_class.__name__ = "AmbiguousModel"
        
        # Should work with explicit type
        learner = auto_sklearn(mock_class, task_type="regr")
        assert isinstance(learner, LearnerRegrSKLearn)
        
    def test_id_creation(self):
        """Test automatic ID creation from class names."""
        from mlpy.learners.sklearn.auto_wrap import _create_id
        
        assert _create_id(type("RandomForestClassifier", (), {})) == "random_forest"
        assert _create_id(type("MLPRegressor", (), {})) == "mlp"
        assert _create_id(type("GradientBoostingClassifier", (), {})) == "gradient_boosting"
        
    @pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
    def test_auto_wrap_real_sklearn(self, sample_task_classif):
        """Test auto-wrapping real sklearn models."""
        # Wrap RandomForestClassifier
        learner = auto_sklearn(
            RandomForestClassifier,
            n_estimators=10,
            random_state=42
        )
        
        learner.train(sample_task_classif)
        pred = learner.predict(sample_task_classif)
        
        assert isinstance(pred, PredictionClassif)
        assert learner.estimator.n_estimators == 10
        
    def test_list_available_models(self):
        """Test listing available sklearn models."""
        from mlpy.learners.sklearn.auto_wrap import list_available_sklearn_models
        
        models = list_available_sklearn_models()
        
        assert isinstance(models, dict)
        assert 'classifiers' in models
        assert 'regressors' in models
        
        if _HAS_SKLEARN:
            # Should have found some models
            assert len(models['classifiers']) > 0
            assert len(models['regressors']) > 0


@pytest.mark.skipif(not _HAS_MLPY_SKLEARN, reason="MLPY sklearn wrappers not available")
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_sklearn_import(self):
        """Test error when sklearn not installed."""
        with patch('mlpy.learners.sklearn.classification._HAS_SKLEARN', False):
            with pytest.raises(ImportError, match="scikit-learn is required"):
                LearnerLogisticRegression()
                
    def test_untrained_prediction(self, sample_task_classif):
        """Test error when predicting without training."""
        mock_estimator = Mock()
        mock_estimator.__name__ = "MockEstimator"
        mock_estimator._estimator_type = "classifier"
        
        learner = LearnerSKLearn(estimator_class=mock_estimator)
        
        with pytest.raises(ValueError, match="must be trained"):
            learner.predict(sample_task_classif)
            
    def test_no_predict_proba_fallback(self, sample_task_classif):
        """Test fallback when model doesn't support predict_proba."""
        # Mock estimator without predict_proba
        mock_estimator = Mock()
        mock_estimator.__name__ = "NoProbClassifier"
        mock_estimator._estimator_type = "classifier"
        
        mock_instance = Mock()
        mock_instance.predict.return_value = np.array(['A', 'B', 'A'])
        mock_instance.classes_ = np.array(['A', 'B'])
        del mock_instance.predict_proba  # Remove predict_proba
        
        mock_estimator.return_value = mock_instance
        
        learner = LearnerSKLearn(
            estimator_class=mock_estimator,
            predict_type="prob"
        )
        learner.is_trained = True
        learner.estimator = mock_instance
        
        with warnings.catch_warnings(record=True) as w:
            pred = learner.predict(sample_task_classif)
            assert len(w) == 1
            assert "does not support probability predictions" in str(w[0].message)
            
        assert pred.prob is None
        assert pred.response is not None


if __name__ == "__main__":
    pytest.main([__file__])