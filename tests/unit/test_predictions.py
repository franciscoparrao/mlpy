"""Tests for Prediction classes."""

import pytest
import numpy as np
import pandas as pd

from mlpy.prediction import PredictionClassif, PredictionRegr
from mlpy.tasks import TaskClassif, TaskRegr


class TestPredictionClassif:
    """Test classification predictions."""
    
    @pytest.fixture
    def binary_task(self):
        """Create a binary classification task."""
        data = pd.DataFrame({
            "x1": [1, 2, 3, 4],
            "x2": [4, 3, 2, 1],
            "y": ["cat", "dog", "cat", "dog"],
        })
        return TaskClassif(data, target="y", positive="dog")
    
    @pytest.fixture
    def multiclass_task(self):
        """Create a multiclass task."""
        data = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6],
            "y": ["A", "B", "C", "A", "B", "C"],
        })
        return TaskClassif(data, target="y")
    
    def test_response_only_prediction(self, binary_task):
        """Test prediction with response only."""
        row_ids = [0, 1, 2, 3]
        truth = np.array(["cat", "dog", "cat", "dog"])
        response = np.array(["cat", "cat", "cat", "dog"])
        
        pred = PredictionClassif(
            row_ids=row_ids,
            truth=truth,
            response=response,
            task=binary_task
        )
        
        assert pred.n == 4
        assert pred.has_truth
        assert pred.predict_types["response"]
        assert not pred.predict_types["prob"]
        
        # Get response
        assert np.array_equal(pred.get_response(), response)
        
        # Should error on prob
        with pytest.raises(RuntimeError, match="No probabilities"):
            pred.get_prob()
    
    def test_prob_only_prediction(self, binary_task):
        """Test prediction with probabilities only."""
        row_ids = [0, 1, 2, 3]
        prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])
        
        pred = PredictionClassif(
            row_ids=row_ids,
            prob=prob,
            task=binary_task
        )
        
        assert not pred.has_truth
        assert not pred.predict_types["response"]  # Not explicitly provided
        assert pred.predict_types["prob"]
        
        # Get probabilities
        assert np.array_equal(pred.get_prob(), prob)
        
        # Get specific class prob
        dog_prob = pred.get_prob("dog")
        assert np.array_equal(dog_prob, prob[:, 1])
        
        # Get response from prob
        response = pred.get_response()
        expected = ["cat", "dog", "cat", "dog"]  # Based on max prob
        assert list(response) == expected
    
    def test_binary_prob_1d(self, binary_task):
        """Test binary classification with 1D probabilities."""
        row_ids = [0, 1, 2, 3]
        # 1D prob for positive class
        prob_1d = np.array([0.2, 0.7, 0.4, 0.9])
        
        pred = PredictionClassif(
            row_ids=row_ids,
            prob=prob_1d,
            task=binary_task
        )
        
        # Should be expanded to 2D
        assert pred.prob.shape == (4, 2)
        assert np.allclose(pred.prob[:, 1], prob_1d)
        assert np.allclose(pred.prob[:, 0], 1 - prob_1d)
    
    def test_confusion_matrix(self, multiclass_task):
        """Test confusion matrix calculation."""
        row_ids = list(range(6))
        truth = np.array(["A", "B", "C", "A", "B", "C"])
        response = np.array(["A", "B", "C", "B", "B", "A"])  # Some errors
        
        pred = PredictionClassif(
            row_ids=row_ids,
            truth=truth,
            response=response,
            task=multiclass_task
        )
        
        cm = pred.confusion_matrix()
        assert cm is not None
        assert cm.shape == (3, 3)
        
        # Check specific values
        assert cm[0, 0] == 1  # A->A correct
        assert cm[0, 1] == 1  # A->B error
        assert cm[1, 1] == 2  # B->B correct (including one misclassified A)
        assert cm[2, 2] == 1  # C->C correct
        assert cm[2, 0] == 1  # C->A error
    
    def test_as_data_frame(self, binary_task):
        """Test DataFrame conversion."""
        row_ids = [10, 20, 30, 40]
        truth = np.array(["cat", "dog", "cat", "dog"])
        response = np.array(["cat", "cat", "dog", "dog"])
        prob = np.array([[0.7, 0.3], [0.6, 0.4], [0.2, 0.8], [0.1, 0.9]])
        
        pred = PredictionClassif(
            row_ids=row_ids,
            truth=truth,
            response=response,
            prob=prob,
            task=binary_task
        )
        
        df = pred.as_data_frame()
        
        assert len(df) == 4
        assert list(df["row_id"]) == row_ids
        assert list(df["truth"]) == list(truth)
        assert list(df["response"]) == list(response)
        assert "prob.cat" in df.columns
        assert "prob.dog" in df.columns
        assert np.allclose(df["prob.cat"], prob[:, 0])
        assert np.allclose(df["prob.dog"], prob[:, 1])
    
    def test_invalid_inputs(self, binary_task):
        """Test validation of invalid inputs."""
        # No predictions provided
        with pytest.raises(ValueError, match="At least one"):
            PredictionClassif(row_ids=[1, 2])
        
        # Length mismatch
        with pytest.raises(ValueError, match="Length mismatch"):
            PredictionClassif(
                row_ids=[1, 2],
                response=["A", "B", "C"]  # 3 vs 2
            )
        
        # Wrong prob shape
        with pytest.raises(ValueError, match="Shape mismatch"):
            PredictionClassif(
                row_ids=[1, 2],
                prob=np.array([[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]]),  # 3 vs 2
                task=binary_task
            )


class TestPredictionRegr:
    """Test regression predictions."""
    
    @pytest.fixture
    def regr_task(self):
        """Create a regression task."""
        data = pd.DataFrame({
            "x1": [1, 2, 3, 4, 5],
            "x2": [5, 4, 3, 2, 1],
            "y": [1.5, 2.5, 3.5, 4.5, 5.5],
        })
        return TaskRegr(data, target="y")
    
    def test_basic_prediction(self, regr_task):
        """Test basic regression prediction."""
        row_ids = [0, 1, 2, 3, 4]
        truth = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        response = np.array([1.6, 2.4, 3.6, 4.4, 5.6])
        
        pred = PredictionRegr(
            row_ids=row_ids,
            truth=truth,
            response=response,
            task=regr_task
        )
        
        assert pred.n == 5
        assert pred.has_truth
        assert pred.predict_types["response"]
        assert not pred.predict_types["se"]
        
        # Check response
        assert np.array_equal(pred.response, response)
    
    def test_residuals(self, regr_task):
        """Test residual calculation."""
        row_ids = [0, 1, 2]
        truth = np.array([2.0, 4.0, 6.0])
        response = np.array([2.1, 3.8, 6.3])
        
        pred = PredictionRegr(
            row_ids=row_ids,
            truth=truth,
            response=response,
            task=regr_task
        )
        
        residuals = pred.residuals()
        assert residuals is not None
        expected = truth - response
        assert np.allclose(residuals, expected)
        
        # Absolute residuals
        abs_residuals = pred.abs_residuals()
        assert np.allclose(abs_residuals, np.abs(expected))
    
    def test_prediction_with_se(self, regr_task):
        """Test predictions with standard errors."""
        row_ids = [0, 1, 2]
        response = np.array([2.0, 4.0, 6.0])
        se = np.array([0.1, 0.2, 0.15])
        
        pred = PredictionRegr(
            row_ids=row_ids,
            response=response,
            se=se,
            task=regr_task
        )
        
        assert pred.predict_types["se"]
        assert np.array_equal(pred.se, se)
        
        # Prediction intervals
        intervals = pred.prediction_intervals(alpha=0.05)
        assert intervals is not None
        assert "lower" in intervals
        assert "upper" in intervals
        assert all(intervals["lower"] < response)
        assert all(intervals["upper"] > response)
    
    def test_no_truth(self, regr_task):
        """Test prediction without truth values."""
        pred = PredictionRegr(
            row_ids=[0, 1],
            response=[2.0, 4.0],
            task=regr_task
        )
        
        assert not pred.has_truth
        assert pred.residuals() is None
        assert pred.abs_residuals() is None
    
    def test_as_data_frame(self, regr_task):
        """Test DataFrame conversion."""
        row_ids = [10, 20, 30]
        truth = np.array([1.0, 2.0, 3.0])
        response = np.array([1.1, 1.9, 3.2])
        se = np.array([0.1, 0.15, 0.12])
        
        pred = PredictionRegr(
            row_ids=row_ids,
            truth=truth,
            response=response,
            se=se,
            task=regr_task
        )
        
        df = pred.as_data_frame()
        
        assert len(df) == 3
        assert list(df["row_id"]) == row_ids
        assert np.array_equal(df["truth"], truth)
        assert np.array_equal(df["response"], response)
        assert np.array_equal(df["se"], se)
    
    def test_validation(self, regr_task):
        """Test input validation."""
        # Response is required
        with pytest.raises(ValueError, match="Response predictions are required"):
            PredictionRegr(row_ids=[1, 2], task=regr_task)
        
        # Length validation
        with pytest.raises(ValueError, match="Length mismatch"):
            PredictionRegr(
                row_ids=[1, 2],
                response=[1.0, 2.0, 3.0]  # Wrong length
            )