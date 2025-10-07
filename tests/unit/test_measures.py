"""Fixed unit tests for measures."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from mlpy.measures import (
    Measure, MeasureClassif, MeasureRegr,
    MeasureClassifAccuracy, MeasureClassifCE, MeasureClassifAUC,
    MeasureClassifLogLoss, MeasureClassifF1, MeasureClassifPrecision,
    MeasureClassifRecall, MeasureRegrMSE, MeasureRegrRMSE,
    MeasureRegrMAE, MeasureRegrMAPE, MeasureRegrR2,
    MeasureRegrMedianAE, MeasureRegrMSLE, MeasureRegrRMSLE
)
from mlpy.predictions import PredictionClassif, PredictionRegr
from mlpy.utils.registry import mlpy_measures


class TestMeasureBase:
    """Test base Measure functionality."""
    
    def test_measure_properties(self):
        """Test measure basic properties."""
        measure = MeasureClassifAccuracy()
        
        assert measure.id == "classif.acc"
        assert measure.task_type == "classif"
        assert measure.minimize is False
        assert measure.range == (0, 1)
        
    def test_measure_registry(self):
        """Test measure registration."""
        # Check some common measures are registered
        assert "classif.acc" in mlpy_measures
        assert "classif.auc" in mlpy_measures
        assert "regr.mse" in mlpy_measures
        assert "regr.rmse" in mlpy_measures
        
    def test_measure_validation(self):
        """Test measure validation with wrong prediction type."""
        measure_classif = MeasureClassifAccuracy()
        measure_regr = MeasureRegrMSE()
        
        # Create mock task and predictions
        mock_task_classif = Mock()
        mock_task_classif.task_type = "classif"
        mock_task_regr = Mock()
        mock_task_regr.task_type = "regr"
        
        pred_classif = PredictionClassif(
            task=mock_task_classif,
            learner_id="test",
            row_ids=[0, 1],
            truth=np.array(['a', 'b']),
            response=np.array(['a', 'b'])
        )
        pred_regr = PredictionRegr(
            task=mock_task_regr,
            learner_id="test",
            row_ids=[0, 1],
            truth=np.array([1.0, 2.0]),
            response=np.array([1.1, 2.1])
        )
        
        # Correct types should work
        measure_classif.score(pred_classif)
        measure_regr.score(pred_regr)
        
        # Wrong types should raise TypeError
        with pytest.raises(TypeError):
            measure_classif.score(pred_regr)
        with pytest.raises(TypeError):
            measure_regr.score(pred_classif)
            
    def test_aggregate(self):
        """Test measure aggregation."""
        measure = MeasureClassifAccuracy()
        scores = [0.8, 0.85, 0.9, 0.82, 0.88]
        
        agg = measure.aggregate(scores)
        
        assert agg["mean"] == pytest.approx(0.85)
        assert agg["std"] == pytest.approx(np.std(scores, ddof=1))
        assert agg["min"] == 0.8
        assert agg["max"] == 0.9
        assert agg["median"] == 0.85
        
        
class TestClassificationMeasures:
    """Test classification measures."""
    
    @pytest.fixture
    def mock_task(self):
        """Create mock classification task."""
        mock = Mock()
        mock.task_type = "classif"
        mock.class_names = ['cat', 'dog']
        mock.positive = 'cat'
        return mock
    
    @pytest.fixture
    def binary_prediction(self, mock_task):
        """Create binary classification prediction."""
        return PredictionClassif(
            task=mock_task,
            learner_id="test",
            row_ids=[0, 1, 2, 3, 4],
            response=np.array(['cat', 'dog', 'cat', 'cat', 'dog']),
            truth=np.array(['cat', 'dog', 'dog', 'cat', 'dog'])
        )
        
    @pytest.fixture
    def binary_prediction_with_prob(self, mock_task):
        """Create binary classification with probabilities."""
        prob_df = pd.DataFrame({
            'cat': [0.7, 0.3, 0.6, 0.8, 0.2],
            'dog': [0.3, 0.7, 0.4, 0.2, 0.8]
        })
        return PredictionClassif(
            task=mock_task,
            learner_id="test",
            row_ids=[0, 1, 2, 3, 4],
            response=np.array(['cat', 'dog', 'cat', 'cat', 'dog']),
            truth=np.array(['cat', 'dog', 'dog', 'cat', 'dog']),
            prob=prob_df
        )
        
    @pytest.fixture
    def multiclass_prediction(self):
        """Create multiclass prediction."""
        mock_task = Mock()
        mock_task.task_type = "classif"
        mock_task.class_names = ['a', 'b', 'c']
        
        return PredictionClassif(
            task=mock_task,
            learner_id="test",
            row_ids=[0, 1, 2, 3, 4, 5],
            response=np.array(['a', 'b', 'c', 'a', 'b', 'c']),
            truth=np.array(['a', 'b', 'c', 'a', 'c', 'b'])
        )
        
    def test_accuracy(self, binary_prediction):
        """Test accuracy measure."""
        measure = MeasureClassifAccuracy()
        score = measure.score(binary_prediction)
        
        # 4 out of 5 correct
        assert score == 0.8
        
        # Test with normalize=False
        measure_count = MeasureClassifAccuracy(normalize=False)
        score_count = measure_count.score(binary_prediction)
        assert score_count == 4
        
    def test_classification_error(self, binary_prediction):
        """Test classification error."""
        measure = MeasureClassifCE()
        score = measure.score(binary_prediction)
        
        # 1 out of 5 incorrect
        assert score == 0.2
        
    def test_auc(self, binary_prediction_with_prob):
        """Test AUC measure."""
        measure = MeasureClassifAUC()
        score = measure.score(binary_prediction_with_prob)
        
        # Should be between 0 and 1
        assert 0 <= score <= 1
        
    def test_logloss(self, binary_prediction_with_prob):
        """Test log loss measure."""
        measure = MeasureClassifLogLoss()
        score = measure.score(binary_prediction_with_prob)
        
        # Should be positive
        assert score >= 0
        
    def test_f1_score(self, binary_prediction):
        """Test F1 score."""
        measure = MeasureClassifF1()
        score = measure.score(binary_prediction)
        
        # Should be between 0 and 1
        assert 0 <= score <= 1
        
    def test_precision_recall(self, binary_prediction):
        """Test precision and recall."""
        precision = MeasureClassifPrecision()
        recall = MeasureClassifRecall()
        
        prec_score = precision.score(binary_prediction)
        rec_score = recall.score(binary_prediction)
        
        # Both should be between 0 and 1
        assert 0 <= prec_score <= 1
        assert 0 <= rec_score <= 1
        
    def test_multiclass_averaging(self, multiclass_prediction):
        """Test multiclass averaging options."""
        # Test different averaging methods
        f1_micro = MeasureClassifF1(average='micro')
        f1_macro = MeasureClassifF1(average='macro')
        f1_weighted = MeasureClassifF1(average='weighted')
        
        scores = [
            f1_micro.score(multiclass_prediction),
            f1_macro.score(multiclass_prediction),
            f1_weighted.score(multiclass_prediction)
        ]
        
        # All should be valid scores
        for score in scores:
            assert 0 <= score <= 1
            
    def test_missing_values(self, mock_task):
        """Test handling of missing values."""
        # Create prediction with missing truth
        pred = PredictionClassif(
            task=mock_task,
            learner_id="test",
            row_ids=[0, 1, 2, 3],
            response=np.array(['cat', 'dog', 'cat', 'dog']),
            truth=np.array(['cat', None, 'dog', 'dog'])
        )
        
        measure = MeasureClassifAccuracy()
        score = measure.score(pred)
        
        # Should handle missing values gracefully
        assert isinstance(score, float)
        

class TestRegressionMeasures:
    """Test regression measures."""
    
    @pytest.fixture
    def mock_task(self):
        """Create mock regression task."""
        mock = Mock()
        mock.task_type = "regr"
        return mock
        
    @pytest.fixture
    def regression_prediction(self, mock_task):
        """Create regression prediction."""
        return PredictionRegr(
            task=mock_task,
            learner_id="test",
            row_ids=[0, 1, 2, 3, 4],
            truth=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            response=np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        )
        
    @pytest.fixture
    def perfect_prediction(self, mock_task):
        """Create perfect regression prediction."""
        return PredictionRegr(
            task=mock_task,
            learner_id="test",
            row_ids=[0, 1, 2],
            truth=np.array([1.0, 2.0, 3.0]),
            response=np.array([1.0, 2.0, 3.0])
        )
        
    def test_mse(self, regression_prediction):
        """Test Mean Squared Error."""
        measure = MeasureRegrMSE()
        score = measure.score(regression_prediction)
        
        # Calculate expected MSE
        truth = regression_prediction.truth
        pred = regression_prediction.response
        expected = np.mean((truth - pred) ** 2)
        
        assert score == pytest.approx(expected)
        
    def test_rmse(self, regression_prediction):
        """Test Root Mean Squared Error."""
        measure = MeasureRegrRMSE()
        score = measure.score(regression_prediction)
        
        # RMSE should be sqrt of MSE
        mse = MeasureRegrMSE().score(regression_prediction)
        assert score == pytest.approx(np.sqrt(mse))
        
    def test_mae(self, regression_prediction):
        """Test Mean Absolute Error."""
        measure = MeasureRegrMAE()
        score = measure.score(regression_prediction)
        
        # Calculate expected MAE
        truth = regression_prediction.truth
        pred = regression_prediction.response
        expected = np.mean(np.abs(truth - pred))
        
        assert score == pytest.approx(expected)
        
    def test_mape(self):
        """Test Mean Absolute Percentage Error."""
        # Create prediction without zeros in truth
        mock_task = Mock()
        mock_task.task_type = "regr"
        
        pred = PredictionRegr(
            task=mock_task,
            learner_id="test",
            row_ids=[0, 1, 2],
            truth=np.array([100.0, 200.0, 300.0]),
            response=np.array([110.0, 190.0, 320.0])
        )
        
        measure = MeasureRegrMAPE()
        score = measure.score(pred)
        
        # Should be positive percentage
        assert score > 0
        
    def test_r2(self, perfect_prediction):
        """Test R-squared score."""
        measure = MeasureRegrR2()
        
        # Perfect prediction should have R2 = 1
        score = measure.score(perfect_prediction)
        assert score == pytest.approx(1.0)
        
        # Create worse prediction
        mock_task = Mock()
        mock_task.task_type = "regr"
        
        bad_pred = PredictionRegr(
            task=mock_task,
            learner_id="test",
            row_ids=[0, 1, 2],
            truth=np.array([1.0, 2.0, 3.0]),
            response=np.array([3.0, 1.0, 2.0])  # Bad predictions
        )
        
        bad_score = measure.score(bad_pred)
        assert bad_score < 1.0
        
    def test_median_ae(self, regression_prediction):
        """Test Median Absolute Error."""
        measure = MeasureRegrMedianAE()
        score = measure.score(regression_prediction)
        
        # Calculate expected MedianAE
        truth = regression_prediction.truth
        pred = regression_prediction.response
        expected = np.median(np.abs(truth - pred))
        
        assert score == pytest.approx(expected)
        
    def test_msle(self):
        """Test Mean Squared Logarithmic Error."""
        # Create prediction with positive values only
        mock_task = Mock()
        mock_task.task_type = "regr"
        
        pred = PredictionRegr(
            task=mock_task,
            learner_id="test",
            row_ids=[0, 1, 2],
            truth=np.array([1.0, 2.0, 3.0]),
            response=np.array([1.1, 2.2, 2.8])
        )
        
        measure = MeasureRegrMSLE()
        score = measure.score(pred)
        
        # Should be positive
        assert score >= 0
        
    def test_missing_truth(self):
        """Test handling of missing truth values."""
        mock_task = Mock()
        mock_task.task_type = "regr"
        
        # Create prediction with NaN in truth
        pred = PredictionRegr(
            task=mock_task,
            learner_id="test",
            row_ids=[0, 1, 2, 3],
            truth=np.array([1.0, np.nan, 3.0, 4.0]),
            response=np.array([1.1, 2.0, 3.1, 3.9])
        )
        
        measure = MeasureRegrMSE()
        score = measure.score(pred)
        
        # Should handle NaN gracefully
        assert isinstance(score, float)
        assert not np.isnan(score)


class TestMeasureAliases:
    """Test measure alias functionality."""
    
    def test_get_from_registry(self):
        """Test getting measures from registry."""
        # Test various aliases
        acc = mlpy_measures.get("classif.acc")
        assert isinstance(acc, MeasureClassifAccuracy)
        assert acc.id == "classif.acc"
        
        rmse = mlpy_measures.get("regr.rmse")
        assert isinstance(rmse, MeasureRegrRMSE)
        assert rmse.id == "regr.rmse"
        
        # Test non-existent measure
        assert mlpy_measures.get("fake.measure") is None