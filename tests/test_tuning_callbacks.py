"""Test tuning with callbacks."""

import pytest
import numpy as np
import pandas as pd

from mlpy.tasks import TaskClassif
from mlpy.learners import Learner
from mlpy.resamplings import ResamplingHoldout
from mlpy.measures import MeasureClassifAccuracy
from mlpy.automl import (
    ParamInt, ParamFloat, ParamCategorical, ParamSet,
    TunerGrid, TunerRandom
)
from mlpy.callbacks import CallbackHistory, CallbackTimer, CallbackEarlyStopping
from tests.test_callbacks import DummyLearner, TrackerCallback


class TunableLearner(Learner):
    """Learner with tunable parameters for testing."""
    
    def __init__(self, id="tunable", alpha=1.0, beta=0.5, method="A"):
        super().__init__(
            id=id,
            predict_type="response",
            feature_types=["numeric", "factor"],
            properties=["multiclass", "twoclass"],
            packages=[]
        )
        self.alpha = alpha
        self.beta = beta
        self.method = method
        self.model = None
        self._task_type = "classif"
        
    @property
    def task_type(self):
        return self._task_type
        
    def train(self, task, row_ids=None):
        # Simulate training with parameters affecting performance
        self.model = {
            "trained": True,
            "alpha": self.alpha,
            "beta": self.beta,
            "method": self.method
        }
        return self
        
    def predict(self, task, row_ids=None):
        n = len(row_ids) if row_ids is not None else task.nrow
        from mlpy.predictions import PredictionClassif
        
        # Simulate performance based on parameters
        # Better performance with higher alpha, lower beta, and method B
        np.random.seed(int(self.alpha * 100 + self.beta * 10))
        accuracy = 0.5 + 0.1 * self.alpha - 0.1 * self.beta
        if self.method == "B":
            accuracy += 0.1
        accuracy = np.clip(accuracy, 0.0, 1.0)
        
        # Generate predictions based on accuracy
        truth = task.truth(row_ids) if row_ids is not None else task.truth()
        response = truth.copy()
        n_errors = int(n * (1 - accuracy))
        if n_errors > 0:
            error_idx = np.random.choice(n, n_errors, replace=False)
            classes = task.class_names
            for idx in error_idx:
                response[idx] = np.random.choice([c for c in classes if c != truth[idx]])
        
        return PredictionClassif(
            task=task,
            learner_id=self.id,
            row_ids=row_ids if row_ids is not None else list(range(task.nrow)),
            truth=truth,
            response=response
        )
        
    def clone(self):
        return TunableLearner(
            id=self.id,
            alpha=self.alpha,
            beta=self.beta,
            method=self.method
        )


def test_tuner_grid_with_callbacks():
    """Test grid search tuner with callbacks."""
    # Create test data
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'y': np.random.choice(['A', 'B'], n)
    })
    
    task = TaskClassif(data=data, target='y')
    learner = TunableLearner()
    
    # Define parameter space
    param_set = ParamSet([
        ParamFloat("alpha", lower=0.1, upper=2.0),
        ParamFloat("beta", lower=0.1, upper=1.0),
        ParamCategorical("method", values=["A", "B"])
    ])
    
    # Create callbacks
    tracker = TrackerCallback()
    history = CallbackHistory()
    timer = CallbackTimer()
    
    # Create tuner
    tuner = TunerGrid(resolution=3)
    
    # Run tuning with callbacks
    result = tuner.tune(
        learner=learner,
        task=task,
        resampling=ResamplingHoldout(),
        measure=MeasureClassifAccuracy(),
        param_set=param_set,
        callbacks=[tracker, history, timer]
    )
    
    # Check that tuning callbacks were called
    call_types = [call[0] for call in tracker.calls]
    
    # Should start with tune_begin
    assert call_types[0] == "tune_begin"
    assert tracker.calls[0][2] > 0  # n_configs
    
    # Should have config_begin and config_end for each config
    config_begins = [c for c in call_types if c == "config_begin"]
    config_ends = [c for c in call_types if c == "config_end"]
    assert len(config_begins) == len(config_ends)
    assert len(config_begins) > 0
    
    # Should end with tune_end
    assert call_types[-1] == "tune_end"
    
    # Check history
    assert len(history.configs) > 0
    
    # Check timer
    timings = timer.get_experiment_times()
    assert 'tuning' in timings
    assert timings['tuning']['total'] > 0
    
    # Check result
    assert result.best_config is not None
    assert result.best_score is not None


def test_tuner_random_with_callbacks():
    """Test random search tuner with callbacks."""
    # Create test data
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'y': np.random.choice(['A', 'B'], n)
    })
    
    task = TaskClassif(data=data, target='y')
    learner = TunableLearner()
    
    # Define parameter space
    param_set = ParamSet([
        ParamFloat("alpha", lower=0.1, upper=2.0),
        ParamFloat("beta", lower=0.1, upper=1.0),
        ParamCategorical("method", values=["A", "B"])
    ])
    
    # Create callbacks
    tracker = TrackerCallback()
    early_stop = CallbackEarlyStopping(patience=3, verbose=False)
    
    # Create tuner
    tuner = TunerRandom(n_evals=10, seed=42)
    
    # Run tuning with callbacks
    result = tuner.tune(
        learner=learner,
        task=task,
        resampling=ResamplingHoldout(),
        measure=MeasureClassifAccuracy(),
        param_set=param_set,
        callbacks=[tracker, early_stop]
    )
    
    # Check callbacks
    call_types = [call[0] for call in tracker.calls]
    assert "tune_begin" in call_types
    assert "tune_end" in call_types
    
    # Check early stopping tracked best
    best_info = early_stop.get_best()
    assert 'best_score' in best_info
    assert 'best_config' in best_info


def test_tuning_error_handling_with_callbacks():
    """Test tuning with callbacks - simplified test."""
    # Create test data
    np.random.seed(42)
    n = 50
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'y': np.random.choice(['A', 'B'], n)
    })
    
    task = TaskClassif(data=data, target='y')
    learner = TunableLearner()
    
    # Define parameter space
    param_set = ParamSet([
        ParamFloat("alpha", lower=0.1, upper=2.0),
        ParamFloat("beta", lower=0.1, upper=1.0)
    ])
    
    # Create tracker callback
    tracker = TrackerCallback()
    
    # Create tuner
    tuner = TunerGrid(resolution=3)
    
    # Run tuning with callbacks
    result = tuner.tune(
        learner=learner,
        task=task,
        resampling=ResamplingHoldout(),
        measure=MeasureClassifAccuracy(),
        param_set=param_set,
        callbacks=tracker
    )
    
    # Check that tuning completed successfully
    assert len(result.scores) > 0
    assert result.best_config is not None
    assert not np.isnan(result.best_score)
    
    # Check callbacks were called
    call_types = [call[0] for call in tracker.calls]
    assert "tune_begin" in call_types
    assert "tune_end" in call_types
    assert call_types.count("config_begin") == len(result.scores)
    assert call_types.count("config_end") == len(result.scores)


if __name__ == "__main__":
    pytest.main([__file__])