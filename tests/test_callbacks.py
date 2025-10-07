"""Test callback system."""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from mlpy.callbacks import (
    Callback, CallbackSet,
    CallbackHistory, CallbackLogger, CallbackProgress,
    CallbackTimer, CallbackEarlyStopping, CallbackCheckpoint
)
from mlpy.tasks import TaskClassif
from mlpy.learners import Learner
from mlpy.resamplings import ResamplingHoldout, ResamplingCV
from mlpy.measures import MeasureClassifAccuracy
from mlpy.resample import resample
from mlpy.benchmark import benchmark


class TrackerCallback(Callback):
    """Test callback that tracks method calls."""
    
    def __init__(self):
        super().__init__(id="tracker")
        self.calls = []
        
    def on_resample_begin(self, task, learner, resampling):
        self.calls.append(("resample_begin", task.id, learner.id))
        
    def on_iteration_begin(self, iteration, train_set, test_set):
        self.calls.append(("iteration_begin", iteration, len(train_set), len(test_set)))
        
    def on_train_begin(self, task, learner):
        self.calls.append(("train_begin", task.id, learner.id))
        
    def on_train_end(self, learner):
        self.calls.append(("train_end", learner.id))
        
    def on_predict_begin(self, task, learner):
        self.calls.append(("predict_begin", task.id, learner.id))
        
    def on_predict_end(self, prediction):
        self.calls.append(("predict_end", len(prediction.row_ids)))
        
    def on_iteration_end(self, iteration, scores, train_time, predict_time):
        self.calls.append(("iteration_end", iteration, list(scores.keys())))
        
    def on_resample_end(self, result):
        self.calls.append(("resample_end", result.n_iters))
        
    def on_benchmark_begin(self, tasks, learners):
        self.calls.append(("benchmark_begin", len(tasks), len(learners)))
        
    def on_experiment_begin(self, task, learner, experiment_num):
        self.calls.append(("experiment_begin", task.id, learner.id, experiment_num))
        
    def on_experiment_end(self, task, learner, result, error):
        self.calls.append(("experiment_end", task.id, learner.id, error is None))
        
    def on_benchmark_end(self, result):
        self.calls.append(("benchmark_end", result.n_experiments))
        
    def on_tune_begin(self, learner, param_set, n_configs):
        self.calls.append(("tune_begin", learner.id, n_configs))
        
    def on_config_begin(self, config_num, config):
        self.calls.append(("config_begin", config_num, list(config.keys())))
        
    def on_config_end(self, config_num, score):
        self.calls.append(("config_end", config_num, score))
        
    def on_tune_end(self, result):
        self.calls.append(("tune_end", len(result.configs)))
        
    def on_error(self, error, context):
        self.calls.append(("error", type(error).__name__, context))


class DummyLearner(Learner):
    """Dummy learner for testing."""
    
    def __init__(self, id="dummy", fail=False):
        super().__init__(
            id=id,
            predict_type="response",
            feature_types=["numeric", "factor"],
            properties=["multiclass", "twoclass"],
            packages=[]
        )
        self.fail = fail
        self.model = None
        self._task_type = "classif"
        
    @property
    def task_type(self):
        return self._task_type
        
    def train(self, task, row_ids=None):
        if self.fail:
            raise ValueError("Dummy failure")
        self.model = {"trained": True}
        return self
        
    def predict(self, task, row_ids=None):
        if self.fail:
            raise ValueError("Dummy failure")
        n = len(row_ids) if row_ids is not None else task.nrow
        from mlpy.predictions import PredictionClassif
        
        # Generate predictions
        response = np.random.choice(task.class_names, n)
        
        return PredictionClassif(
            task=task,
            learner_id=self.id,
            row_ids=row_ids if row_ids is not None else list(range(task.nrow)),
            truth=task.truth(row_ids) if row_ids is not None else task.truth(),
            response=response
        )
        
    def clone(self):
        return DummyLearner(id=self.id, fail=self.fail)


def test_callback_set():
    """Test CallbackSet functionality."""
    cb1 = TrackerCallback()
    cb2 = TrackerCallback()
    
    callback_set = CallbackSet([cb1, cb2])
    
    # Create test data for task
    data = pd.DataFrame({
        'x1': [1, 2, 3],
        'x2': [4, 5, 6],
        'y': ['A', 'B', 'A']
    })
    
    # Test that callbacks are called
    callback_set.on_resample_begin(
        task=TaskClassif(data=data, target='y', id="test_task"),
        learner=DummyLearner(),
        resampling=ResamplingHoldout()
    )
    
    assert len(cb1.calls) == 1
    assert len(cb2.calls) == 1
    assert cb1.calls[0][0] == "resample_begin"
    assert cb2.calls[0][0] == "resample_begin"


def test_resample_with_callbacks():
    """Test callbacks in resample function."""
    # Create test data
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'y': np.random.choice(['A', 'B'], n)
    })
    
    task = TaskClassif(data=data, target='y')
    learner = DummyLearner()
    
    # Create tracker callback
    tracker = TrackerCallback()
    
    # Run resample with callback
    result = resample(
        task=task,
        learner=learner,
        resampling=ResamplingCV(folds=3),
        measures=MeasureClassifAccuracy(),
        callbacks=tracker
    )
    
    # Check that callbacks were called in correct order
    call_types = [call[0] for call in tracker.calls]
    
    # Should start with resample_begin
    assert call_types[0] == "resample_begin"
    
    # Should have 3 iterations (3-fold CV)
    iteration_begins = [c for c in call_types if c == "iteration_begin"]
    assert len(iteration_begins) == 3
    
    # Each iteration should follow pattern
    # Pattern: iteration_begin, train_begin, train_end, predict_begin, predict_end, iteration_end
    assert call_types.count("train_begin") == 3
    assert call_types.count("train_end") == 3
    assert call_types.count("predict_begin") == 3
    assert call_types.count("predict_end") == 3
    assert call_types.count("iteration_end") == 3
    
    # Should end with resample_end
    assert call_types[-1] == "resample_end"


def test_benchmark_with_callbacks():
    """Test callbacks in benchmark function."""
    # Create test data
    np.random.seed(42)
    n = 50
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'y': np.random.choice(['A', 'B'], n)
    })
    
    task = TaskClassif(data=data, target='y')
    learners = [DummyLearner(id="dummy1"), DummyLearner(id="dummy2")]
    
    # Create tracker callback
    tracker = TrackerCallback()
    
    # Run benchmark with callback
    result = benchmark(
        tasks=task,
        learners=learners,
        resampling=ResamplingHoldout(),
        measures=MeasureClassifAccuracy(),
        callbacks=tracker
    )
    
    # Check callbacks
    call_types = [call[0] for call in tracker.calls]
    
    # Should start with benchmark_begin
    assert call_types[0] == "benchmark_begin"
    assert tracker.calls[0][1] == 1  # 1 task
    assert tracker.calls[0][2] == 2  # 2 learners
    
    # Should have 2 experiments
    experiment_begins = [c for c in call_types if c == "experiment_begin"]
    assert len(experiment_begins) == 2
    
    # Should end with benchmark_end
    assert call_types[-1] == "benchmark_end"


def test_callback_history():
    """Test history callback."""
    # Create test data
    np.random.seed(42)
    n = 50
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'y': np.random.choice(['A', 'B'], n)
    })
    
    task = TaskClassif(data=data, target='y')
    learner = DummyLearner()
    
    # Create history callback
    history = CallbackHistory()
    
    # Run resample
    result = resample(
        task=task,
        learner=learner,
        resampling=ResamplingCV(folds=2),
        measures=MeasureClassifAccuracy(),
        callbacks=history
    )
    
    # Check history
    assert len(history.iterations) == 2
    assert all(hasattr(it, 'scores') for it in history.iterations)
    assert all(hasattr(it, 'train_time') for it in history.iterations)
    assert all(hasattr(it, 'predict_time') for it in history.iterations)
    
    # Get summary
    summary = history.get_iteration_summary()
    assert summary['n_iterations'] == 2
    assert summary['total_train_time'] >= 0
    assert summary['total_predict_time'] >= 0
    assert 'scores_summary' in summary


def test_callback_timer():
    """Test timer callback."""
    # Create test data
    np.random.seed(42)
    n = 50
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'y': np.random.choice(['A', 'B'], n)
    })
    
    task = TaskClassif(data=data, target='y')
    learner = DummyLearner()
    
    # Create timer callback
    timer = CallbackTimer()
    
    # Run resample
    result = resample(
        task=task,
        learner=learner,
        resampling=ResamplingHoldout(),
        measures=MeasureClassifAccuracy(),
        callbacks=timer
    )
    
    # Check timings
    timings = timer.get_experiment_times()
    assert 'resample' in timings
    assert timings['resample']['total'] > 0
    assert len(timings['resample']['iterations']) == 1


def test_callback_early_stopping():
    """Test early stopping callback."""
    # Create test data
    np.random.seed(42)
    n = 50
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'y': np.random.choice(['A', 'B'], n)
    })
    
    task = TaskClassif(data=data, target='y')
    learner = DummyLearner()
    
    # Create early stopping callback
    early_stop = CallbackEarlyStopping(patience=2, verbose=False)
    
    # Run resample
    result = resample(
        task=task,
        learner=learner,
        resampling=ResamplingCV(folds=5),
        measures=MeasureClassifAccuracy(),
        callbacks=early_stop
    )
    
    # Check that best score was tracked
    best_info = early_stop.get_best()
    assert 'best_score' in best_info
    assert 'best_iteration' in best_info


def test_callback_checkpoint():
    """Test checkpoint callback."""
    import tempfile
    import shutil
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test data
        np.random.seed(42)
        n = 50
        data = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'y': np.random.choice(['A', 'B'], n)
        })
        
        task = TaskClassif(data=data, target='y')
        learner = DummyLearner()
        
        # Create checkpoint callback
        checkpoint = CallbackCheckpoint(
            checkpoint_dir=temp_dir,
            save_freq=1
        )
        
        # Run resample
        result = resample(
            task=task,
            learner=learner,
            resampling=ResamplingCV(folds=2),
            measures=MeasureClassifAccuracy(),
            callbacks=checkpoint
        )
        
        # Check that checkpoints were created
        checkpoints = checkpoint.list_checkpoints()
        assert len(checkpoints) > 0
        
        # Should have init and final checkpoints
        assert any('init' in cp for cp in checkpoints)
        assert any('final' in cp for cp in checkpoints)
        
        # Load a checkpoint
        init_data = checkpoint.load_checkpoint(f"resample_{task.id}_{learner.id}_init")
        assert init_data['type'] == 'resample'
        assert init_data['status'] == 'started'
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_multiple_callbacks():
    """Test using multiple callbacks together."""
    # Create test data
    np.random.seed(42)
    n = 50
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'y': np.random.choice(['A', 'B'], n)
    })
    
    task = TaskClassif(data=data, target='y')
    learner = DummyLearner()
    
    # Create multiple callbacks
    tracker = TrackerCallback()
    history = CallbackHistory()
    timer = CallbackTimer()
    
    # Run resample with multiple callbacks
    result = resample(
        task=task,
        learner=learner,
        resampling=ResamplingCV(folds=3),
        measures=MeasureClassifAccuracy(),
        callbacks=[tracker, history, timer]
    )
    
    # Check all callbacks were used
    assert len(tracker.calls) > 0
    assert len(history.iterations) == 3
    assert timer.get_experiment_times()['resample']['total'] > 0


def test_callback_error_handling():
    """Test callback error handling."""
    # Create test data
    np.random.seed(42)
    n = 50
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'y': np.random.choice(['A', 'B'], n)
    })
    
    task = TaskClassif(data=data, target='y')
    learner = DummyLearner(fail=True)  # Will fail during training
    
    # Create tracker callback
    tracker = TrackerCallback()
    
    # Run resample - should handle errors gracefully
    result = resample(
        task=task,
        learner=learner,
        resampling=ResamplingCV(folds=2),
        measures=MeasureClassifAccuracy(),
        callbacks=tracker
    )
    
    # Check that errors were tracked
    error_calls = [c for c in tracker.calls if c[0] == "error"]
    assert len(error_calls) == 2  # One for each fold
    
    # Check result has errors
    assert result.n_errors == 2


if __name__ == "__main__":
    pytest.main([__file__])