"""Unit tests for parallel execution functionality."""

import pytest
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from mlpy.parallel import (
    Backend, BackendSequential, BackendThreading,
    BackendMultiprocessing, get_backend, set_backend,
    backend_context, parallel_map, parallel_starmap
)
from mlpy.tasks import TaskClassif
from mlpy.learners import learner_sklearn
from mlpy.resamplings import ResamplingCV, ResamplingHoldout
from mlpy.measures import MeasureClassifAccuracy
from mlpy.resample import resample
from mlpy.benchmark import benchmark


# Test functions for parallel execution
def slow_square(x):
    """Slow function for testing parallelism."""
    time.sleep(0.01)  # Simulate work
    return x ** 2


def slow_add(a, b):
    """Slow function with multiple args."""
    time.sleep(0.01)
    return a + b


def failing_function(x):
    """Function that fails for testing error handling."""
    if x == 3:
        raise ValueError("Failed on purpose")
    return x


class TestBackends:
    """Test backend implementations."""
    
    def test_sequential_backend(self):
        """Test sequential backend."""
        backend = BackendSequential()
        
        # Test map
        results = backend.map(lambda x: x**2, [1, 2, 3, 4])
        assert results == [1, 4, 9, 16]
        
        # Test starmap
        results = backend.starmap(lambda a, b: a + b, [(1, 2), (3, 4)])
        assert results == [3, 7]
        
    def test_threading_backend(self):
        """Test threading backend."""
        backend = BackendThreading(n_jobs=2)
        
        # Test map
        data = list(range(10))
        results = backend.map(slow_square, data)
        assert results == [x**2 for x in data]
        
        # Test starmap
        args = [(i, i+1) for i in range(5)]
        results = backend.starmap(slow_add, args)
        assert results == [a + b for a, b in args]
        
        # Cleanup
        backend.close()
        
    def test_multiprocessing_backend(self):
        """Test multiprocessing backend."""
        backend = BackendMultiprocessing(n_jobs=2)
        
        # Test map
        data = list(range(10))
        results = backend.map(slow_square, data)
        assert results == [x**2 for x in data]
        
        # Test starmap
        args = [(i, i+1) for i in range(5)]
        results = backend.starmap(slow_add, args)
        assert results == [a + b for a, b in args]
        
        # Cleanup
        backend.close()
        
    @pytest.mark.skipif(True, reason="Joblib tests require joblib installation")
    def test_joblib_backend(self):
        """Test joblib backend (if available)."""
        try:
            from mlpy.parallel import BackendJoblib
        except ImportError:
            pytest.skip("Joblib not available")
            
        backend = BackendJoblib(n_jobs=2)
        
        # Test map
        data = list(range(10))
        results = backend.map(slow_square, data)
        assert results == [x**2 for x in data]
        
    def test_backend_error_handling(self):
        """Test error handling in backends."""
        backend = BackendSequential()
        
        # Sequential should warn but continue
        results = backend.map(failing_function, [1, 2, 3, 4], verbose=0)
        assert results[0] == 1
        assert results[1] == 2
        assert results[2] is None  # Failed
        assert results[3] == 4
        
    def test_n_jobs_normalization(self):
        """Test n_jobs parameter normalization."""
        # -1 should use all cores
        backend = BackendThreading(n_jobs=-1)
        assert backend.n_jobs > 0
        
        # Positive values should be preserved
        backend = BackendThreading(n_jobs=4)
        assert backend.n_jobs == 4
        
        # Invalid values should raise
        with pytest.raises(ValueError):
            BackendThreading(n_jobs=0)


class TestParallelUtils:
    """Test parallel utility functions."""
    
    def test_get_set_backend(self):
        """Test getting and setting default backend."""
        # Default should be sequential
        default = get_backend()
        assert isinstance(default, BackendSequential)
        
        # Set new backend
        new_backend = BackendThreading(n_jobs=2)
        set_backend(new_backend)
        assert get_backend() is new_backend
        
        # Set by string
        set_backend("sequential")
        assert isinstance(get_backend(), BackendSequential)
        
        # Cleanup
        new_backend.close()
        
    def test_backend_context(self):
        """Test backend context manager."""
        original = get_backend()
        
        with backend_context("threading") as backend:
            assert isinstance(backend, BackendThreading)
            assert isinstance(get_backend(), BackendThreading)
            
        # Should restore original
        assert get_backend() is original
        
    def test_parallel_map(self):
        """Test parallel_map utility."""
        # With explicit backend
        results = parallel_map(
            lambda x: x**2,
            [1, 2, 3, 4],
            backend="threading"
        )
        assert results == [1, 4, 9, 16]
        
        # With default backend
        results = parallel_map(lambda x: x**2, [1, 2, 3, 4])
        assert results == [1, 4, 9, 16]
        
    def test_parallel_starmap(self):
        """Test parallel_starmap utility."""
        results = parallel_starmap(
            lambda a, b: a + b,
            [(1, 2), (3, 4), (5, 6)],
            backend="threading"
        )
        assert results == [3, 7, 11]


class TestParallelResample:
    """Test parallel execution in resample."""
    
    @pytest.fixture
    def iris_task(self):
        """Create Iris classification task."""
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        return TaskClassif(data=df, target='species')
        
    def test_resample_sequential(self, iris_task):
        """Test sequential resampling (baseline)."""
        learner = learner_sklearn(DecisionTreeClassifier(random_state=42))
        
        result = resample(
            task=iris_task,
            learner=learner,
            resampling=ResamplingCV(folds=5),
            measures=MeasureClassifAccuracy(),
            backend=None  # Sequential
        )
        
        assert result.n_iters == 5
        assert all(s > 0.8 for s in result.scores['classif.acc'])
        
    @pytest.mark.slow
    def test_resample_parallel_threading(self, iris_task):
        """Test parallel resampling with threading."""
        learner = learner_sklearn(DecisionTreeClassifier(random_state=42))
        backend = BackendThreading(n_jobs=2)
        
        result = resample(
            task=iris_task,
            learner=learner,
            resampling=ResamplingCV(folds=5),
            measures=MeasureClassifAccuracy(),
            backend=backend
        )
        
        assert result.n_iters == 5
        assert all(s > 0.8 for s in result.scores['classif.acc'])
        
        backend.close()
        
    @pytest.mark.slow
    def test_resample_parallel_multiprocessing(self, iris_task):
        """Test parallel resampling with multiprocessing."""
        learner = learner_sklearn(DecisionTreeClassifier(random_state=42))
        backend = BackendMultiprocessing(n_jobs=2)
        
        result = resample(
            task=iris_task,
            learner=learner,
            resampling=ResamplingCV(folds=5),
            measures=MeasureClassifAccuracy(),
            backend=backend
        )
        
        assert result.n_iters == 5
        assert all(s > 0.8 for s in result.scores['classif.acc'])
        
        backend.close()
        
    def test_resample_parallel_consistency(self, iris_task):
        """Test that parallel and sequential give same results."""
        learner = learner_sklearn(RandomForestClassifier(n_estimators=10, random_state=42))
        resampling = ResamplingCV(folds=3, seed=42)
        measure = MeasureClassifAccuracy()
        
        # Sequential
        result_seq = resample(
            task=iris_task,
            learner=learner,
            resampling=resampling,
            measures=measure,
            backend=None
        )
        
        # Parallel
        backend = BackendThreading(n_jobs=2)
        result_par = resample(
            task=iris_task,
            learner=learner,
            resampling=resampling,
            measures=measure,
            backend=backend
        )
        backend.close()
        
        # Results should be very similar (allowing for minor floating point differences)
        seq_scores = result_seq.scores['classif.acc']
        par_scores = result_par.scores['classif.acc']
        
        assert len(seq_scores) == len(par_scores)
        for s1, s2 in zip(seq_scores, par_scores):
            assert abs(s1 - s2) < 0.01  # Allow small differences


class TestParallelBenchmark:
    """Test parallel execution in benchmark."""
    
    @pytest.fixture
    def tasks(self):
        """Create multiple tasks."""
        # Create two simple tasks
        np.random.seed(42)
        
        # Task 1
        df1 = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'y': np.random.choice([0, 1], 100)
        })
        task1 = TaskClassif(data=df1, target='y', id='task1')
        
        # Task 2
        df2 = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100),
            'y': np.random.choice([0, 1, 2], 100)
        })
        task2 = TaskClassif(data=df2, target='y', id='task2')
        
        return [task1, task2]
        
    @pytest.fixture
    def learners(self):
        """Create multiple learners."""
        return [
            learner_sklearn(DecisionTreeClassifier(max_depth=2), id='dt2'),
            learner_sklearn(DecisionTreeClassifier(max_depth=5), id='dt5'),
            learner_sklearn(RandomForestClassifier(n_estimators=10), id='rf10')
        ]
        
    def test_benchmark_sequential(self, tasks, learners):
        """Test sequential benchmark."""
        result = benchmark(
            tasks=tasks,
            learners=learners,
            resampling=ResamplingHoldout(ratio=0.7),
            measures=MeasureClassifAccuracy(),
            backend=None
        )
        
        assert result.n_experiments == 6  # 2 tasks × 3 learners
        assert result.n_successful == 6
        assert result.n_errors == 0
        
    @pytest.mark.slow
    def test_benchmark_parallel(self, tasks, learners):
        """Test parallel benchmark."""
        backend = BackendThreading(n_jobs=2)
        
        result = benchmark(
            tasks=tasks,
            learners=learners,
            resampling=ResamplingHoldout(ratio=0.7),
            measures=MeasureClassifAccuracy(),
            backend=backend
        )
        
        assert result.n_experiments == 6
        assert result.n_successful == 6
        assert result.n_errors == 0
        
        backend.close()
        
    def test_benchmark_parallel_with_errors(self, tasks):
        """Test parallel benchmark with failing learner."""
        # Create a learner that will fail
        from sklearn.base import BaseEstimator, ClassifierMixin
        
        class FailingEstimator(BaseEstimator, ClassifierMixin):
            def fit(self, X, y):
                raise RuntimeError("Intentional failure")
            def predict(self, X):
                return np.zeros(len(X))
                
        learners = [
            learner_sklearn(DecisionTreeClassifier(), id='good'),
            learner_sklearn(FailingEstimator(), id='bad')
        ]
        
        backend = BackendThreading(n_jobs=2)
        
        result = benchmark(
            tasks=tasks,
            learners=learners,
            resampling=ResamplingHoldout(),
            measures=MeasureClassifAccuracy(),
            backend=backend
        )
        
        assert result.n_experiments == 4  # 2 tasks × 2 learners
        
        # Check that we have both successful and failed experiments
        # The good learner should succeed, bad learner should have errors during training
        good_results = [result.get_result(task.id, 'good') for task in tasks]
        bad_results = [result.get_result(task.id, 'bad') for task in tasks]
        
        # Good learner should have valid results
        assert all(res is not None for res in good_results)
        
        # Bad learner should have failed during training (NaN scores from resample errors)
        for res in bad_results:
            if res is not None:
                # If resample completed (despite training errors), check for NaN scores
                assert any(np.isnan(score) for score in res.scores['classif.acc'])
        
        backend.close()


class TestPerformance:
    """Test performance improvements with parallelization."""
    
    @pytest.mark.slow
    def test_parallel_speedup(self):
        """Test that parallel execution is faster."""
        # Create a slow function
        def slow_computation(x):
            time.sleep(0.1)  # 100ms per item
            return x ** 2
            
        data = list(range(8))
        
        # Sequential timing
        start = time.time()
        results_seq = parallel_map(slow_computation, data, backend="sequential")
        time_seq = time.time() - start
        
        # Parallel timing (2 workers)
        backend = BackendThreading(n_jobs=2)
        start = time.time()
        results_par = parallel_map(slow_computation, data, backend=backend)
        time_par = time.time() - start
        backend.close()
        
        # Check results are same
        assert results_seq == results_par
        
        # Parallel should be faster (at least 1.5x speedup)
        assert time_par < time_seq * 0.7
        
        print(f"Sequential: {time_seq:.2f}s, Parallel: {time_par:.2f}s")
        print(f"Speedup: {time_seq / time_par:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__])