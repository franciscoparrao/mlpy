"""
Tests for advanced benchmarking system in MLPY.
"""

import pytest
import numpy as np
import pandas as pd
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners.baseline import LearnerBaseline
from mlpy.resamplings import ResamplingCV, ResamplingHoldout
from mlpy.measures import create_measure
from mlpy.benchmark_advanced import (
    BenchmarkDesign,
    BenchmarkResult,
    BenchmarkScore,
    benchmark_grid,
    benchmark,
    compare_learners
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y_classif = (X[:, 0] + X[:, 1] > 0).astype(int)
    y_regr = X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1
    
    df_classif = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    df_classif['target'] = y_classif
    
    df_regr = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    df_regr['target'] = y_regr
    
    return df_classif, df_regr


@pytest.fixture
def sample_tasks(sample_data):
    """Create sample tasks."""
    df_classif, df_regr = sample_data
    
    task_classif = TaskClassif(
        data=df_classif,
        target='target',
        id='test_classif'
    )
    
    task_regr = TaskRegr(
        data=df_regr,
        target='target',
        id='test_regr'
    )
    
    return task_classif, task_regr


@pytest.fixture
def sample_learners():
    """Create sample learners."""
    return [
        LearnerBaseline(id='baseline1'),
        LearnerBaseline(id='baseline2')
    ]


class TestBenchmarkDesign:
    """Tests for BenchmarkDesign."""
    
    def test_basic_creation(self, sample_tasks, sample_learners):
        """Test basic design creation."""
        task_classif, _ = sample_tasks
        
        design = BenchmarkDesign(
            tasks=[task_classif],
            learners=sample_learners,
            resamplings=[ResamplingCV(folds=3)],
            measures=[create_measure('accuracy')]
        )
        
        assert len(design.tasks) == 1
        assert len(design.learners) == 2
        assert len(design.resamplings) == 1
        assert len(design.measures) == 1
        assert design.n_experiments == 2
    
    def test_single_inputs_converted_to_lists(self, sample_tasks, sample_learners):
        """Test that single inputs are converted to lists."""
        task_classif, _ = sample_tasks
        
        design = BenchmarkDesign(
            tasks=task_classif,  # Single task
            learners=sample_learners[0],  # Single learner
            resamplings=ResamplingCV(folds=3),  # Single resampling
            measures=create_measure('accuracy'),  # Single measure
            paired=True
        )
        
        assert len(design.tasks) == 1
        assert len(design.learners) == 1
        assert len(design.resamplings) == 1
        assert len(design.measures) == 1
    
    def test_empty_inputs_raise_error(self):
        """Test that empty inputs raise errors."""
        with pytest.raises(ValueError, match="At least one task required"):
            BenchmarkDesign(
                tasks=[],
                learners=[LearnerBaseline()],
                resamplings=[ResamplingCV(folds=3)],
                measures=[create_measure('accuracy')]
            )
    
    def test_experiment_grid(self, sample_tasks, sample_learners):
        """Test experiment grid generation."""
        task_classif, task_regr = sample_tasks
        
        design = BenchmarkDesign(
            tasks=[task_classif, task_regr],
            learners=sample_learners,
            resamplings=[ResamplingCV(folds=3), ResamplingHoldout()],
            measures=[create_measure('accuracy')]
        )
        
        grid = design.grid()
        assert len(grid) == 8  # 2 tasks * 2 learners * 2 resamplings


class TestBenchmarkScore:
    """Tests for BenchmarkScore."""
    
    def test_score_creation(self):
        """Test benchmark score creation."""
        score = BenchmarkScore(
            task_id='test_task',
            learner_id='test_learner',
            resampling_id='test_resampling',
            measure_id='accuracy',
            iteration=0,
            score=0.85,
            train_time=1.5,
            predict_time=0.1
        )
        
        assert score.task_id == 'test_task'
        assert score.score == 0.85
        assert score.train_time == 1.5


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""
    
    def test_result_creation(self, sample_tasks, sample_learners):
        """Test benchmark result creation."""
        design = BenchmarkDesign(
            tasks=sample_tasks,
            learners=sample_learners,
            resamplings=[ResamplingCV(folds=2)],
            measures=[create_measure('accuracy')]
        )
        
        result = BenchmarkResult(design)
        
        assert result.design == design
        assert len(result.scores) == 0
        assert len(result.errors) == 0
    
    def test_add_score(self, sample_tasks, sample_learners):
        """Test adding scores."""
        design = BenchmarkDesign(
            tasks=sample_tasks[:1],
            learners=sample_learners[:1],
            resamplings=[ResamplingCV(folds=2)],
            measures=[create_measure('accuracy')]
        )
        
        result = BenchmarkResult(design)
        
        score = BenchmarkScore(
            task_id='test_task',
            learner_id='test_learner',
            resampling_id='test_resampling',
            measure_id='accuracy',
            iteration=0,
            score=0.85
        )
        
        result.add_score(score)
        assert len(result.scores) == 1
    
    def test_add_error(self, sample_tasks, sample_learners):
        """Test adding errors."""
        design = BenchmarkDesign(
            tasks=sample_tasks[:1],
            learners=sample_learners[:1],
            resamplings=[ResamplingCV(folds=2)],
            measures=[create_measure('accuracy')]
        )
        
        result = BenchmarkResult(design)
        
        error = ValueError("Test error")
        result.add_error('task1', 'learner1', 'resampling1', error)
        
        assert len(result.errors) == 1
        assert ('task1', 'learner1', 'resampling1') in result.errors
    
    def test_to_dataframe(self, sample_tasks, sample_learners):
        """Test conversion to dataframe."""
        design = BenchmarkDesign(
            tasks=sample_tasks[:1],
            learners=sample_learners[:1],
            resamplings=[ResamplingCV(folds=2)],
            measures=[create_measure('accuracy')]
        )
        
        result = BenchmarkResult(design)
        
        # Add some scores
        for i in range(3):
            score = BenchmarkScore(
                task_id='test_task',
                learner_id='test_learner',
                resampling_id='test_resampling',
                measure_id='accuracy',
                iteration=i,
                score=0.8 + i * 0.05
            )
            result.add_score(score)
        
        df = result.to_dataframe()
        assert len(df) == 3
        assert 'score' in df.columns
        assert 'task' in df.columns
        
        # Test wide format
        df_wide = result.to_dataframe(wide=True)
        assert 'test_learner' in df_wide.columns
    
    def test_aggregate(self, sample_tasks, sample_learners):
        """Test score aggregation."""
        design = BenchmarkDesign(
            tasks=sample_tasks[:1],
            learners=sample_learners,
            resamplings=[ResamplingCV(folds=2)],
            measures=[create_measure('accuracy')]
        )
        
        result = BenchmarkResult(design)
        
        # Add scores for both learners
        scores = [0.8, 0.85, 0.82]
        for learner in sample_learners:
            for i, score_val in enumerate(scores):
                score = BenchmarkScore(
                    task_id='test_task',
                    learner_id=learner.id,
                    resampling_id='test_resampling',
                    measure_id='accuracy',
                    iteration=i,
                    score=score_val
                )
                result.add_score(score)
        
        # Test aggregation
        agg_df = result.aggregate(
            measure='accuracy',
            group_by=['learner'],
            aggr_func='mean'
        )
        
        assert len(agg_df) == 2
        assert 'learner' in agg_df.columns
        assert 'score' in agg_df.columns
    
    def test_rank_learners(self, sample_tasks, sample_learners):
        """Test learner ranking."""
        design = BenchmarkDesign(
            tasks=sample_tasks[:1],
            learners=sample_learners,
            resamplings=[ResamplingCV(folds=2)],
            measures=[create_measure('accuracy')]
        )
        
        result = BenchmarkResult(design)
        
        # Add scores with different performance
        for i, learner in enumerate(sample_learners):
            score = BenchmarkScore(
                task_id='test_task',
                learner_id=learner.id,
                resampling_id='test_resampling',
                measure_id='accuracy',
                iteration=0,
                score=0.8 + i * 0.1  # Different scores
            )
            result.add_score(score)
        
        rankings = result.rank_learners('accuracy')
        
        assert len(rankings) == 2
        assert 'rank' in rankings.columns
        assert 'final_rank' in rankings.columns
        # Best performer should be ranked first
        assert rankings.iloc[0]['final_rank'] == 1


class TestBenchmarkGrid:
    """Tests for benchmark_grid function."""
    
    def test_grid_creation(self, sample_tasks, sample_learners):
        """Test benchmark grid creation."""
        task_classif, _ = sample_tasks
        
        design = benchmark_grid(
            tasks=task_classif,
            learners=sample_learners,
            resamplings=ResamplingCV(folds=3),
            measures=['accuracy', 'auc']
        )
        
        assert isinstance(design, BenchmarkDesign)
        assert len(design.measures) == 2
        assert design.measures[0].id == 'accuracy'
        assert design.measures[1].id == 'auc'
    
    def test_string_measures_converted(self, sample_tasks, sample_learners):
        """Test that string measures are converted to Measure objects."""
        task_classif, _ = sample_tasks
        
        design = benchmark_grid(
            tasks=task_classif,
            learners=sample_learners,
            resamplings=ResamplingCV(folds=3),
            measures=['accuracy']
        )
        
        assert hasattr(design.measures[0], 'id')
        assert design.measures[0].id == 'accuracy'


class TestBenchmark:
    """Tests for benchmark function."""
    
    def test_basic_benchmark(self, sample_tasks, sample_learners):
        """Test basic benchmark execution."""
        task_classif, _ = sample_tasks
        
        design = benchmark_grid(
            tasks=task_classif,
            learners=sample_learners[:1],  # Use one learner to speed up
            resamplings=ResamplingCV(folds=2),
            measures=['accuracy']
        )
        
        result = benchmark(design, parallel=False, verbose=0)
        
        assert isinstance(result, BenchmarkResult)
        assert len(result.scores) > 0
        assert result.start_time is not None
        assert result.end_time is not None
    
    def test_benchmark_with_errors(self, sample_tasks):
        """Test benchmark with learners that might fail."""
        task_classif, _ = sample_tasks
        
        # Create a learner that will fail
        class FailingLearner(LearnerBaseline):
            def train(self, task):
                raise ValueError("Intentional failure")
        
        failing_learner = FailingLearner(id='failing')
        
        design = benchmark_grid(
            tasks=task_classif,
            learners=[failing_learner],
            resamplings=ResamplingCV(folds=2),
            measures=['accuracy']
        )
        
        result = benchmark(design, parallel=False, verbose=0)
        
        # Should have errors recorded
        assert len(result.errors) > 0
    
    def test_benchmark_stores_metadata(self, sample_tasks, sample_learners):
        """Test that benchmark stores timing information."""
        task_classif, _ = sample_tasks
        
        design = benchmark_grid(
            tasks=task_classif,
            learners=sample_learners[:1],
            resamplings=ResamplingCV(folds=2),
            measures=['accuracy']
        )
        
        result = benchmark(design, parallel=False, verbose=0)
        
        assert result.start_time is not None
        assert result.end_time is not None
        assert result.end_time >= result.start_time


class TestCompareLearners:
    """Tests for compare_learners function."""
    
    def test_basic_comparison(self, sample_tasks, sample_learners):
        """Test basic learner comparison."""
        task_classif, _ = sample_tasks
        
        results = compare_learners(
            task=task_classif,
            learners=sample_learners,
            cv_folds=2,
            measures=['accuracy'],
            test='friedman',
            show_plot=False
        )
        
        assert 'rankings' in results
        assert 'statistical_tests' in results
        assert 'scores' in results
        assert isinstance(results['scores'], pd.DataFrame)
    
    def test_comparison_with_default_measures(self, sample_tasks, sample_learners):
        """Test comparison with default measures."""
        task_classif, task_regr = sample_tasks
        
        # Test classification task
        results_classif = compare_learners(
            task=task_classif,
            learners=sample_learners,
            cv_folds=2,
            measures=None,  # Should use defaults
            show_plot=False
        )
        
        assert 'accuracy' in results_classif['rankings']
        assert 'auc' in results_classif['rankings']
        
        # Test regression task
        results_regr = compare_learners(
            task=task_regr,
            learners=sample_learners,
            cv_folds=2,
            measures=None,  # Should use defaults
            show_plot=False
        )
        
        assert 'rmse' in results_regr['rankings']
        assert 'mae' in results_regr['rankings']


class TestResultMethods:
    """Tests for result analysis methods."""
    
    def test_summary_generation(self, sample_tasks, sample_learners):
        """Test summary text generation."""
        design = BenchmarkDesign(
            tasks=sample_tasks[:1],
            learners=sample_learners,
            resamplings=[ResamplingCV(folds=2)],
            measures=[create_measure('accuracy')]
        )
        
        result = BenchmarkResult(design)
        
        # Add some scores
        for learner in sample_learners:
            score = BenchmarkScore(
                task_id='test_task',
                learner_id=learner.id,
                resampling_id='test_resampling',
                measure_id='accuracy',
                iteration=0,
                score=np.random.rand()
            )
            result.add_score(score)
        
        summary = result.summary()
        
        assert 'BENCHMARK RESULTS SUMMARY' in summary
        assert 'Tasks:' in summary
        assert 'Learners:' in summary
    
    def test_statistical_test_empty_data(self, sample_tasks, sample_learners):
        """Test statistical test with empty data."""
        design = BenchmarkDesign(
            tasks=sample_tasks[:1],
            learners=sample_learners,
            resamplings=[ResamplingCV(folds=2)],
            measures=[create_measure('accuracy')]
        )
        
        result = BenchmarkResult(design)
        
        # No scores added
        test_result = result.statistical_test('accuracy')
        
        assert 'error' in test_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])