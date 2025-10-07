"""Unit tests for benchmark functionality."""

import pytest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlpy import benchmark, BenchmarkResult
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners import LearnerClassifDebug, LearnerRegrDebug, learner_sklearn
from mlpy.resamplings import ResamplingCV, ResamplingHoldout
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifCE, MeasureRegrMSE, MeasureRegrMAE
# from mlpy.data_backends import DataBackendPandas  # Not needed


class TestBenchmarkResult:
    """Test BenchmarkResult class functionality."""
    
    @pytest.fixture
    def iris_task(self):
        """Create Iris classification task."""
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target_names[iris.target]
        return TaskClassif(data=df, target='species', id='iris')
        
    @pytest.fixture
    def wine_task(self):
        """Create Wine classification task."""
        from sklearn.datasets import load_wine
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['class'] = wine.target
        return TaskClassif(data=df, target='class', id='wine')
        
    @pytest.fixture
    def regression_task(self):
        """Create synthetic regression task."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        y = X[:, 0] * 2 + X[:, 1] * -1 + np.random.randn(n) * 0.5
        
        df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
        df['y'] = y
        return TaskRegr(data=df, target='y', id='synth_regr')
        
    def test_benchmark_single_task_single_learner(self, iris_task):
        """Test benchmark with single task and learner."""
        learner = LearnerClassifDebug()
        measure = MeasureClassifAccuracy()
        resampling = ResamplingHoldout(ratio=0.8)
        
        result = benchmark(
            tasks=iris_task,
            learners=learner,
            resampling=resampling,
            measures=measure
        )
        
        assert isinstance(result, BenchmarkResult)
        assert len(result.tasks) == 1
        assert len(result.learners) == 1
        assert result.n_experiments == 1
        assert result.n_successful == 1
        assert result.n_errors == 0
        
    def test_benchmark_multiple_learners(self, iris_task):
        """Test benchmark with multiple learners."""
        learners = [
            LearnerClassifDebug(id='debug1'),
            learner_sklearn(DecisionTreeClassifier(max_depth=3), id='dt'),
            learner_sklearn(LogisticRegression(max_iter=100), id='lr')
        ]
        measures = [MeasureClassifAccuracy(), MeasureClassifCE()]
        resampling = ResamplingCV(folds=3)
        
        result = benchmark(
            tasks=iris_task,
            learners=learners,
            resampling=resampling,
            measures=measures
        )
        
        assert len(result.learners) == 3
        assert result.n_experiments == 3
        assert result.n_successful == 3
        
        # Check all combinations have results
        for learner in learners:
            res = result.get_result(iris_task.id, learner.id)
            assert res is not None
            assert len(res.iterations) == 3  # 3-fold CV
            
    def test_benchmark_multiple_tasks(self, iris_task, wine_task):
        """Test benchmark with multiple tasks."""
        learners = [
            learner_sklearn(DecisionTreeClassifier(max_depth=3), id='dt'),
            learner_sklearn(RandomForestClassifier(n_estimators=10), id='rf')
        ]
        measure = MeasureClassifAccuracy()
        resampling = ResamplingHoldout(ratio=0.7)
        
        result = benchmark(
            tasks=[iris_task, wine_task],
            learners=learners,
            resampling=resampling,
            measures=measure
        )
        
        assert len(result.tasks) == 2
        assert result.n_experiments == 4  # 2 tasks × 2 learners
        assert result.n_successful == 4
        
    def test_score_table(self, iris_task, wine_task):
        """Test score table generation."""
        learners = [
            learner_sklearn(DecisionTreeClassifier(max_depth=3), id='dt'),
            learner_sklearn(RandomForestClassifier(n_estimators=10), id='rf')
        ]
        measure = MeasureClassifAccuracy()
        resampling = ResamplingCV(folds=3)
        
        result = benchmark(
            tasks=[iris_task, wine_task],
            learners=learners,
            resampling=resampling,
            measures=measure
        )
        
        # Get score table
        scores = result.score_table()
        
        assert isinstance(scores, pd.DataFrame)
        assert scores.shape == (2, 2)  # 2 tasks × 2 learners
        assert list(scores.index) == ['iris', 'wine']
        assert list(scores.columns) == ['dt', 'rf']
        
        # All scores should be between 0 and 1 for accuracy
        assert (scores >= 0).all().all()
        assert (scores <= 1).all().all()
        
    def test_aggregate_methods(self, iris_task):
        """Test different aggregation methods."""
        learner = learner_sklearn(DecisionTreeClassifier(max_depth=3))
        measure = MeasureClassifAccuracy()
        resampling = ResamplingCV(folds=5)
        
        result = benchmark(
            tasks=iris_task,
            learners=learner,
            resampling=resampling,
            measures=measure
        )
        
        # Test different aggregations
        mean_scores = result.aggregate(measure.id, "mean")
        std_scores = result.aggregate(measure.id, "std")
        min_scores = result.aggregate(measure.id, "min")
        max_scores = result.aggregate(measure.id, "max")
        median_scores = result.aggregate(measure.id, "median")
        
        # All should return DataFrames of same shape
        for df in [mean_scores, std_scores, min_scores, max_scores, median_scores]:
            assert isinstance(df, pd.DataFrame)
            assert df.shape == (1, 1)
            
        # Mean should be between min and max
        assert min_scores.iloc[0, 0] <= mean_scores.iloc[0, 0] <= max_scores.iloc[0, 0]
        
    def test_rank_learners(self, iris_task):
        """Test learner ranking."""
        learners = [
            LearnerClassifDebug(id='debug', predict_response='sample'),  # Random
            learner_sklearn(DecisionTreeClassifier(max_depth=2), id='dt_shallow'),
            learner_sklearn(DecisionTreeClassifier(max_depth=10), id='dt_deep'),
            learner_sklearn(RandomForestClassifier(n_estimators=50), id='rf')
        ]
        measure = MeasureClassifAccuracy()
        resampling = ResamplingCV(folds=3)
        
        result = benchmark(
            tasks=iris_task,
            learners=learners,
            resampling=resampling,
            measures=measure
        )
        
        # Get rankings
        rankings = result.rank_learners()
        
        assert isinstance(rankings, pd.DataFrame)
        assert len(rankings) == 4
        assert list(rankings.columns) == ['learner', 'mean_score', 'rank']
        assert list(rankings['rank']) == [1, 2, 3, 4]
        
        # RF should typically outperform shallow DT
        rf_rank = rankings[rankings['learner'] == 'rf']['rank'].iloc[0]
        dt_shallow_rank = rankings[rankings['learner'] == 'dt_shallow']['rank'].iloc[0]
        assert rf_rank < dt_shallow_rank
        
    def test_rank_learners_minimize(self, regression_task):
        """Test learner ranking with minimization measure."""
        learners = [
            LearnerRegrDebug(id='debug'),
            learner_sklearn(DecisionTreeRegressor(max_depth=3), id='dt'),
        ]
        measure = MeasureRegrMSE()  # Should be minimized
        resampling = ResamplingHoldout()
        
        result = benchmark(
            tasks=regression_task,
            learners=learners,
            resampling=resampling,
            measures=measure
        )
        
        rankings = result.rank_learners()
        
        # Lower MSE should get better (lower) rank
        assert rankings.iloc[0]['mean_score'] < rankings.iloc[1]['mean_score']
        
    def test_to_long_format(self, iris_task):
        """Test conversion to long format."""
        learners = [
            learner_sklearn(DecisionTreeClassifier(max_depth=3), id='dt'),
            learner_sklearn(RandomForestClassifier(n_estimators=10), id='rf')
        ]
        measures = [MeasureClassifAccuracy(), MeasureClassifCE()]
        resampling = ResamplingCV(folds=3)
        
        result = benchmark(
            tasks=iris_task,
            learners=learners,
            resampling=resampling,
            measures=measures
        )
        
        # Convert to long format
        long_df = result.to_long_format()
        
        assert isinstance(long_df, pd.DataFrame)
        # Should have 6 rows: 2 learners × 3 folds
        assert len(long_df) == 6
        
        # Check columns
        expected_cols = [
            'task_id', 'learner_id', 'iteration', 
            'train_time', 'predict_time',
            'score_classif.acc', 'score_classif.ce'
        ]
        assert all(col in long_df.columns for col in expected_cols)
        
    def test_error_handling(self, iris_task, regression_task):
        """Test handling of errors during benchmark."""
        # Mix classification and regression learners/tasks with incompatible measures
        learners = [
            learner_sklearn(DecisionTreeClassifier(), id='clf'),
            learner_sklearn(DecisionTreeRegressor(), id='reg')
        ]
        tasks = [iris_task, regression_task]
        measure_clf = MeasureClassifAccuracy()
        measure_reg = MeasureRegrMSE()
        resampling = ResamplingHoldout()
        
        # This should fail for all combinations because measures are incompatible
        # with some tasks (classification measure on regression task and vice versa)
        result = benchmark(
            tasks=tasks,
            learners=learners,
            resampling=resampling,
            measures=[measure_clf, measure_reg]
        )
        
        assert result.n_experiments == 4
        assert result.n_successful == 0  # All fail due to incompatible measures
        assert result.n_errors == 4
        
        # All combinations should have errors
        assert result.get_error('iris', 'clf') is not None
        assert result.get_error('iris', 'reg') is not None
        assert result.get_error('synth_regr', 'clf') is not None
        assert result.get_error('synth_regr', 'reg') is not None
        
    def test_encapsulation(self, iris_task):
        """Test learner encapsulation during benchmark."""
        base_learner = learner_sklearn(DecisionTreeClassifier())
        measure = MeasureClassifAccuracy()
        resampling = ResamplingCV(folds=3)
        
        # Benchmark with encapsulation (default)
        result1 = benchmark(
            tasks=iris_task,
            learners=base_learner,
            resampling=resampling,
            measures=measure,
            encapsulate=True
        )
        
        # Original learner should not be trained
        assert not base_learner.is_trained
        
        # Benchmark without encapsulation
        result2 = benchmark(
            tasks=iris_task,
            learners=base_learner,
            resampling=resampling,
            measures=measure,
            encapsulate=False
        )
        
        # Now learner should be trained
        assert base_learner.is_trained
        
    def test_multiple_measures(self, iris_task):
        """Test benchmark with multiple measures."""
        learner = learner_sklearn(RandomForestClassifier(n_estimators=10))
        measures = [
            MeasureClassifAccuracy(),
            MeasureClassifCE(),
        ]
        resampling = ResamplingCV(folds=3)
        
        result = benchmark(
            tasks=iris_task,
            learners=learner,
            resampling=resampling,
            measures=measures
        )
        
        # Check both measures are available
        acc_table = result.score_table('classif.acc')
        ce_table = result.score_table('classif.ce')
        
        assert acc_table.shape == (1, 1)
        assert ce_table.shape == (1, 1)
        
        # Accuracy + CE should approximately equal 1
        assert abs(acc_table.iloc[0, 0] + ce_table.iloc[0, 0] - 1.0) < 0.01
        
    def test_repr(self, iris_task, wine_task):
        """Test string representation."""
        learners = [
            learner_sklearn(DecisionTreeClassifier(), id='dt'),
            learner_sklearn(RandomForestClassifier(), id='rf')
        ]
        result = benchmark(
            tasks=[iris_task, wine_task],
            learners=learners,
            resampling=ResamplingHoldout(),
            measures=MeasureClassifAccuracy()
        )
        
        repr_str = repr(result)
        assert "2 tasks × 2 learners" in repr_str
        assert "4 successful" in repr_str


class TestBenchmarkFunction:
    """Test benchmark function."""
    
    @pytest.fixture
    def classification_tasks(self):
        """Create multiple classification tasks."""
        tasks = []
        
        # Iris
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        tasks.append(TaskClassif(data=df, target='species', id='iris'))
        
        # Wine  
        from sklearn.datasets import load_wine
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['class'] = wine.target
        tasks.append(TaskClassif(data=df, target='class', id='wine'))
        
        return tasks
        
    def test_input_validation(self):
        """Test input validation."""
        # Empty tasks
        with pytest.raises(ValueError, match="At least one task"):
            benchmark(tasks=[], learners=LearnerClassifDebug(), 
                     resampling=ResamplingHoldout(), measures=MeasureClassifAccuracy())
            
        # Empty learners
        task = TaskClassif(
            data=pd.DataFrame({'x': [1, 2], 'y': ['a', 'b']}),
            target='y'
        )
        with pytest.raises(ValueError, match="At least one learner"):
            benchmark(tasks=task, learners=[], 
                     resampling=ResamplingHoldout(), measures=MeasureClassifAccuracy())
            
        # Empty measures
        with pytest.raises(ValueError, match="At least one measure"):
            benchmark(tasks=task, learners=LearnerClassifDebug(),
                     resampling=ResamplingHoldout(), measures=[])
            
    def test_store_options(self, classification_tasks):
        """Test store_predictions and store_models options."""
        learner = learner_sklearn(DecisionTreeClassifier())
        measure = MeasureClassifAccuracy()
        resampling = ResamplingCV(folds=2)
        
        # With models stored
        result_with = benchmark(
            tasks=classification_tasks[0],
            learners=learner,
            resampling=resampling,
            measures=measure,
            store_models=True,
            store_backends=False
        )
        
        res = result_with.get_result('iris', learner.id)
        assert res.predictions[0] is not None  # Predictions always stored
        
        # Without models stored (default)
        result_without = benchmark(
            tasks=classification_tasks[0],
            learners=learner,
            resampling=resampling,
            measures=measure,
            store_models=False,
            store_backends=False
        )
        
        res = result_without.get_result('iris', learner.id)
        assert res.predictions[0] is not None  # Predictions still stored
        
    def test_real_world_benchmark(self, classification_tasks):
        """Test a realistic benchmark scenario."""
        # Multiple sklearn learners
        learners = [
            learner_sklearn(
                DecisionTreeClassifier(max_depth=3, random_state=42),
                id='dt_shallow'
            ),
            learner_sklearn(
                DecisionTreeClassifier(max_depth=None, random_state=42),
                id='dt_deep'
            ),
            learner_sklearn(
                RandomForestClassifier(n_estimators=50, random_state=42),
                id='rf'
            ),
            learner_sklearn(
                LogisticRegression(max_iter=200, random_state=42),
                id='logreg'
            )
        ]
        
        # Multiple measures
        measures = [
            MeasureClassifAccuracy(),
            MeasureClassifCE()
        ]
        
        # Run benchmark
        result = benchmark(
            tasks=classification_tasks,
            learners=learners,
            resampling=ResamplingCV(folds=5),
            measures=measures
        )
        
        # All experiments should succeed
        assert result.n_successful == 8  # 2 tasks × 4 learners
        assert result.n_errors == 0
        
        # Check score table
        acc_scores = result.score_table('classif.acc')
        assert acc_scores.shape == (2, 4)
        
        # RF should generally perform well
        rf_scores = acc_scores['rf']
        assert all(rf_scores > 0.8)  # Should get good accuracy
        
        # Rankings should make sense
        rankings = result.rank_learners('classif.acc')
        top_learner = rankings.iloc[0]['learner']
        assert top_learner in ['rf', 'logreg']  # These should perform best