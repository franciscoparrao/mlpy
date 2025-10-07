"""Unit tests for AutoML functionality."""

import pytest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlpy.automl import (
    ParamInt, ParamFloat, ParamCategorical, ParamSet,
    TunerGrid, TunerRandom, TuneResult,
    AutoFeaturesNumeric, AutoFeaturesCategorical, AutoFeaturesInteraction
)
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners import learner_sklearn
from mlpy.resamplings import ResamplingCV, ResamplingHoldout
from mlpy.measures import MeasureClassifAccuracy, MeasureRegrMSE
from mlpy.pipelines import linear_pipeline, GraphLearner, PipeOpLearner


class TestParameterDefinitions:
    """Test parameter space definitions."""
    
    def test_param_int(self):
        """Test integer parameter."""
        param = ParamInt("n_estimators", lower=10, upper=100)
        
        # Test sampling
        values = param.sample(10, seed=42)
        assert len(values) == 10
        assert all(isinstance(v, int) for v in values)
        assert all(10 <= v <= 100 for v in values)
        
        # Test grid
        grid = param.grid(resolution=5)
        assert len(grid) >= 5
        assert grid[0] == 10
        assert grid[-1] <= 100
        
    def test_param_int_log_scale(self):
        """Test integer parameter with log scale."""
        param = ParamInt("n_estimators", lower=10, upper=1000, log_scale=True)
        
        values = param.sample(10, seed=42)
        assert len(values) == 10
        assert all(10 <= v <= 1000 for v in values)
        
        # Grid should be more spread out on log scale
        grid = param.grid(resolution=5)
        assert len(grid) >= 3
        # Check that spacing increases
        diffs = np.diff(grid)
        assert diffs[-1] > diffs[0]
        
    def test_param_float(self):
        """Test float parameter."""
        param = ParamFloat("learning_rate", lower=0.001, upper=1.0)
        
        values = param.sample(10, seed=42)
        assert len(values) == 10
        assert all(isinstance(v, float) for v in values)
        assert all(0.001 <= v <= 1.0 for v in values)
        
        grid = param.grid(resolution=5)
        assert len(grid) == 5
        assert abs(grid[0] - 0.001) < 1e-6
        assert abs(grid[-1] - 1.0) < 1e-6
        
    def test_param_categorical(self):
        """Test categorical parameter."""
        param = ParamCategorical("criterion", values=["gini", "entropy"])
        
        values = param.sample(10, seed=42)
        assert len(values) == 10
        assert all(v in ["gini", "entropy"] for v in values)
        
        grid = param.grid()
        assert grid == ["gini", "entropy"]
        
    def test_param_set(self):
        """Test parameter set."""
        params = ParamSet([
            ParamInt("n_estimators", 10, 100),
            ParamFloat("max_features", 0.1, 1.0),
            ParamCategorical("criterion", ["gini", "entropy"])
        ])
        
        # Test sampling
        configs = params.sample(5, seed=42)
        assert len(configs) == 5
        for config in configs:
            assert "n_estimators" in config
            assert "max_features" in config
            assert "criterion" in config
            assert 10 <= config["n_estimators"] <= 100
            assert 0.1 <= config["max_features"] <= 1.0
            assert config["criterion"] in ["gini", "entropy"]
            
        # Test grid
        grid = params.grid(resolution=3)
        assert len(grid) == 3 * 3 * 2  # 3 ints × 3 floats × 2 categoricals


class TestTuning:
    """Test hyperparameter tuning."""
    
    @pytest.fixture
    def iris_task(self):
        """Create Iris classification task."""
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        return TaskClassif(data=df, target='species')
        
    @pytest.fixture
    def simple_task(self):
        """Create simple classification task."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        df = pd.DataFrame(X, columns=[f'x{i}' for i in range(5)])
        df['y'] = y
        return TaskClassif(data=df, target='y')
        
    def test_grid_tuner(self, simple_task):
        """Test grid search tuning."""
        learner = learner_sklearn(DecisionTreeClassifier())
        
        param_set = ParamSet([
            ParamInt("max_depth", 1, 5),
            ParamCategorical("criterion", ["gini", "entropy"])
        ])
        
        tuner = TunerGrid(resolution=3)
        result = tuner.tune(
            learner=learner,
            task=simple_task,
            resampling=ResamplingHoldout(ratio=0.7),
            measure=MeasureClassifAccuracy(),
            param_set=param_set
        )
        
        assert isinstance(result, TuneResult)
        assert len(result.configs) == 3 * 2  # 3 depths × 2 criteria
        assert len(result.scores) == len(result.configs)
        assert result.best_config in result.configs
        assert result.best_score == max(result.scores)  # Accuracy is maximized
        
    def test_random_tuner(self, simple_task):
        """Test random search tuning."""
        learner = learner_sklearn(RandomForestClassifier(n_estimators=10))
        
        param_set = ParamSet([
            ParamInt("max_depth", 1, 10),
            ParamFloat("max_features", 0.1, 1.0),
            ParamCategorical("criterion", ["gini", "entropy"])
        ])
        
        tuner = TunerRandom(n_evals=10, seed=42)
        result = tuner.tune(
            learner=learner,
            task=simple_task,
            resampling=ResamplingCV(folds=3),
            measure=MeasureClassifAccuracy(),
            param_set=param_set
        )
        
        assert isinstance(result, TuneResult)
        assert len(result.configs) == 10
        assert len(result.scores) == 10
        assert result.best_config in result.configs
        assert all(0 <= s <= 1 for s in result.scores)  # Valid accuracy scores
        
    def test_tune_result_dataframe(self, simple_task):
        """Test TuneResult conversion to DataFrame."""
        learner = learner_sklearn(DecisionTreeClassifier())
        param_set = ParamSet([ParamInt("max_depth", 1, 3)])
        
        tuner = TunerGrid(resolution=3)
        result = tuner.tune(
            learner=learner,
            task=simple_task,
            resampling=ResamplingHoldout(),
            measure=MeasureClassifAccuracy(),
            param_set=param_set
        )
        
        df = result.as_data_frame()
        assert len(df) == 3
        assert "max_depth" in df.columns
        assert "classif.acc_score" in df.columns
        assert "is_best" in df.columns
        assert df["is_best"].sum() == 1  # Only one best


class TestAutoFeatures:
    """Test automatic feature engineering."""
    
    @pytest.fixture
    def numeric_task(self):
        """Create task with numeric features."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.exponential(2, 100),  # Positive for log
            'x2': np.random.randn(100) ** 2,      # Non-negative for sqrt
            'x3': np.random.randn(100),           # Can be negative
            'x4': np.random.uniform(1, 10, 100),  # No zeros for reciprocal
            'y': np.random.choice([0, 1], 100)
        })
        return TaskClassif(data=df, target='y')
        
    @pytest.fixture
    def categorical_task(self):
        """Create task with categorical features."""
        np.random.seed(42)
        df = pd.DataFrame({
            'color': np.random.choice(['red', 'blue', 'green'], 100, p=[0.5, 0.3, 0.2]),
            'size': np.random.choice(['S', 'M', 'L', 'XL'], 100),
            'brand': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100),
            'y': np.random.choice([0, 1], 100)
        })
        return TaskClassif(data=df, target='y')
        
    @pytest.fixture
    def mixed_task(self):
        """Create task with mixed features."""
        np.random.seed(42)
        df = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.exponential(1, 100),
            'cat1': np.random.choice(['A', 'B', 'C'], 100),
            'cat2': np.random.choice(['X', 'Y'], 100),
            'y': np.random.randn(100)  # Regression target
        })
        return TaskRegr(data=df, target='y')
        
    def test_auto_features_numeric(self, numeric_task):
        """Test numeric feature engineering."""
        op = AutoFeaturesNumeric(
            transforms=["log", "sqrt", "square", "reciprocal", "bins"]
        )
        
        # Train
        result = op.train({"input": numeric_task})
        transformed_task = result["output"]
        
        assert op.is_trained
        
        # Check new features were created
        original_features = set(numeric_task.feature_names)
        new_features = set(transformed_task.feature_names)
        assert len(new_features) > len(original_features)
        
        # Check specific transformations
        data = transformed_task.data()
        assert 'x1_log' in data.columns  # x1 is positive
        assert 'x2_sqrt' in data.columns  # x2 is non-negative
        assert 'x3_sq' in data.columns   # All can be squared
        assert 'x4_inv' in data.columns  # x4 has no zeros
        assert 'x1_bin' in data.columns  # All can be binned
        
        # Test predict
        result = op.predict({"input": numeric_task})
        assert isinstance(result["output"], TaskClassif)
        
    def test_auto_features_categorical(self, categorical_task):
        """Test categorical feature engineering."""
        op = AutoFeaturesCategorical(
            methods=["count", "frequency", "rare"],
            min_frequency=0.15
        )
        
        # Train
        result = op.train({"input": categorical_task})
        transformed_task = result["output"]
        
        assert op.is_trained
        
        # Check new features
        data = transformed_task.data()
        assert 'color_count' in data.columns
        assert 'color_freq' in data.columns
        assert 'size_count' in data.columns
        
        # Check count encoding
        color_counts = categorical_task.data()['color'].value_counts()
        for color, count in color_counts.items():
            mask = categorical_task.data()['color'] == color
            assert all(data.loc[mask, 'color_count'] == count)
            
        # Check rare grouping (green should be rare with p=0.2)
        if 'color_grouped' in data.columns:
            assert '__rare__' in data['color_grouped'].values
            
    def test_auto_features_interaction(self, mixed_task):
        """Test interaction features."""
        op = AutoFeaturesInteraction(
            max_interactions=5,
            numeric_ops=["multiply", "divide"]
        )
        
        # Train
        result = op.train({"input": mixed_task})
        transformed_task = result["output"]
        
        assert op.is_trained
        
        # Check interactions were created
        data = transformed_task.data()
        new_cols = [col for col in data.columns if col not in mixed_task.data().columns]
        assert len(new_cols) >= 3  # At least some interactions
        
        # Check specific interaction types
        interaction_types = {
            '_x_': 'multiply',
            '_div_': 'divide',
            '_AND_': 'concat',
            '_by_': 'group_mean'
        }
        
        found_types = set()
        for col in new_cols:
            for pattern, itype in interaction_types.items():
                if pattern in col:
                    found_types.add(itype)
                    
        assert len(found_types) >= 2  # At least 2 types of interactions
        
    def test_auto_features_in_pipeline(self, mixed_task):
        """Test auto features in a pipeline."""
        from mlpy.pipelines import PipeOpScale, PipeOpImpute, PipeOpEncode
        
        # Create pipeline with auto feature engineering
        # Use appropriate learner for task type
        if isinstance(mixed_task, TaskRegr):
            from sklearn.linear_model import LinearRegression
            learner = learner_sklearn(LinearRegression())
        else:
            learner = learner_sklearn(LogisticRegression())
            
        pipeline = linear_pipeline(
            AutoFeaturesNumeric(transforms=["log", "square"]),
            AutoFeaturesCategorical(methods=["count"]),
            AutoFeaturesInteraction(max_interactions=3),
            PipeOpEncode(method="onehot"),  # Encode categorical features
            PipeOpScale(),
            PipeOpLearner(learner)
        )
        
        graph_learner = GraphLearner(pipeline)
        
        # Should be able to train
        graph_learner.train(mixed_task)
        assert graph_learner.is_trained
        
        # And predict
        predictions = graph_learner.predict(mixed_task)
        assert len(predictions.response) == mixed_task.nrow
        
    def test_auto_features_with_missing(self):
        """Test auto features with missing values."""
        df = pd.DataFrame({
            'x1': [1, 2, np.nan, 4, 5],
            'x2': [0, 1, 2, np.nan, 4],  # Has zero and missing
            'cat': ['A', None, 'B', 'A', 'B'],
            'y': [0, 1, 0, 1, 0]
        })
        task = TaskClassif(data=df, target='y')
        
        # Numeric features should handle missing
        op_num = AutoFeaturesNumeric(transforms=["square", "reciprocal"])
        result = op_num.train({"input": task})
        data = result["output"].data()
        
        # Square should work with NaN
        assert 'x1_sq' in data.columns
        assert pd.isna(data.loc[2, 'x1_sq'])
        
        # Reciprocal should handle zeros
        if 'x2_inv' in data.columns:
            assert pd.isna(data.loc[0, 'x2_inv'])  # 1/0 -> NaN


class TestAutoMLIntegration:
    """Integration tests for AutoML components."""
    
    @pytest.fixture
    def complex_task(self):
        """Create a complex task for testing."""
        np.random.seed(42)
        n = 300
        
        # Generate features
        df = pd.DataFrame({
            'num1': np.random.randn(n),
            'num2': np.random.exponential(2, n),
            'num3': np.random.uniform(0, 10, n),
            'cat1': np.random.choice(['A', 'B', 'C', 'D'], n),
            'cat2': np.random.choice(['X', 'Y', 'Z'], n, p=[0.5, 0.3, 0.2]),
        })
        
        # Add some missing values
        df.loc[10:20, 'num1'] = np.nan
        df.loc[50:55, 'cat1'] = None
        
        # Create target with some signal
        y = (df['num1'].fillna(0) > 0).astype(int)
        y += (df['cat1'] == 'A').astype(int)
        y = (y > 0).astype(int)
        df['target'] = y
        
        return TaskClassif(data=df, target='target')
        
    def test_full_automl_pipeline(self, complex_task):
        """Test complete AutoML workflow."""
        from mlpy.pipelines import PipeOpImpute, PipeOpEncode, PipeOpScale, PipeOpSelect
        
        # Create simpler pipeline without interaction features
        # (they can create complex dependencies between train/test)
        base_pipeline = linear_pipeline(
            PipeOpImpute(strategy="mean"),
            AutoFeaturesNumeric(transforms=["square"]),  # Simple transform only
            PipeOpEncode(method="onehot"),
            PipeOpScale(),
            PipeOpLearner(learner_sklearn(RandomForestClassifier()))
        )
        
        # Tune the pipeline
        pipeline_learner = GraphLearner(base_pipeline)
        
        # For GraphLearner with pipeline, parameters can be accessed via learner.param
        param_set = ParamSet([
            ParamInt("learner.n_estimators", 10, 100),
            ParamInt("learner.max_depth", 2, 10),
            ParamFloat("learner.max_features", 0.3, 1.0)
        ])
        
        tuner = TunerRandom(n_evals=5, seed=42)
        result = tuner.tune(
            learner=pipeline_learner,
            task=complex_task,
            resampling=ResamplingCV(folds=3),
            measure=MeasureClassifAccuracy(),
            param_set=param_set
        )
        
        assert result.best_score > 0.5  # Should do better than random
        assert "learner.n_estimators" in result.best_config
        
        # Verify that tuning worked and we got results
        print(f"Best config: {result.best_config}")
        print(f"Best score: {result.best_score}")
        
        # Best score should be reasonable (at least better than random)
        assert result.best_score > 0.6  # Should do better than random for this task


if __name__ == "__main__":
    pytest.main([__file__])