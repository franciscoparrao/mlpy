"""Unit tests for pipeline functionality."""

import pytest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from mlpy.pipelines import (
    PipeOp, PipeOpInput, PipeOpOutput, PipeOpState,
    PipeOpLearner, PipeOpNOP,
    PipeOpScale, PipeOpImpute, PipeOpSelect, PipeOpEncode,
    Graph, GraphLearner, linear_pipeline
)
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners import learner_sklearn, LearnerClassifDebug
from mlpy.predictions import PredictionClassif


class TestPipeOpBase:
    """Test base PipeOp functionality."""
    
    def test_pipeop_state(self):
        """Test PipeOpState class."""
        state = PipeOpState()
        
        assert not state.is_trained
        assert state["foo"] is None
        
        state["foo"] = "bar"
        state.is_trained = True
        
        assert state["foo"] == "bar"
        assert state.is_trained
        
    def test_pipeop_nop(self):
        """Test no-operation PipeOp."""
        op = PipeOpNOP()
        
        assert op.id == "nop"
        assert len(op.input) == 1
        assert len(op.output) == 1
        assert not op.is_trained
        
        # Train with arbitrary input
        result = op.train({"input": "hello"})
        assert result == {"output": "hello"}
        assert op.is_trained
        
        # Predict
        result = op.predict({"input": "world"})
        assert result == {"output": "world"}
        
    def test_pipeop_validation(self):
        """Test input validation."""
        op = PipeOpNOP()
        
        # Missing input
        with pytest.raises(ValueError, match="Missing required input"):
            op.train({})
            
        # Extra input
        with pytest.raises(ValueError, match="Unexpected inputs"):
            op.train({"input": 1, "extra": 2})
            
    def test_pipeop_reset(self):
        """Test resetting PipeOp."""
        op = PipeOpNOP()
        op.train({"input": 1})
        
        assert op.is_trained
        
        op.reset()
        assert not op.is_trained
        
    def test_pipeop_clone(self):
        """Test cloning PipeOp."""
        op = PipeOpNOP(id="test")
        op.train({"input": 1})
        
        cloned = op.clone()
        
        assert cloned.id == "test"
        assert not cloned.is_trained  # Clone is reset


class TestPipeOpLearner:
    """Test PipeOpLearner functionality."""
    
    @pytest.fixture
    def iris_task(self):
        """Create Iris classification task."""
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        return TaskClassif(data=df, target='species', id='iris')
        
    def test_pipeop_learner_basic(self, iris_task):
        """Test basic PipeOpLearner functionality."""
        learner = learner_sklearn(DecisionTreeClassifier(max_depth=2))
        op = PipeOpLearner(learner)
        
        assert op.id == f"learner.{learner.id}"
        assert not op.is_trained
        
        # Train
        result = op.train({"input": iris_task})
        assert "output" in result
        assert isinstance(result["output"], PredictionClassif)
        assert op.is_trained
        
        # Predict
        result = op.predict({"input": iris_task})
        assert isinstance(result["output"], PredictionClassif)
        
    def test_pipeop_learner_untrained_error(self, iris_task):
        """Test error when predicting without training."""
        learner = LearnerClassifDebug()
        op = PipeOpLearner(learner)
        
        with pytest.raises(RuntimeError, match="must be trained"):
            op.predict({"input": iris_task})


class TestPipeOpOperators:
    """Test built-in pipeline operators."""
    
    @pytest.fixture
    def numeric_task(self):
        """Create task with numeric features."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100) * 10 + 5,
            'x3': np.random.uniform(0, 1, 100),
            'y': np.random.choice(['A', 'B'], 100)
        })
        return TaskClassif(data=df, target='y')
        
    @pytest.fixture
    def missing_task(self):
        """Create task with missing values."""
        df = pd.DataFrame({
            'x1': [1, 2, np.nan, 4, 5],
            'x2': ['a', 'b', None, 'd', 'e'],
            'x3': [10, np.nan, 30, np.nan, 50],
            'y': [0, 1, 0, 1, 0]
        })
        return TaskClassif(data=df, target='y')
        
    @pytest.fixture
    def categorical_task(self):
        """Create task with categorical features."""
        df = pd.DataFrame({
            'color': ['red', 'blue', 'green', 'red', 'blue'],
            'size': ['S', 'M', 'L', 'M', 'S'],
            'numeric': [1, 2, 3, 4, 5],
            'y': [0, 1, 0, 1, 0]
        })
        return TaskClassif(data=df, target='y')
        
    def test_pipeop_scale(self, numeric_task):
        """Test scaling operator."""
        op = PipeOpScale(method="standard")
        
        # Train
        result = op.train({"input": numeric_task})
        scaled_task = result["output"]
        
        assert op.is_trained
        assert isinstance(scaled_task, TaskClassif)
        
        # Check that data was scaled
        scaled_data = scaled_task.data()
        for col in ['x1', 'x2', 'x3']:
            assert abs(scaled_data[col].mean()) < 1e-10  # Near zero
            assert abs(scaled_data[col].std() - 1.0) < 0.1  # Near 1
            
        # Predict
        result = op.predict({"input": numeric_task})
        assert isinstance(result["output"], TaskClassif)
        
    def test_pipeop_scale_methods(self, numeric_task):
        """Test different scaling methods."""
        for method in ["standard", "minmax", "robust"]:
            op = PipeOpScale(method=method)
            result = op.train({"input": numeric_task})
            assert op.is_trained
            
    def test_pipeop_impute(self, missing_task):
        """Test imputation operator."""
        op = PipeOpImpute(strategy="mean")
        
        # Train
        result = op.train({"input": missing_task})
        imputed_task = result["output"]
        
        assert op.is_trained
        
        # Check no missing values
        imputed_data = imputed_task.data()
        assert not imputed_data.isnull().any().any()
        
        # Check numeric column was imputed with mean
        assert imputed_data.loc[2, 'x1'] == 3.0  # mean of [1,2,4,5]
        
    def test_pipeop_impute_with_indicator(self, missing_task):
        """Test imputation with missing indicators."""
        op = PipeOpImpute(strategy="constant", fill_value=0, add_indicator=True)
        
        result = op.train({"input": missing_task})
        imputed_task = result["output"]
        imputed_data = imputed_task.data()
        
        # Check indicator columns were added
        assert 'x1_was_missing' in imputed_data.columns
        assert 'x3_was_missing' in imputed_data.columns
        
        # Check indicators
        assert imputed_data.loc[2, 'x1_was_missing'] == 1
        assert imputed_data.loc[0, 'x1_was_missing'] == 0
        
    def test_pipeop_select(self, numeric_task):
        """Test feature selection operator."""
        op = PipeOpSelect(k=2)
        
        # Train
        result = op.train({"input": numeric_task})
        selected_task = result["output"]
        
        assert op.is_trained
        assert len(selected_task.feature_names) == 2
        
        # Original had 3 features, now should have 2
        assert len(numeric_task.feature_names) == 3
        assert len(selected_task.feature_names) == 2
        
    def test_pipeop_encode_onehot(self, categorical_task):
        """Test one-hot encoding operator."""
        op = PipeOpEncode(method="onehot")
        
        # Train
        result = op.train({"input": categorical_task})
        encoded_task = result["output"]
        
        assert op.is_trained
        
        # Check that categorical columns were encoded
        encoded_data = encoded_task.data()
        assert 'color_blue' in encoded_data.columns
        assert 'color_green' in encoded_data.columns
        assert 'color_red' in encoded_data.columns
        assert 'size_L' in encoded_data.columns
        assert 'size_M' in encoded_data.columns
        assert 'size_S' in encoded_data.columns
        
        # Numeric column should remain
        assert 'numeric' in encoded_data.columns
        
    def test_pipeop_encode_label(self, categorical_task):
        """Test label encoding operator."""
        op = PipeOpEncode(method="label")
        
        # Train
        result = op.train({"input": categorical_task})
        encoded_task = result["output"]
        encoded_data = encoded_task.data()
        
        # Check that categorical columns are now numeric
        assert pd.api.types.is_numeric_dtype(encoded_data['color'])
        assert pd.api.types.is_numeric_dtype(encoded_data['size'])
        
        # Check values are integers
        assert set(encoded_data['color'].unique()).issubset({0, 1, 2})


class TestGraph:
    """Test Graph functionality."""
    
    def test_graph_construction(self):
        """Test building a graph."""
        graph = Graph()
        
        # Add ops
        op1 = PipeOpNOP(id="op1")
        op2 = PipeOpNOP(id="op2")
        
        graph.add_pipeop(op1)
        graph.add_pipeop(op2)
        
        assert len(graph.pipeops) == 2
        
        # Add edge
        graph.add_edge("op1", "output", "op2", "input")
        assert len(graph.edges) == 1
        
    def test_graph_validation_errors(self):
        """Test graph validation errors."""
        graph = Graph()
        op = PipeOpNOP(id="op1")
        graph.add_pipeop(op)
        
        # Duplicate ID
        with pytest.raises(ValueError, match="already exists"):
            graph.add_pipeop(PipeOpNOP(id="op1"))
            
        # Invalid edge - missing source
        with pytest.raises(ValueError, match="not found"):
            graph.add_edge("missing", "output", "op1", "input")
            
        # Invalid edge - missing channel
        with pytest.raises(ValueError, match="has no output channel"):
            graph.add_edge("op1", "missing", "op1", "input")
            
    def test_graph_topological_sort(self):
        """Test topological sorting."""
        graph = Graph()
        
        # Create linear pipeline: op1 -> op2 -> op3
        for i in range(1, 4):
            graph.add_pipeop(PipeOpNOP(id=f"op{i}"))
            
        graph.add_edge("op1", "output", "op2", "input")
        graph.add_edge("op2", "output", "op3", "input")
        
        sorted_ids = graph.topological_sort()
        assert sorted_ids == ["op1", "op2", "op3"]
        
    def test_graph_cycle_detection(self):
        """Test cycle detection."""
        graph = Graph()
        
        # Create cycle: op1 -> op2 -> op1
        graph.add_pipeop(PipeOpNOP(id="op1"))
        graph.add_pipeop(PipeOpNOP(id="op2"))
        
        graph.add_edge("op1", "output", "op2", "input")
        graph.add_edge("op2", "output", "op1", "input")
        
        with pytest.raises(ValueError, match="contains cycles"):
            graph.topological_sort()
            
    def test_graph_source_sink(self):
        """Test finding sources and sinks."""
        graph = Graph()
        
        # op1 -> op2 -> op3
        #     -> op4 ->
        for i in range(1, 5):
            graph.add_pipeop(PipeOpNOP(id=f"op{i}"))
            
        graph.add_edge("op1", "output", "op2", "input")
        graph.add_edge("op2", "output", "op3", "input")
        graph.add_edge("op1", "output", "op4", "input")
        graph.add_edge("op4", "output", "op3", "input")
        
        sources = graph.get_source_ops()
        sinks = graph.get_sink_ops()
        
        assert sources == ["op1"]
        assert sinks == ["op3"]


class TestGraphLearner:
    """Test GraphLearner functionality."""
    
    @pytest.fixture
    def iris_task(self):
        """Create Iris classification task."""
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        return TaskClassif(data=df, target='species', id='iris')
        
    @pytest.fixture
    def numeric_missing_task(self):
        """Create task with numeric features and missing values."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100) * 10,
            'x3': np.random.randn(100),
            'y': np.random.choice([0, 1], 100)
        })
        
        # Add some missing values
        df.loc[5:10, 'x1'] = np.nan
        df.loc[15:20, 'x2'] = np.nan
        
        return TaskClassif(data=df, target='y')
        
    def test_linear_pipeline_helper(self, iris_task):
        """Test linear_pipeline helper function."""
        # Create simple pipeline: scale -> learner
        scale = PipeOpScale()
        learner = PipeOpLearner(learner_sklearn(DecisionTreeClassifier()))
        
        graph = linear_pipeline(scale, learner)
        
        assert len(graph.pipeops) == 2
        assert len(graph.edges) == 1
        
        # Check connection
        edge = graph.edges[0]
        assert edge.src_id == scale.id
        assert edge.dst_id == learner.id
        
    def test_graph_learner_simple(self, iris_task):
        """Test simple GraphLearner."""
        # Create pipeline: scale -> learner
        graph = linear_pipeline(
            PipeOpScale(),
            PipeOpLearner(learner_sklearn(DecisionTreeClassifier(max_depth=3)))
        )
        
        gl = GraphLearner(graph)
        
        assert not gl.is_trained
        
        # Train
        gl.train(iris_task)
        assert gl.is_trained
        
        # Predict
        predictions = gl.predict(iris_task)
        assert isinstance(predictions, PredictionClassif)
        assert len(predictions.response) == iris_task.nrow
        
    def test_graph_learner_complex(self, numeric_missing_task):
        """Test complex pipeline."""
        # Create pipeline: impute -> scale -> select -> learner
        graph = linear_pipeline(
            PipeOpImpute(strategy="mean"),
            PipeOpScale(method="standard"),
            PipeOpSelect(k=2),
            PipeOpLearner(learner_sklearn(LogisticRegression()))
        )
        
        gl = GraphLearner(graph, id="complex_pipeline")
        
        # Train
        gl.train(numeric_missing_task)
        
        # Predict
        predictions = gl.predict(numeric_missing_task)
        assert isinstance(predictions, PredictionClassif)
        
        # Check that pipeline was executed correctly
        # The Select op should have reduced features to 2
        select_op = graph.pipeops["select"]
        assert len(select_op.state["selected_features"]) == 2
        
    def test_graph_learner_validation(self):
        """Test GraphLearner validation."""
        # Graph with multiple sources (op1 and op3 both have no incoming edges)
        graph = Graph()
        op1 = PipeOpNOP(id="op1")
        op2 = PipeOpNOP(id="op2")
        op3 = PipeOpNOP(id="op3")
        graph.add_pipeop(op1)
        graph.add_pipeop(op2)
        graph.add_pipeop(op3)
        graph.add_edge("op1", "output", "op2", "input")
        graph.add_edge("op3", "output", "op2", "input")
        
        with pytest.raises(ValueError, match="exactly one source"):
            GraphLearner(graph)
            
    def test_graph_learner_clone(self, iris_task):
        """Test cloning GraphLearner."""
        graph = linear_pipeline(
            PipeOpScale(),
            PipeOpLearner(learner_sklearn(DecisionTreeClassifier()))
        )
        
        gl = GraphLearner(graph)
        gl.train(iris_task)
        
        # Clone
        gl_clone = gl.clone()
        
        assert gl_clone.id == gl.id
        assert not gl_clone.is_trained  # Clone is reset
        assert gl.is_trained  # Original unchanged
        
        # Train clone
        gl_clone.train(iris_task)
        predictions = gl_clone.predict(iris_task)
        assert isinstance(predictions, PredictionClassif)


class TestIntegration:
    """Integration tests for pipelines."""
    
    @pytest.fixture
    def mixed_task(self):
        """Create task with mixed feature types."""
        np.random.seed(42)
        df = pd.DataFrame({
            'num1': np.random.randn(200),
            'num2': np.random.randn(200) * 5 + 2,
            'cat1': np.random.choice(['A', 'B', 'C'], 200),
            'cat2': np.random.choice(['X', 'Y'], 200),
            'y': np.random.choice([0, 1], 200)
        })
        
        # Add missing values
        df.loc[10:20, 'num1'] = np.nan
        df.loc[30:35, 'cat1'] = None
        
        return TaskClassif(data=df, target='y')
        
    def test_full_preprocessing_pipeline(self, mixed_task):
        """Test full preprocessing pipeline."""
        # Build comprehensive pipeline
        graph = linear_pipeline(
            PipeOpImpute(strategy="mean", add_indicator=True),
            PipeOpEncode(method="onehot"),
            PipeOpScale(method="standard"),
            PipeOpSelect(k=5),
            PipeOpLearner(learner_sklearn(
                LogisticRegression(max_iter=200)
            ))
        )
        
        gl = GraphLearner(graph, id="full_pipeline")
        
        # Should handle mixed types, missing values, etc.
        gl.train(mixed_task)
        predictions = gl.predict(mixed_task)
        
        assert isinstance(predictions, PredictionClassif)
        assert len(predictions.response) == mixed_task.nrow
        
        # Check accuracy is reasonable
        accuracy = (predictions.response == predictions.truth).mean()
        assert accuracy > 0.5  # Better than random
        
    def test_pipeline_in_resample(self, mixed_task):
        """Test using pipeline in resample."""
        from mlpy import resample
        from mlpy.resamplings import ResamplingCV
        from mlpy.measures import MeasureClassifAccuracy
        
        # Create pipeline learner
        graph = linear_pipeline(
            PipeOpImpute(),
            PipeOpEncode(),
            PipeOpScale(),
            PipeOpLearner(learner_sklearn(DecisionTreeClassifier()))
        )
        
        pipeline_learner = GraphLearner(graph)
        
        # Use in resample
        result = resample(
            task=mixed_task,
            learner=pipeline_learner,
            resampling=ResamplingCV(folds=3),
            measures=MeasureClassifAccuracy()
        )
        
        assert result.n_iters == 3
        assert result.score() > 0.5  # Should get decent accuracy
        
    def test_pipeline_in_benchmark(self, mixed_task):
        """Test using pipelines in benchmark."""
        from mlpy import benchmark
        from mlpy.resamplings import ResamplingHoldout
        from mlpy.measures import MeasureClassifAccuracy
        
        # Create different pipelines
        # Simple pipeline needs encoding for categorical features
        simple_pipeline = GraphLearner(
            linear_pipeline(
                PipeOpEncode(),
                PipeOpLearner(learner_sklearn(DecisionTreeClassifier()))
            ),
            id="simple"
        )
        
        complex_pipeline = GraphLearner(
            linear_pipeline(
                PipeOpImpute(),
                PipeOpEncode(),
                PipeOpScale(),
                PipeOpLearner(learner_sklearn(LogisticRegression()))
            ),
            id="complex"
        )
        
        # Benchmark
        result = benchmark(
            tasks=mixed_task,
            learners=[simple_pipeline, complex_pipeline],
            resampling=ResamplingHoldout(),
            measures=MeasureClassifAccuracy()
        )
        
        assert result.n_successful == 2
        assert result.n_errors == 0
        
        # Both should work
        scores = result.score_table()
        assert scores.shape == (1, 2)
        assert all(scores.iloc[0] > 0.5)