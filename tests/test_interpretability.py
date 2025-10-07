"""Test interpretability functionality."""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners import LearnerClassifFeatureless, LearnerRegrFeatureless
from mlpy.interpretability import (
    Interpreter, InterpretationResult, FeatureImportance,
    SHAPInterpreter, SHAPExplanation,
    LIMEInterpreter, LIMEExplanation,
    plot_feature_importance, plot_shap_summary, plot_lime_explanation,
    plot_interpretation_comparison, create_interpretation_report
)


@pytest.fixture
def sample_task_classif():
    """Create a sample classification task."""
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n),
        'y': np.random.choice(['A', 'B'], n)
    })
    return TaskClassif(data=data, target='y', id='test_task')


@pytest.fixture
def sample_task_regr():
    """Create a sample regression task."""
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n),
        'y': np.random.randn(n)
    })
    return TaskRegr(data=data, target='y', id='test_task')


@pytest.fixture
def trained_learner_classif(sample_task_classif):
    """Create a trained classification learner."""
    learner = LearnerClassifFeatureless(id='test_learner')
    learner.train(sample_task_classif)
    return learner


@pytest.fixture
def trained_learner_regr(sample_task_regr):
    """Create a trained regression learner."""
    learner = LearnerRegrFeatureless(id='test_learner')
    learner.train(sample_task_regr)
    return learner


class TestFeatureImportance:
    """Test FeatureImportance class."""
    
    def test_creation(self):
        """Test FeatureImportance creation."""
        features = ['x1', 'x2', 'x3']
        importances = np.array([0.5, 0.3, 0.2])
        
        fi = FeatureImportance(
            features=features,
            importances=importances,
            method='test'
        )
        
        assert fi.features == features
        assert np.array_equal(fi.importances, importances)
        assert fi.method == 'test'
        
    def test_validation(self):
        """Test input validation."""
        with pytest.raises(ValueError, match="same length"):
            FeatureImportance(
                features=['x1', 'x2'],
                importances=[0.5, 0.3, 0.2],
                method='test'
            )
            
    def test_as_dataframe(self):
        """Test conversion to DataFrame."""
        fi = FeatureImportance(
            features=['x1', 'x2', 'x3'],
            importances=[0.2, 0.5, 0.3],
            method='test'
        )
        
        df = fi.as_dataframe()
        assert len(df) == 3
        assert df.iloc[0]['feature'] == 'x2'  # Highest importance
        assert df.iloc[0]['importance'] == 0.5
        
    def test_top_features(self):
        """Test getting top features."""
        fi = FeatureImportance(
            features=['x1', 'x2', 'x3', 'x4'],
            importances=[0.1, 0.4, 0.3, 0.2],
            method='test'
        )
        
        top = fi.top_features(2)
        assert top == ['x2', 'x3']


class TestInterpretationResult:
    """Test InterpretationResult class."""
    
    def test_creation(self, trained_learner_classif, sample_task_classif):
        """Test result creation."""
        result = InterpretationResult(
            learner=trained_learner_classif,
            task=sample_task_classif,
            method='test'
        )
        
        assert result.learner == trained_learner_classif
        assert result.task == sample_task_classif
        assert result.method == 'test'
        assert result.id == 'test_result'
        
    def test_global_importance(self, trained_learner_classif, sample_task_classif):
        """Test global importance handling."""
        fi = FeatureImportance(
            features=['x1', 'x2'],
            importances=[0.6, 0.4],
            method='test'
        )
        
        result = InterpretationResult(
            learner=trained_learner_classif,
            task=sample_task_classif,
            method='test',
            global_importance=fi
        )
        
        assert result.has_global_importance()
        assert result.global_importance == fi
        
    def test_local_explanations(self, trained_learner_classif, sample_task_classif):
        """Test local explanations handling."""
        local_exp = {
            0: "explanation_0",
            5: "explanation_5"
        }
        
        result = InterpretationResult(
            learner=trained_learner_classif,
            task=sample_task_classif,
            method='test',
            local_explanations=local_exp
        )
        
        assert result.has_local_explanations()
        assert result.get_local_explanation(0) == "explanation_0"
        
        with pytest.raises(ValueError, match="No explanation"):
            result.get_local_explanation(10)


class TestSHAPInterpreter:
    """Test SHAP interpreter."""
    
    @patch('mlpy.interpretability.shap_interpreter._HAS_SHAP', False)
    def test_missing_shap(self):
        """Test error when SHAP not installed."""
        with pytest.raises(ImportError, match="SHAP is not installed"):
            SHAPInterpreter()
            
    @patch('mlpy.interpretability.shap_interpreter._HAS_SHAP', True)
    def test_creation(self):
        """Test SHAP interpreter creation."""
        interp = SHAPInterpreter(explainer_type="tree")
        assert interp.id == "shap_interpreter"
        assert interp.explainer_type == "tree"
        
    @patch('mlpy.interpretability.shap_interpreter._HAS_SHAP', True)
    @patch('mlpy.interpretability.shap_interpreter.shap')
    def test_interpret_mock(self, mock_shap, trained_learner_classif, sample_task_classif):
        """Test interpretation with mocked SHAP."""
        # Mock explainer
        mock_explainer = Mock()
        mock_shap.TreeExplainer.return_value = mock_explainer
        
        # Mock SHAP values
        n_samples = len(sample_task_classif.X)
        n_features = len(sample_task_classif.feature_names)
        mock_values = np.random.randn(n_samples, n_features)
        mock_explainer.return_value = Mock(
            values=mock_values,
            base_values=0.5,
            data=sample_task_classif.X.values
        )
        
        # Create interpreter and run
        interp = SHAPInterpreter(explainer_type="tree")
        result = interp.interpret(
            trained_learner_classif,
            sample_task_classif,
            indices=[0, 1]
        )
        
        assert isinstance(result, InterpretationResult)
        assert result.method == "shap"
        assert result.has_global_importance()
        assert result.has_local_explanations()
        assert len(result.local_explanations) == 2
        
    def test_shap_explanation(self):
        """Test SHAPExplanation class."""
        values = np.array([[0.1, 0.2], [0.3, 0.4]])
        base_values = 0.5
        
        exp = SHAPExplanation(
            values=values,
            base_values=base_values,
            feature_names=['x1', 'x2']
        )
        
        assert np.array_equal(exp.values, values)
        assert exp.base_values == base_values
        
        # Test feature importance calculation
        fi = exp.get_feature_importance()
        assert fi.method == "shap_mean_abs"
        assert len(fi.features) == 2


class TestLIMEInterpreter:
    """Test LIME interpreter."""
    
    @patch('mlpy.interpretability.lime_interpreter._HAS_LIME', False)
    def test_missing_lime(self):
        """Test error when LIME not installed."""
        with pytest.raises(ImportError, match="LIME is not installed"):
            LIMEInterpreter()
            
    @patch('mlpy.interpretability.lime_interpreter._HAS_LIME', True)
    def test_creation(self):
        """Test LIME interpreter creation."""
        interp = LIMEInterpreter(num_features=5)
        assert interp.id == "lime_interpreter"
        assert interp.num_features == 5
        assert interp.mode == "tabular"
        
    @patch('mlpy.interpretability.lime_interpreter._HAS_LIME', True)
    @patch('mlpy.interpretability.lime_interpreter.lime')
    def test_interpret_mock(self, mock_lime, trained_learner_classif, sample_task_classif):
        """Test interpretation with mocked LIME."""
        # Mock explainer
        mock_explainer = Mock()
        mock_lime.lime_tabular.LimeTabularExplainer.return_value = mock_explainer
        
        # Mock explanation
        mock_exp = Mock()
        mock_exp.as_list.return_value = [
            ('x1 > 0.5', 0.3),
            ('x2 <= 1.0', -0.2),
            ('x3', 0.1)
        ]
        mock_explainer.explain_instance.return_value = mock_exp
        
        # Create interpreter and run
        interp = LIMEInterpreter(num_features=3)
        result = interp.interpret(
            trained_learner_classif,
            sample_task_classif,
            indices=[0]
        )
        
        assert isinstance(result, InterpretationResult)
        assert result.method == "lime"
        assert result.has_local_explanations()
        assert 0 in result.local_explanations
        
    def test_lime_explanation(self):
        """Test LIMEExplanation class."""
        exp = LIMEExplanation(
            instance_idx=0,
            explanation=Mock(),
            feature_importance={'x1': 0.5, 'x2': -0.3, 'x3': 0.1},
            prediction=np.array([0.7, 0.3])
        )
        
        # Test top features
        top = exp.get_top_features(2)
        assert len(top) == 2
        assert top[0][0] == 'x1'
        assert top[0][1] == 0.5
        
        # Test DataFrame conversion
        df = exp.as_dataframe()
        assert len(df) == 3
        assert df.iloc[0]['feature'] == 'x1'


class TestVisualizationUtils:
    """Test visualization utilities."""
    
    def test_plot_feature_importance(self):
        """Test feature importance plotting."""
        fi = FeatureImportance(
            features=['x1', 'x2', 'x3'],
            importances=[0.5, 0.3, 0.2],
            method='test'
        )
        
        fig, ax = plot_feature_importance(fi, max_features=2)
        
        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() == 'Importance Score'
        assert 'Feature Importance' in ax.get_title()
        
        plt.close('all')
        
    @patch('mlpy.interpretability.utils._HAS_SHAP', True)
    @patch('mlpy.interpretability.utils.shap')
    def test_plot_shap_summary_mock(self, mock_shap):
        """Test SHAP summary plot with mock."""
        exp = SHAPExplanation(
            values=np.random.randn(10, 3),
            base_values=0.5,
            data=np.random.randn(10, 3),
            feature_names=['x1', 'x2', 'x3']
        )
        
        mock_shap.summary_plot = Mock()
        
        fig, ax = plot_shap_summary(exp)
        
        assert mock_shap.summary_plot.called
        plt.close('all')
        
    def test_plot_lime_explanation(self):
        """Test LIME explanation plotting."""
        exp = LIMEExplanation(
            instance_idx=0,
            explanation=Mock(),
            feature_importance={'x1': 0.5, 'x2': -0.3, 'x3': 0.1},
            prediction=np.array([0.7, 0.3])
        )
        
        fig, ax = plot_lime_explanation(exp, num_features=3)
        
        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() == 'Feature Contribution'
        assert 'LIME Explanation' in ax.get_title()
        
        plt.close('all')
        
    def test_plot_interpretation_comparison(self, trained_learner_classif, sample_task_classif):
        """Test comparison plotting."""
        # Create mock results
        fi1 = FeatureImportance(
            features=['x1', 'x2', 'x3'],
            importances=[0.5, 0.3, 0.2],
            method='method1'
        )
        
        fi2 = FeatureImportance(
            features=['x1', 'x2', 'x3'],
            importances=[0.4, 0.4, 0.2],
            method='method2'
        )
        
        result1 = InterpretationResult(
            learner=trained_learner_classif,
            task=sample_task_classif,
            method='method1',
            global_importance=fi1
        )
        
        result2 = InterpretationResult(
            learner=trained_learner_classif,
            task=sample_task_classif,
            method='method2',
            global_importance=fi2
        )
        
        fig, ax = plot_interpretation_comparison([result1, result2])
        
        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel().startswith('Importance Score')
        assert ax.get_legend() is not None
        
        plt.close('all')
        
    def test_create_interpretation_report(self, trained_learner_classif, sample_task_classif):
        """Test report creation."""
        fi = FeatureImportance(
            features=['x1', 'x2', 'x3'],
            importances=[0.5, 0.3, 0.2],
            method='test'
        )
        
        result = InterpretationResult(
            learner=trained_learner_classif,
            task=sample_task_classif,
            method='test',
            global_importance=fi,
            metadata={'param1': 'value1'}
        )
        
        # Test report generation
        report = create_interpretation_report(result)
        
        assert 'Model Interpretation Report' in report
        assert 'Method: TEST' in report
        assert 'Global Feature Importance' in report
        assert 'x1' in report
        assert '0.5000' in report
        assert 'param1: value1' in report
        
        # Test saving
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            output_path = f.name
            
        create_interpretation_report(result, output_path=output_path)
        
        assert os.path.exists(output_path)
        with open(output_path, 'r') as f:
            saved_report = f.read()
        assert saved_report == report
        
        os.unlink(output_path)


class TestIntegration:
    """Test integration scenarios."""
    
    def test_interpreter_abstract(self):
        """Test that Interpreter is abstract."""
        with pytest.raises(TypeError):
            Interpreter()
            
    def test_result_plot_methods(self, trained_learner_classif, sample_task_classif):
        """Test plot methods added to InterpretationResult."""
        fi = FeatureImportance(
            features=['x1', 'x2'],
            importances=[0.6, 0.4],
            method='test'
        )
        
        result = InterpretationResult(
            learner=trained_learner_classif,
            task=sample_task_classif,
            method='test',
            global_importance=fi
        )
        
        # Test plot_importance method
        assert hasattr(result, 'plot_importance')
        fig, ax = result.plot_importance()
        assert fig is not None
        plt.close('all')
        
        # Test create_report method
        assert hasattr(result, 'create_report')
        report = result.create_report()
        assert 'Model Interpretation Report' in report
        
    def test_no_global_importance(self, trained_learner_classif, sample_task_classif):
        """Test handling when no global importance available."""
        result = InterpretationResult(
            learner=trained_learner_classif,
            task=sample_task_classif,
            method='test'
        )
        
        assert not result.has_global_importance()
        
        with pytest.raises(ValueError, match="No global importance"):
            result.plot_importance()


if __name__ == "__main__":
    pytest.main([__file__])