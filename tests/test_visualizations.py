"""Test visualization functionality."""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners import LearnerClassifFeatureless, LearnerRegrFeatureless
from mlpy.resamplings import ResamplingCV, ResamplingHoldout
from mlpy.measures import MeasureClassifAccuracy, MeasureRegrMSE
from mlpy.resample import resample
from mlpy.benchmark import benchmark
from mlpy.automl import ParamInt, ParamFloat, ParamCategorical, ParamSet, TunerGrid

# Import visualization functions
from mlpy.visualizations import (
    plot_theme, set_plot_theme, THEMES,
    plot_resample_boxplot, plot_resample_iterations,
    plot_benchmark_boxplot, plot_benchmark_heatmap, plot_benchmark_critical_difference,
    plot_tuning_performance, plot_tuning_parallel_coordinates,
    save_plot, create_figure
)
from mlpy.visualizations.base import PlotBase, get_colors


@pytest.fixture
def sample_task():
    """Create a sample classification task."""
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'y': np.random.choice(['A', 'B', 'C'], n)
    })
    return TaskClassif(data=data, target='y', id='test_task')


@pytest.fixture
def sample_resample_result(sample_task):
    """Create a sample resample result."""
    learner = LearnerClassifFeatureless(id='featureless')
    result = resample(
        task=sample_task,
        learner=learner,
        resampling=ResamplingCV(folds=5),
        measures=MeasureClassifAccuracy()
    )
    return result


@pytest.fixture
def sample_benchmark_result():
    """Create a sample benchmark result."""
    np.random.seed(42)
    
    # Create multiple tasks
    tasks = []
    for i in range(2):
        n = 50
        data = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'y': np.random.choice(['A', 'B'], n)
        })
        tasks.append(TaskClassif(data=data, target='y', id=f'task_{i}'))
        
    # Create learners
    learners = [
        LearnerClassifFeatureless(id='featureless', predict_type='response'),
        LearnerClassifFeatureless(id='featureless_weighted', predict_type='response', method='weighted')
    ]
    
    # Run benchmark
    result = benchmark(
        tasks=tasks,
        learners=learners,
        resampling=ResamplingHoldout(),
        measures=MeasureClassifAccuracy()
    )
    
    return result


@pytest.fixture
def sample_tune_result(sample_task):
    """Create a sample tuning result."""
    from mlpy.automl import TunerGrid
    
    # Create a simple tunable learner
    class TunableLearner(LearnerClassifFeatureless):
        def __init__(self, alpha=1.0, beta=0.5, **kwargs):
            super().__init__(**kwargs)
            self.alpha = alpha
            self.beta = beta
    
    learner = TunableLearner(id='tunable')
    
    # Define parameter space
    param_set = ParamSet([
        ParamFloat("alpha", lower=0.1, upper=2.0),
        ParamFloat("beta", lower=0.1, upper=1.0)
    ])
    
    # Run tuning
    tuner = TunerGrid(resolution=3)
    result = tuner.tune(
        learner=learner,
        task=sample_task,
        resampling=ResamplingHoldout(),
        measure=MeasureClassifAccuracy(),
        param_set=param_set
    )
    
    return result


class TestPlotBase:
    """Test base plotting functionality."""
    
    def test_themes(self):
        """Test theme functionality."""
        # Test getting current theme
        theme = plot_theme()
        assert isinstance(theme, dict)
        assert 'figure.figsize' in theme
        
        # Test setting theme
        plot_theme('minimal')
        theme = plot_theme()
        assert theme == THEMES['minimal']
        
        # Test invalid theme
        with pytest.raises(ValueError):
            plot_theme('invalid_theme')
            
        # Reset to default
        plot_theme('default')
        
    def test_set_plot_theme(self):
        """Test matplotlib theme setting."""
        original_figsize = plt.rcParams['figure.figsize']
        
        # Set theme
        set_plot_theme('publication')
        assert plt.rcParams['font.family'] == ['serif']
        
        # Set custom theme
        custom = {'figure.figsize': (20, 10)}
        set_plot_theme(custom)
        assert plt.rcParams['figure.figsize'] == [20, 10]
        
        # Reset
        set_plot_theme('default')
        
    def test_get_colors(self):
        """Test color palette functionality."""
        # Test default colors
        colors = get_colors(5)
        assert len(colors) == 5
        assert all(isinstance(c, tuple) and len(c) == 3 for c in colors)
        
        # Test hex colors
        colors_hex = get_colors(3, as_hex=True)
        assert all(isinstance(c, str) and c.startswith('#') for c in colors_hex)
        
        # Test different palette
        colors_cb = get_colors(5, palette='colorblind')
        assert len(colors_cb) == 5
        
        # Test cycling
        colors_many = get_colors(20, palette='default')
        assert len(colors_many) == 20


class TestResampleVisualizations:
    """Test ResampleResult visualizations."""
    
    def test_resample_boxplot(self, sample_resample_result):
        """Test resample boxplot."""
        fig, ax = plot_resample_boxplot(sample_resample_result)
        
        assert fig is not None
        assert ax is not None
        assert ax.get_ylabel() == 'classif.acc score'
        
        # Test with options
        fig2, ax2 = plot_resample_boxplot(
            sample_resample_result,
            show_points=False,
            figsize=(8, 6)
        )
        assert fig2.get_figwidth() == 8
        assert fig2.get_figheight() == 6
        
        plt.close('all')
        
    def test_resample_iterations(self, sample_resample_result):
        """Test iteration plot."""
        fig, ax = plot_resample_iterations(sample_resample_result)
        
        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() == 'Iteration'
        assert ax.get_ylabel() == 'Score'
        
        # Check that lines were plotted
        lines = ax.get_lines()
        assert len(lines) > 0
        
        plt.close('all')
        
    def test_resample_plot_methods(self, sample_resample_result):
        """Test plot methods added to ResampleResult."""
        # Test that methods exist
        assert hasattr(sample_resample_result, 'plot_boxplot')
        assert hasattr(sample_resample_result, 'plot_iterations')
        
        # Test calling methods
        fig, ax = sample_resample_result.plot_boxplot()
        assert fig is not None
        
        plt.close('all')


class TestBenchmarkVisualizations:
    """Test BenchmarkResult visualizations."""
    
    def test_benchmark_boxplot(self, sample_benchmark_result):
        """Test benchmark boxplot."""
        fig, ax = plot_benchmark_boxplot(sample_benchmark_result)
        
        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() == 'Learner'
        
        # Test grouping by task
        fig2, ax2 = plot_benchmark_boxplot(
            sample_benchmark_result,
            by='task'
        )
        assert ax2.get_xlabel() == 'Task'
        
        plt.close('all')
        
    def test_benchmark_heatmap(self, sample_benchmark_result):
        """Test benchmark heatmap."""
        fig, ax = plot_benchmark_heatmap(sample_benchmark_result)
        
        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() == 'Learner'
        assert ax.get_ylabel() == 'Task'
        
        plt.close('all')
        
    def test_benchmark_critical_difference(self, sample_benchmark_result):
        """Test critical difference diagram."""
        fig, ax = plot_benchmark_critical_difference(sample_benchmark_result)
        
        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() == 'Average Rank'
        
        plt.close('all')
        
    def test_benchmark_plot_methods(self, sample_benchmark_result):
        """Test plot methods added to BenchmarkResult."""
        # Test that methods exist
        assert hasattr(sample_benchmark_result, 'plot_boxplot')
        assert hasattr(sample_benchmark_result, 'plot_heatmap')
        assert hasattr(sample_benchmark_result, 'plot_critical_difference')
        
        # Test calling methods
        fig, ax = sample_benchmark_result.plot_heatmap()
        assert fig is not None
        
        plt.close('all')


class TestTuningVisualizations:
    """Test TuneResult visualizations."""
    
    def test_tuning_performance(self, sample_tune_result):
        """Test tuning performance plot."""
        fig, ax = plot_tuning_performance(sample_tune_result)
        
        assert fig is not None
        assert ax is not None
        
        # Test with specific parameter
        fig2, ax2 = plot_tuning_performance(
            sample_tune_result,
            param='alpha'
        )
        assert ax2.get_xlabel() == 'alpha'
        
        plt.close('all')
        
    def test_tuning_parallel_coordinates(self, sample_tune_result):
        """Test parallel coordinates plot."""
        fig, ax = plot_tuning_parallel_coordinates(sample_tune_result)
        
        assert fig is not None
        assert ax is not None
        
        # Check that lines were plotted
        lines = ax.get_lines()
        assert len(lines) > 0
        
        plt.close('all')
        
    def test_tune_plot_methods(self, sample_tune_result):
        """Test plot methods added to TuneResult."""
        # Test that methods exist
        assert hasattr(sample_tune_result, 'plot_performance')
        assert hasattr(sample_tune_result, 'plot_parallel_coordinates')
        
        # Test calling methods
        fig, ax = sample_tune_result.plot_performance()
        assert fig is not None
        
        plt.close('all')


class TestVisualizationUtils:
    """Test visualization utility functions."""
    
    def test_create_figure(self):
        """Test figure creation."""
        fig, ax = create_figure()
        assert fig is not None
        assert ax is not None
        
        # Test with subplots
        fig2, axes = create_figure(2, 2)
        assert axes.shape == (2, 2)
        
        plt.close('all')
        
    def test_save_plot(self):
        """Test plot saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple plot
            fig, ax = create_figure()
            ax.plot([1, 2, 3], [1, 2, 3])
            
            # Save in single format
            output_path = Path(tmpdir) / "test_plot"
            save_plot(fig, output_path)
            assert (output_path.with_suffix('.png')).exists()
            
            # Save in multiple formats
            output_path2 = Path(tmpdir) / "test_plot2"
            save_plot(fig, output_path2, formats=['png', 'pdf'])
            assert (output_path2.with_suffix('.png')).exists()
            assert (output_path2.with_suffix('.pdf')).exists()
            
        plt.close('all')


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_scores(self):
        """Test plotting with empty or all-NaN scores."""
        # Create result with NaN scores
        task = TaskClassif(
            data=pd.DataFrame({'x': [1, 2], 'y': ['A', 'B']}),
            target='y'
        )
        learner = LearnerClassifFeatureless()
        
        result = resample(
            task=task,
            learner=learner,
            resampling=ResamplingHoldout(),
            measures=MeasureClassifAccuracy()
        )
        
        # Manually set scores to NaN
        result.scores[result.measures[0].id] = [np.nan]
        
        # Should handle gracefully
        with pytest.warns(UserWarning):
            fig, ax = plot_resample_boxplot(result)
            
        plt.close('all')
        
    def test_plot_base_abstract(self):
        """Test that PlotBase is abstract."""
        with pytest.raises(TypeError):
            PlotBase()
            
    def test_plot_before_create(self):
        """Test error when trying to show/save before creating."""
        class TestPlot(PlotBase):
            def create(self):
                return None, None
                
        plotter = TestPlot()
        
        with pytest.raises(ValueError, match="Plot has not been created"):
            plotter.show()
            
        with pytest.raises(ValueError, match="Plot has not been created"):
            plotter.save("test.png")


if __name__ == "__main__":
    pytest.main([__file__])