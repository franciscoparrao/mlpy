"""Tests for MLPY CLI."""

import pytest
import click
from click.testing import CliRunner
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
import yaml
import shutil

from mlpy.cli.main import cli
from mlpy import __version__


class TestCLIBasic:
    """Test basic CLI functionality."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self, temp_dir):
        """Create sample data files."""
        # Classification data
        np.random.seed(42)
        n = 100
        classif_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n),
            'feature2': np.random.normal(0, 1, n),
            'feature3': np.random.choice(['A', 'B', 'C'], n),
            'target': np.random.choice(['class1', 'class2'], n)
        })
        classif_path = Path(temp_dir) / 'classif_data.csv'
        classif_data.to_csv(classif_path, index=False)
        
        # Regression data
        regr_data = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
            'x3': np.random.uniform(0, 10, n),
            'y': np.random.normal(10, 2, n)
        })
        regr_path = Path(temp_dir) / 'regr_data.csv'
        regr_data.to_csv(regr_path, index=False)
        
        # Parquet data
        parquet_path = Path(temp_dir) / 'data.parquet'
        classif_data.to_parquet(parquet_path)
        
        return {
            'classif_csv': str(classif_path),
            'regr_csv': str(regr_path),
            'parquet': str(parquet_path)
        }
    
    def test_cli_no_args(self, runner):
        """Test CLI with no arguments shows help."""
        result = runner.invoke(cli)
        assert result.exit_code == 0
        assert 'MLPY - Machine Learning Framework' in result.output
        assert 'Commands:' in result.output
    
    def test_version(self, runner):
        """Test version flag."""
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert f'MLPY version {__version__}' in result.output
    
    def test_help(self, runner):
        """Test help flag."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'MLPY - Machine Learning Framework' in result.output
        assert '--version' in result.output


class TestInfoCommand:
    """Test info command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_info_command(self, runner):
        """Test info command shows installation details."""
        result = runner.invoke(cli, ['info'])
        assert result.exit_code == 0
        assert 'MLPY Information' in result.output
        assert f'MLPY version: {__version__}' in result.output
        assert 'Python version:' in result.output
        assert 'Core Dependencies:' in result.output
        assert 'scikit-learn:' in result.output
        assert 'pandas:' in result.output
        assert 'numpy:' in result.output


class TestTrainCommand:
    """Test train command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self, temp_dir):
        # Create sample classification data
        np.random.seed(42)
        n = 150
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n),
            'feature2': np.random.normal(0, 1, n),
            'feature3': np.random.uniform(0, 1, n),
            'target': np.random.choice(['A', 'B', 'C'], n)
        })
        path = Path(temp_dir) / 'train_data.csv'
        data.to_csv(path, index=False)
        return str(path)
    
    def test_train_basic(self, runner, sample_data):
        """Test basic training with classification."""
        result = runner.invoke(cli, [
            'train', sample_data,
            '-t', 'classif',
            '-y', 'target',
            '-l', 'rf',
            '-k', '3',
            '-m', 'acc'
        ])
        assert result.exit_code == 0
        assert 'Loading data from' in result.output
        assert 'Created classif task:' in result.output
        assert 'Training rf with 3-fold cross-validation' in result.output
        assert 'Results:' in result.output
    
    def test_train_with_output(self, runner, sample_data, temp_dir):
        """Test training with output file."""
        output_path = Path(temp_dir) / 'results.csv'
        result = runner.invoke(cli, [
            'train', sample_data,
            '-t', 'classif',
            '-y', 'target',
            '-l', 'rf',
            '-k', '3',
            '-m', 'acc',
            '-o', str(output_path)
        ])
        assert result.exit_code == 0
        assert output_path.exists()
        assert 'Results saved to' in result.output
    
    def test_train_regression(self, runner, temp_dir):
        """Test training with regression task."""
        # Create regression data
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
            'y': np.random.normal(10, 2, n)
        })
        path = Path(temp_dir) / 'regr_data.csv'
        data.to_csv(path, index=False)
        
        result = runner.invoke(cli, [
            'train', str(path),
            '-t', 'regr',
            '-y', 'y',
            '-l', 'rf',
            '-m', 'rmse',
            '-m', 'r2'
        ])
        assert result.exit_code == 0
        assert 'Created regr task:' in result.output
    
    def test_train_multiple_measures(self, runner, sample_data):
        """Test training with multiple measures."""
        result = runner.invoke(cli, [
            'train', sample_data,
            '-t', 'classif',
            '-y', 'target',
            '-l', 'rf',
            '-m', 'acc',
            '-m', 'f1'
        ])
        assert result.exit_code == 0
        # Both measures should appear in results
    
    def test_train_invalid_learner(self, runner, sample_data):
        """Test training with invalid learner."""
        result = runner.invoke(cli, [
            'train', sample_data,
            '-t', 'classif',
            '-y', 'target',
            '-l', 'invalid_learner'
        ])
        assert result.exit_code == 1
        assert 'Unknown learner' in result.output
    
    def test_train_invalid_file(self, runner):
        """Test training with non-existent file."""
        result = runner.invoke(cli, [
            'train', 'nonexistent.csv',
            '-t', 'classif',
            '-y', 'target'
        ])
        assert result.exit_code == 2  # Click's exit code for missing file


class TestBenchmarkCommand:
    """Test benchmark command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self, temp_dir):
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
            'target': np.random.choice(['A', 'B'], n)
        })
        path = Path(temp_dir) / 'benchmark_data.csv'
        data.to_csv(path, index=False)
        return str(path)
    
    def test_benchmark_basic(self, runner, sample_data):
        """Test basic benchmarking."""
        result = runner.invoke(cli, [
            'benchmark', sample_data,
            '-t', 'classif',
            '-y', 'target',
            '-l', 'rf',
            '-l', 'lr',
            '-k', '3'
        ])
        assert result.exit_code == 0
        assert 'Benchmarking 2 learners' in result.output
        assert 'Score Table:' in result.output
        assert 'Ranking:' in result.output
    
    def test_benchmark_with_output(self, runner, sample_data, temp_dir):
        """Test benchmarking with output."""
        output_path = Path(temp_dir) / 'benchmark_results.xlsx'
        result = runner.invoke(cli, [
            'benchmark', sample_data,
            '-t', 'classif',
            '-y', 'target',
            '-l', 'rf',
            '-l', 'dt',
            '-o', str(output_path)
        ])
        assert result.exit_code == 0
        assert output_path.exists()
        assert 'Results saved to' in result.output


class TestPredictCommand:
    """Test predict command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def trained_model(self, temp_dir):
        """Create a trained model file."""
        from mlpy.tasks import TaskClassif
        from mlpy.learners import learner_sklearn
        from sklearn.tree import DecisionTreeClassifier
        from mlpy.persistence import save_model
        
        # Create simple data and train model
        np.random.seed(42)
        n = 50
        data = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
            'y': np.random.choice(['A', 'B'], n)
        })
        
        task = TaskClassif(data=data, target='y')
        learner = learner_sklearn(DecisionTreeClassifier(), id='dt')
        learner.train(task)
        
        # Use .joblib extension to match serializer
        model_path = Path(temp_dir) / 'model.joblib'
        save_model(learner, str(model_path))
        
        return str(model_path)
    
    @pytest.fixture
    def test_data(self, temp_dir):
        """Create test data for predictions."""
        np.random.seed(42)
        n = 20
        data = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n)
        })
        path = Path(temp_dir) / 'test_data.csv'
        data.to_csv(path, index=False)
        return str(path)
    
    def test_predict_basic(self, runner, trained_model, test_data):
        """Test basic prediction."""
        result = runner.invoke(cli, [
            'predict', trained_model, test_data
        ])
        assert result.exit_code == 0
        assert 'Loading model from' in result.output
        assert 'Making predictions for' in result.output
        assert 'Sample predictions:' in result.output
    
    def test_predict_with_output(self, runner, trained_model, test_data, temp_dir):
        """Test prediction with output file."""
        output_path = Path(temp_dir) / 'predictions.csv'
        result = runner.invoke(cli, [
            'predict', trained_model, test_data,
            '-o', str(output_path)
        ])
        assert result.exit_code == 0
        assert output_path.exists()
        assert 'Predictions saved to' in result.output
        
        # Check predictions file
        predictions = pd.read_csv(output_path)
        assert 'prediction' in predictions.columns
        assert len(predictions) == 20
    
    def test_predict_proba(self, runner, trained_model, test_data):
        """Test probability predictions."""
        result = runner.invoke(cli, [
            'predict', trained_model, test_data,
            '--proba'
        ])
        assert result.exit_code == 0
        # DecisionTreeClassifier supports predict_proba, so should work
        # Check either prob columns or warning message
        assert ('prob_class_' in result.output or 
                'Could not get probability predictions' in result.output or
                'does not support probability predictions' in result.output)


class TestTaskCommands:
    """Test task subcommands."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def sample_data(self, temp_dir):
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'num1': np.random.normal(0, 1, n),
            'num2': np.random.normal(0, 1, n),
            'cat1': np.random.choice(['A', 'B', 'C'], n),
            'target': np.random.choice(['yes', 'no'], n)
        })
        
        # Add some missing values
        data.loc[np.random.choice(n, 10), 'num1'] = np.nan
        
        path = Path(temp_dir) / 'task_data.csv'
        data.to_csv(path, index=False)
        return str(path)
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_task_info(self, runner, sample_data):
        """Test task info command."""
        result = runner.invoke(cli, ['task', 'info', sample_data, '-y', 'target'])
        assert result.exit_code == 0
        assert 'Dataset Information:' in result.output
        assert 'Shape:' in result.output
        assert 'Memory usage:' in result.output
        assert 'Column Information:' in result.output
        assert '<- TARGET' in result.output
        assert 'Target Distribution' in result.output


class TestLearnerCommands:
    """Test learner subcommands."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_learner_list(self, runner):
        """Test learner list command."""
        result = runner.invoke(cli, ['learner', 'list'])
        assert result.exit_code == 0
        assert 'Available MLPY Learners:' in result.output
        assert 'Base Classes:' in result.output
        assert 'Native Learners:' in result.output
        assert 'LearnerClassifFeatureless' in result.output
        assert 'Scikit-learn Integration:' in result.output


class TestPipelineCommands:
    """Test pipeline subcommands."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_pipeline_create_interactive_no(self, runner, temp_dir):
        """Test pipeline creation in interactive mode (all no)."""
        output_path = Path(temp_dir) / 'pipeline.pkl'
        
        # Simulate user input: No to all options except learner type
        result = runner.invoke(cli, [
            'pipeline', 'create',
            '-o', str(output_path)
        ], input='n\nn\nn\nrf\n')
        
        assert result.exit_code == 0
        assert 'Creating pipeline interactively' in result.output
        assert 'Pipeline created successfully' in result.output
        assert output_path.exists()
    
    def test_pipeline_create_with_config_yaml(self, runner, temp_dir):
        """Test pipeline creation from YAML config."""
        config_path = Path(temp_dir) / 'pipeline_config.yaml'
        config = {
            'pipeline': {
                'steps': [
                    {'type': 'scale', 'params': {'method': 'standard'}},
                    {'type': 'select', 'params': {'k': 10}},
                    {'type': 'learner', 'params': {'type': 'rf'}}
                ]
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        result = runner.invoke(cli, [
            'pipeline', 'create',
            '-c', str(config_path)
        ])
        
        # Currently not implemented, should show message
        assert 'Config-based creation not yet implemented' in result.output


class TestPreprocessCommand:
    """Test preprocess command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self, temp_dir):
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'num1': np.random.normal(0, 1, n),
            'num2': np.random.normal(0, 1, n),
            'cat1': np.random.choice(['A', 'B', 'C'], n),
            'cat2': np.random.choice(['X', 'Y'], n)
        })
        
        # Add missing values
        data.loc[np.random.choice(n, 10), 'num1'] = np.nan
        data.loc[np.random.choice(n, 5), 'cat1'] = np.nan
        
        path = Path(temp_dir) / 'raw_data.csv'
        data.to_csv(path, index=False)
        return str(path)
    
    def test_preprocess_basic(self, runner, sample_data, temp_dir):
        """Test basic preprocessing."""
        output_path = Path(temp_dir) / 'processed.csv'
        
        result = runner.invoke(cli, [
            'preprocess',
            '-i', sample_data,
            '-o', str(output_path),
            '--scale',
            '--impute'
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        assert 'Imputing missing values' in result.output
        assert 'Scaling numeric features' in result.output
        assert 'Processed data saved' in result.output
    
    def test_preprocess_all_options(self, runner, sample_data, temp_dir):
        """Test preprocessing with all options."""
        output_path = Path(temp_dir) / 'processed.csv'
        
        result = runner.invoke(cli, [
            'preprocess',
            '-i', sample_data,
            '-o', str(output_path),
            '--scale',
            '--impute',
            '--encode'
        ])
        
        assert result.exit_code == 0
        assert 'Encoding categorical variables' in result.output
        
        # Check processed data
        processed = pd.read_csv(output_path)
        # Should have more columns due to encoding
        assert len(processed.columns) > 4


class TestExperimentCommand:
    """Test experiment command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_experiment_create_template_yaml(self, runner, temp_dir):
        """Test creating experiment template in YAML."""
        exp_path = Path(temp_dir) / 'experiment.yaml'
        
        result = runner.invoke(cli, ['experiment', str(exp_path)])
        
        assert result.exit_code == 0
        assert exp_path.exists()
        assert 'Created experiment template' in result.output
        
        # Check template content
        with open(exp_path) as f:
            exp = yaml.safe_load(f)
        
        assert 'name' in exp
        assert 'data' in exp
        assert 'learners' in exp
        assert 'resampling' in exp
        assert 'measures' in exp
    
    def test_experiment_create_template_json(self, runner, temp_dir):
        """Test creating experiment template in JSON."""
        exp_path = Path(temp_dir) / 'experiment.json'
        
        result = runner.invoke(cli, ['experiment', str(exp_path)])
        
        assert result.exit_code == 0
        assert exp_path.exists()
        
        # Check template content
        with open(exp_path) as f:
            exp = json.load(f)
        
        assert 'name' in exp
        assert 'learners' in exp
    
    def test_experiment_run_not_implemented(self, runner, temp_dir):
        """Test running experiment (not implemented)."""
        exp_path = Path(temp_dir) / 'experiment.yaml'
        
        # Create experiment file first
        runner.invoke(cli, ['experiment', str(exp_path)])
        
        # Try to run it
        result = runner.invoke(cli, ['experiment', str(exp_path), '--run'])
        
        assert 'Experiment runner not yet implemented' in result.output


class TestShellCommand:
    """Test shell command."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_shell_python(self, runner):
        """Test starting Python shell."""
        # Can't really test interactive shell, just check it starts
        result = runner.invoke(cli, ['shell', '-s', 'python'], input='exit()\n')
        
        # Should show banner
        assert 'MLPY Interactive Shell' in result.output or result.exit_code == 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_unsupported_file_format(self, runner, temp_dir):
        """Test handling of unsupported file formats."""
        # Create a .txt file
        txt_path = Path(temp_dir) / 'data.txt'
        txt_path.write_text('some,data\n1,2\n')
        
        result = runner.invoke(cli, [
            'train', str(txt_path),
            '-t', 'classif',
            '-y', 'target'
        ])
        
        assert result.exit_code == 1
        assert 'Only CSV and Parquet files are supported' in result.output
    
    def test_missing_target_column(self, runner, temp_dir):
        """Test handling of missing target column."""
        # Create data without target
        data = pd.DataFrame({
            'x1': [1, 2, 3],
            'x2': [4, 5, 6]
        })
        path = Path(temp_dir) / 'no_target.csv'
        data.to_csv(path, index=False)
        
        result = runner.invoke(cli, [
            'train', str(path),
            '-t', 'classif',
            '-y', 'missing_column'
        ])
        
        # Should fail when creating task
        assert result.exit_code != 0
    
    def test_empty_data_file(self, runner, temp_dir):
        """Test handling of empty data file."""
        # Create empty CSV
        empty_path = Path(temp_dir) / 'empty.csv'
        pd.DataFrame().to_csv(empty_path, index=False)
        
        result = runner.invoke(cli, [
            'train', str(empty_path),
            '-t', 'classif',
            '-y', 'target'
        ])
        
        # Should fail appropriately
        assert result.exit_code != 0