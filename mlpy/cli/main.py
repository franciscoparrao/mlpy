"""Main CLI entry point for MLPY."""

import click
import sys
import os
from pathlib import Path
from typing import Optional
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlpy
from mlpy import __version__


@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--version', '-v', is_flag=True, help='Show version and exit.')
def cli(ctx, version):
    """MLPY - Machine Learning Framework for Python
    
    A unified framework for machine learning inspired by mlr3.
    """
    if version:
        click.echo(f"MLPY version {__version__}")
        ctx.exit()
    elif ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.group()
def task():
    """Manage machine learning tasks."""
    pass


@cli.group()
def learner():
    """Manage learners and models."""
    pass


@cli.group()
def pipeline():
    """Manage ML pipelines."""
    pass


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--task-type', '-t', type=click.Choice(['classif', 'regr']), 
              required=True, help='Type of ML task')
@click.option('--target', '-y', required=True, help='Target column name')
@click.option('--learner', '-l', default='rf', help='Learner to use (default: rf)')
@click.option('--cv-folds', '-k', default=5, help='Number of CV folds')
@click.option('--measure', '-m', multiple=True, help='Evaluation measures')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
def train(data_file, task_type, target, learner, cv_folds, measure, output):
    """Train a model on data with cross-validation.
    
    Example:
        mlpy train data.csv -t classif -y target -l rf -k 5 -m acc -m auc
    """
    import pandas as pd
    from mlpy.tasks import TaskClassif, TaskRegr
    from mlpy.learners import learner_sklearn
    from mlpy.resamplings import ResamplingCV
    from mlpy import resample
    from mlpy.measures import (
        MeasureClassifAccuracy, MeasureClassifAUC, MeasureClassifF1,
        MeasureRegrRMSE, MeasureRegrR2, MeasureRegrMAE
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    
    click.echo(f"Loading data from {data_file}...")
    
    # Load data
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    elif data_file.endswith('.parquet'):
        df = pd.read_parquet(data_file)
    else:
        click.echo("Error: Only CSV and Parquet files are supported", err=True)
        sys.exit(1)
    
    # Create task
    if task_type == 'classif':
        task = TaskClassif(data=df, target=target)
        available_measures = {
            'acc': MeasureClassifAccuracy(),
            'accuracy': MeasureClassifAccuracy(),
            'auc': MeasureClassifAUC(),
            'f1': MeasureClassifF1()
        }
        default_measures = ['acc']
    else:
        task = TaskRegr(data=df, target=target)
        available_measures = {
            'rmse': MeasureRegrRMSE(),
            'r2': MeasureRegrR2(),
            'rsq': MeasureRegrR2(),
            'mae': MeasureRegrMAE()
        }
        default_measures = ['rmse']
    
    click.echo(f"Created {task_type} task: {task.nrow} rows, {len(task.feature_names)} features")
    
    # Create learner
    learner_map = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42) if task_type == 'classif' 
              else RandomForestRegressor(n_estimators=100, random_state=42),
        'lr': LogisticRegression(max_iter=1000) if task_type == 'classif' else Ridge(),
        'dt': DecisionTreeClassifier() if task_type == 'classif' else DecisionTreeRegressor()
    }
    
    if learner not in learner_map:
        click.echo(f"Error: Unknown learner '{learner}'. Available: {list(learner_map.keys())}", err=True)
        sys.exit(1)
    
    sklearn_model = learner_map[learner]
    lrn = learner_sklearn(sklearn_model, id=learner)
    
    # Select measures
    if not measure:
        measure = default_measures
    
    measures_list = []
    for m in measure:
        if m.lower() in available_measures:
            measures_list.append(available_measures[m.lower()])
        else:
            click.echo(f"Warning: Unknown measure '{m}', skipping", err=True)
    
    if not measures_list:
        click.echo("Error: No valid measures specified", err=True)
        sys.exit(1)
    
    # Run cross-validation
    click.echo(f"\nTraining {learner} with {cv_folds}-fold cross-validation...")
    resampling = ResamplingCV(folds=cv_folds)
    
    result = resample(
        task=task,
        learner=lrn,
        resampling=resampling,
        measures=measures_list
    )
    
    # Display results
    scores = result.aggregate()
    click.echo("\nResults:")
    click.echo(scores.to_string(index=False))
    
    # Save if requested
    if output:
        scores.to_csv(output, index=False)
        click.echo(f"\nResults saved to {output}")


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--task-type', '-t', type=click.Choice(['classif', 'regr']), 
              required=True, help='Type of ML task')
@click.option('--target', '-y', required=True, help='Target column name')
@click.option('--learners', '-l', multiple=True, default=['rf', 'lr', 'dt'],
              help='Learners to compare')
@click.option('--cv-folds', '-k', default=5, help='Number of CV folds')
@click.option('--measure', '-m', help='Evaluation measure')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
def benchmark(data_file, task_type, target, learners, cv_folds, measure, output):
    """Compare multiple learners on a dataset.
    
    Example:
        mlpy benchmark data.csv -t classif -y target -l rf -l lr -l dt
    """
    import pandas as pd
    from mlpy.tasks import TaskClassif, TaskRegr
    from mlpy.learners import learner_sklearn
    from mlpy.resamplings import ResamplingCV
    from mlpy import benchmark as mlpy_benchmark
    from mlpy.measures import MeasureClassifAccuracy, MeasureRegrRMSE
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    
    click.echo(f"Loading data from {data_file}...")
    
    # Load data
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    elif data_file.endswith('.parquet'):
        df = pd.read_parquet(data_file)
    else:
        click.echo("Error: Only CSV and Parquet files are supported", err=True)
        sys.exit(1)
    
    # Create task
    if task_type == 'classif':
        task = TaskClassif(data=df, target=target)
        default_measure = MeasureClassifAccuracy()
    else:
        task = TaskRegr(data=df, target=target)
        default_measure = MeasureRegrRMSE()
    
    click.echo(f"Created {task_type} task: {task.nrow} rows, {len(task.feature_names)} features")
    
    # Create learners
    learner_map = {
        'rf': ('Random Forest', 
               RandomForestClassifier(n_estimators=100, random_state=42) if task_type == 'classif' 
               else RandomForestRegressor(n_estimators=100, random_state=42)),
        'lr': ('Linear Model',
               LogisticRegression(max_iter=1000) if task_type == 'classif' else Ridge()),
        'dt': ('Decision Tree',
               DecisionTreeClassifier() if task_type == 'classif' else DecisionTreeRegressor()),
        'svm': ('SVM',
                None)  # Placeholder for SVM
    }
    
    learners_list = []
    for lrn_name in learners:
        if lrn_name in learner_map and learner_map[lrn_name][1] is not None:
            name, model = learner_map[lrn_name]
            learners_list.append(learner_sklearn(model, id=lrn_name))
        else:
            click.echo(f"Warning: Learner '{lrn_name}' not available, skipping", err=True)
    
    if not learners_list:
        click.echo("Error: No valid learners specified", err=True)
        sys.exit(1)
    
    # Run benchmark
    click.echo(f"\nBenchmarking {len(learners_list)} learners with {cv_folds}-fold CV...")
    resampling = ResamplingCV(folds=cv_folds)
    
    result = mlpy_benchmark(
        tasks=[task],
        learners=learners_list,
        resampling=resampling,
        measures=default_measure
    )
    
    # Display results
    click.echo("\nScore Table:")
    score_table = result.score_table()
    click.echo(score_table.to_string())
    
    click.echo("\nRanking:")
    ranking = result.rank_learners()
    click.echo(ranking.to_string(index=False))
    
    # Save if requested
    if output:
        # Save both tables
        with pd.ExcelWriter(output) as writer:
            score_table.to_excel(writer, sheet_name='Scores')
            ranking.to_excel(writer, sheet_name='Ranking', index=False)
        click.echo(f"\nResults saved to {output}")


@cli.command()
@click.argument('model_file', type=click.Path(exists=True))
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for predictions')
@click.option('--proba', is_flag=True, help='Output probabilities instead of classes')
def predict(model_file, data_file, output, proba):
    """Make predictions using a saved model.
    
    Example:
        mlpy predict model.pkl test_data.csv -o predictions.csv
    """
    import pandas as pd
    from mlpy.persistence import load_model
    from mlpy.tasks import TaskClassif, TaskRegr
    
    click.echo(f"Loading model from {model_file}...")
    
    # Load model
    try:
        bundle = load_model(model_file, return_bundle=True)
        model = bundle.model
        metadata = bundle.metadata
        click.echo(f"Loaded {metadata.get('model_type', 'Unknown')} model")
    except:
        # Try loading without bundle
        model = load_model(model_file)
        metadata = {}
        click.echo("Loaded model (no metadata)")
    
    # Load data
    click.echo(f"Loading data from {data_file}...")
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    elif data_file.endswith('.parquet'):
        df = pd.read_parquet(data_file)
    else:
        click.echo("Error: Only CSV and Parquet files are supported", err=True)
        sys.exit(1)
    
    # Create task (dummy target if not present)
    if 'target' in df.columns:
        target_col = 'target'
    else:
        # Add dummy target with at least 2 classes for classification
        df['_dummy_target'] = ['A', 'B'] * (len(df) // 2) + ['A'] * (len(df) % 2)
        target_col = '_dummy_target'
    
    # Infer task type from model
    if hasattr(model, 'task_type'):
        task_type = model.task_type
    else:
        # Try to infer from model type
        task_type = 'classif'  # Default assumption
    
    if task_type == 'classif':
        task = TaskClassif(data=df, target=target_col)
    else:
        task = TaskRegr(data=df, target=target_col)
    
    click.echo(f"Making predictions for {task.nrow} samples...")
    
    # Make predictions
    if proba:
        # Check if model supports probabilities
        if hasattr(model, 'predict_type') and model.predict_type == 'prob':
            # Model already configured for probabilities
            pred = model.predict(task)
            if hasattr(pred, 'prob') and pred.prob is not None:
                pred_df = pd.DataFrame(pred.prob, columns=[f'prob_class_{i}' for i in range(pred.prob.shape[1])])
            else:
                click.echo("Warning: Model does not support probability predictions, falling back to regular predictions", err=True)
                pred_df = pd.DataFrame({'prediction': pred.response})
        else:
            # Try to get probabilities from sklearn model
            if hasattr(model, '_fitted_estimator') and hasattr(model._fitted_estimator, 'predict_proba'):
                # Clone the model with predict_type='prob'
                try:
                    from mlpy.learners import learner_sklearn
                    prob_model = learner_sklearn(model._fitted_estimator, predict_type='prob')
                    prob_model._fitted_estimator = model._fitted_estimator  # Use already trained model
                    pred = prob_model.predict(task)
                    if hasattr(pred, 'prob') and pred.prob is not None:
                        pred_df = pd.DataFrame(pred.prob, columns=[f'prob_class_{i}' for i in range(pred.prob.shape[1])])
                    else:
                        pred_df = pd.DataFrame({'prediction': pred.response})
                except:
                    click.echo("Warning: Could not get probability predictions, falling back to regular predictions", err=True)
                    pred = model.predict(task)
                    pred_df = pd.DataFrame({'prediction': pred.response})
            else:
                click.echo("Warning: Model does not support probability predictions, falling back to regular predictions", err=True)
                pred = model.predict(task)
                pred_df = pd.DataFrame({'prediction': pred.response})
    else:
        pred = model.predict(task)
        pred_df = pd.DataFrame({'prediction': pred.response})
    
    # Add row IDs if available
    if hasattr(task, 'row_ids'):
        pred_df.insert(0, 'row_id', task.row_ids)
    
    # Display sample
    click.echo("\nSample predictions:")
    click.echo(pred_df.head(10).to_string(index=False))
    
    # Save if requested
    if output:
        pred_df.to_csv(output, index=False)
        click.echo(f"\nPredictions saved to {output}")
    else:
        click.echo("\n(Use -o to save predictions to file)")


@cli.command()
def info():
    """Show information about MLPY installation."""
    import mlpy
    import sklearn
    import pandas as pd
    import numpy as np
    
    click.echo("MLPY Information")
    click.echo("=" * 40)
    click.echo(f"MLPY version: {mlpy.__version__}")
    click.echo(f"Python version: {sys.version.split()[0]}")
    click.echo(f"Installation path: {Path(mlpy.__file__).parent}")
    
    click.echo("\nCore Dependencies:")
    click.echo(f"  scikit-learn: {sklearn.__version__}")
    click.echo(f"  pandas: {pd.__version__}")
    click.echo(f"  numpy: {np.__version__}")
    
    # Check optional dependencies
    click.echo("\nOptional Dependencies:")
    optionals = {
        'dask': 'dask.dataframe',
        'vaex': 'vaex',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'shap': 'shap',
        'lime': 'lime'
    }
    
    for name, module in optionals.items():
        try:
            mod = __import__(module.split('.')[0])
            version = getattr(mod, '__version__', 'installed')
            click.echo(f"  {name}: {version}")
        except ImportError:
            click.echo(f"  {name}: not installed")
    
    # Show available components
    click.echo("\nAvailable Components:")
    click.echo(f"  Learners: {len([x for x in dir(mlpy.learners) if x.startswith('Learner')])}")
    click.echo(f"  Measures: {len([x for x in dir(mlpy.measures) if 'Measure' in x])}")
    click.echo(f"  Pipeline Ops: {len([x for x in dir(mlpy.pipelines) if x.startswith('PipeOp')])}")
    click.echo(f"  Resamplings: {len([x for x in dir(mlpy.resamplings) if x.startswith('Resampling')])}")


@cli.command()
@click.option('--shell', '-s', type=click.Choice(['ipython', 'python']), 
              default='ipython', help='Shell to use')
def shell(shell):
    """Start an interactive MLPY shell."""
    banner = """
MLPY Interactive Shell
======================
Pre-imported modules:
  - mlpy (full package)
  - pd (pandas)
  - np (numpy)
  - TaskClassif, TaskRegr
  - learner_sklearn
  - resample, benchmark

Example:
  >>> df = pd.read_csv('data.csv')
  >>> task = TaskClassif(data=df, target='target')
  >>> learner = learner_sklearn(RandomForestClassifier())
  >>> result = resample(task, learner, ResamplingCV())
"""
    
    # Pre-imports
    import pandas as pd
    import numpy as np
    import mlpy
    from mlpy.tasks import TaskClassif, TaskRegr
    from mlpy.learners import learner_sklearn
    from mlpy import resample, benchmark
    from mlpy.resamplings import ResamplingCV, ResamplingHoldout
    from mlpy.measures import (
        MeasureClassifAccuracy, MeasureClassifAUC, MeasureClassifF1,
        MeasureRegrRMSE, MeasureRegrR2, MeasureRegrMAE
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    namespace = {
        'mlpy': mlpy,
        'pd': pd,
        'np': np,
        'TaskClassif': TaskClassif,
        'TaskRegr': TaskRegr,
        'learner_sklearn': learner_sklearn,
        'resample': resample,
        'benchmark': benchmark,
        'ResamplingCV': ResamplingCV,
        'ResamplingHoldout': ResamplingHoldout,
        'RandomForestClassifier': RandomForestClassifier,
        'RandomForestRegressor': RandomForestRegressor,
        'MeasureClassifAccuracy': MeasureClassifAccuracy,
        'MeasureClassifAUC': MeasureClassifAUC,
        'MeasureClassifF1': MeasureClassifF1,
        'MeasureRegrRMSE': MeasureRegrRMSE,
        'MeasureRegrR2': MeasureRegrR2,
        'MeasureRegrMAE': MeasureRegrMAE,
    }
    
    if shell == 'ipython':
        try:
            from IPython import start_ipython
            start_ipython(argv=[], user_ns=namespace, banner1=banner)
        except ImportError:
            click.echo("IPython not installed, falling back to standard Python shell")
            shell = 'python'
    
    if shell == 'python':
        import code
        code.interact(banner=banner, local=namespace)


# Import additional commands
from . import commands

if __name__ == '__main__':
    cli()