"""Additional CLI commands for MLPY."""

import click
import sys
from pathlib import Path
import json
import yaml
from typing import Dict, Any, List

from .main import cli, task, learner, pipeline


@task.command('info')
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--target', '-y', help='Target column name')
def task_info(data_file, target):
    """Show information about a dataset."""
    import pandas as pd
    
    click.echo(f"Loading data from {data_file}...")
    
    # Load data
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    elif data_file.endswith('.parquet'):
        df = pd.read_parquet(data_file)
    else:
        click.echo("Error: Only CSV and Parquet files are supported", err=True)
        sys.exit(1)
    
    click.echo("\nDataset Information:")
    click.echo(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    click.echo(f"  Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    click.echo("\nColumn Information:")
    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        unique = df[col].nunique()
        
        info = f"  {col}: {dtype}"
        if nulls > 0:
            info += f" ({nulls} nulls)"
        if dtype == 'object' or unique < 20:
            info += f" [{unique} unique]"
            
        if target and col == target:
            info += " <- TARGET"
            
        click.echo(info)
    
    if target:
        click.echo(f"\nTarget Distribution ({target}):")
        value_counts = df[target].value_counts()
        for val, count in value_counts.head(10).items():
            pct = count / len(df) * 100
            click.echo(f"  {val}: {count} ({pct:.1f}%)")
        
        if len(value_counts) > 10:
            click.echo(f"  ... and {len(value_counts) - 10} more values")


@learner.command('list')
def learner_list():
    """List available learners."""
    from mlpy import learners
    
    click.echo("Available MLPY Learners:")
    click.echo("-" * 40)
    
    # Base learners
    click.echo("\nBase Classes:")
    click.echo("  - Learner: Abstract base class")
    click.echo("  - LearnerClassif: Base classification learner")
    click.echo("  - LearnerRegr: Base regression learner")
    
    # Native learners
    click.echo("\nNative Learners:")
    click.echo("  - LearnerClassifFeatureless: Baseline classifier (mode prediction)")
    click.echo("  - LearnerRegrFeatureless: Baseline regressor (mean/median prediction)")
    click.echo("  - LearnerClassifDebug: Debug classifier for testing")
    click.echo("  - LearnerRegrDebug: Debug regressor for testing")
    
    # Special learners
    if hasattr(learners, 'LearnerTGPClassifier'):
        click.echo("\nTransport Gaussian Process:")
        click.echo("  - LearnerTGPClassifier: TGP for classification")
        click.echo("  - LearnerTGPRegressor: TGP for regression")
    
    click.echo("\nScikit-learn Integration:")
    click.echo("-" * 40)
    click.echo("  Use learner_sklearn() to wrap any sklearn estimator")
    click.echo("  Examples:")
    click.echo("    - RandomForestClassifier")
    click.echo("    - GradientBoostingRegressor")
    click.echo("    - SVC, SVR")
    click.echo("    - XGBClassifier (if xgboost installed)")


@pipeline.command('create')
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Pipeline configuration file (JSON/YAML)')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file to save pipeline')
def pipeline_create(config, output):
    """Create a pipeline from configuration."""
    from mlpy.pipelines import (
        PipeOpScale, PipeOpImpute, PipeOpSelect, PipeOpEncode,
        PipeOpPCA, PipeOpLearner, linear_pipeline, GraphLearner
    )
    from mlpy.learners import learner_sklearn
    from sklearn.ensemble import RandomForestClassifier
    
    if not config:
        # Interactive mode
        click.echo("Creating pipeline interactively...")
        ops = []
        
        # Scaling
        if click.confirm("Add scaling?"):
            method = click.prompt("Scaling method", 
                                type=click.Choice(['standard', 'minmax', 'robust']),
                                default='standard')
            ops.append(PipeOpScale(method=method))
        
        # Feature selection
        if click.confirm("Add feature selection?"):
            k = click.prompt("Number of features to select", type=int, default=10)
            ops.append(PipeOpSelect(k=k))
        
        # PCA
        if click.confirm("Add PCA?"):
            n_components = click.prompt("Number of components", type=int, default=10)
            ops.append(PipeOpPCA(n_components=n_components))
        
        # Learner
        learner_type = click.prompt("Learner type", 
                                  type=click.Choice(['rf', 'lr', 'dt']),
                                  default='rf')
        
        if learner_type == 'rf':
            model = RandomForestClassifier(n_estimators=100)
        else:
            click.echo("Using default Random Forest")
            model = RandomForestClassifier(n_estimators=100)
        
        lrn = learner_sklearn(model)
        ops.append(PipeOpLearner(lrn))
        
        # Create pipeline
        graph = linear_pipeline(*ops)
        pipeline = GraphLearner(graph)
        
    else:
        # Load from config
        with open(config) as f:
            if config.endswith('.yaml') or config.endswith('.yml'):
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)
        
        click.echo(f"Creating pipeline from {config}")
        # TODO: Implement config-based pipeline creation
        click.echo("Config-based creation not yet implemented")
        return
    
    click.echo("\nPipeline created successfully!")
    
    if output:
        # Save pipeline
        from mlpy.persistence import save_model
        save_model(pipeline, output)
        click.echo(f"Pipeline saved to {output}")


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Input data file')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output file for processed data')
@click.option('--pipeline', '-p', type=click.Path(exists=True),
              help='Pipeline file to use for preprocessing')
@click.option('--scale', is_flag=True, help='Apply standard scaling')
@click.option('--impute', is_flag=True, help='Impute missing values')
@click.option('--encode', is_flag=True, help='Encode categorical variables')
def preprocess(input, output, pipeline, scale, impute, encode):
    """Preprocess data using pipelines.
    
    Example:
        mlpy preprocess -i raw_data.csv -o clean_data.csv --scale --impute
    """
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.pipelines import PipeOpScale, PipeOpImpute, PipeOpEncode
    
    click.echo(f"Loading data from {input}...")
    
    # Load data
    if input.endswith('.csv'):
        df = pd.read_csv(input)
    elif input.endswith('.parquet'):
        df = pd.read_parquet(input)
    else:
        click.echo("Error: Only CSV and Parquet files are supported", err=True)
        sys.exit(1)
    
    click.echo(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    
    if pipeline:
        # Use existing pipeline
        from mlpy.persistence import load_model
        pipe = load_model(pipeline)
        click.echo(f"Using pipeline from {pipeline}")
        # TODO: Apply pipeline to data
        click.echo("Pipeline application not yet implemented")
    else:
        # Apply individual operations
        if impute:
            click.echo("Imputing missing values...")
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            df[categorical_cols] = df[categorical_cols].fillna('missing')
        
        if encode:
            click.echo("Encoding categorical variables...")
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].nunique() < 20:  # Only encode low-cardinality
                    df = pd.get_dummies(df, columns=[col], prefix=col)
        
        if scale:
            click.echo("Scaling numeric features...")
            from sklearn.preprocessing import StandardScaler
            numeric_cols = df.select_dtypes(include=['number']).columns
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Save processed data
    click.echo(f"Saving processed data to {output}...")
    if output.endswith('.csv'):
        df.to_csv(output, index=False)
    elif output.endswith('.parquet'):
        df.to_parquet(output, index=False)
    else:
        df.to_csv(output, index=False)
    
    click.echo(f"Processed data saved: {df.shape[0]} rows x {df.shape[1]} columns")


@cli.command()
@click.argument('experiment_file', type=click.Path())
@click.option('--run', is_flag=True, help='Run the experiment immediately')
def experiment(experiment_file, run):
    """Define and run ML experiments from configuration files.
    
    Example experiment.yaml:
        name: "Iris Classification"
        data:
          file: "iris.csv"
          target: "species"
        learners:
          - type: "rf"
            params:
              n_estimators: 100
          - type: "lr"
        resampling:
          method: "cv"
          folds: 5
        measures: ["acc", "f1"]
    """
    if not experiment_file.endswith(('.yaml', '.yml', '.json')):
        click.echo("Error: Experiment file must be YAML or JSON", err=True)
        sys.exit(1)
    
    if run and Path(experiment_file).exists():
        # Load and run experiment
        with open(experiment_file) as f:
            if experiment_file.endswith(('.yaml', '.yml')):
                exp = yaml.safe_load(f)
            else:
                exp = json.load(f)
        
        click.echo(f"Running experiment: {exp.get('name', 'Unnamed')}")
        # TODO: Implement experiment runner
        click.echo("Experiment runner not yet implemented")
    
    else:
        # Create template
        template = {
            "name": "My Experiment",
            "description": "Experiment description",
            "data": {
                "file": "data.csv",
                "target": "target",
                "task_type": "classif"
            },
            "learners": [
                {"type": "rf", "params": {"n_estimators": 100}},
                {"type": "lr", "params": {}},
                {"type": "dt", "params": {"max_depth": 10}}
            ],
            "resampling": {
                "method": "cv",
                "folds": 5
            },
            "measures": ["acc", "auc"],
            "output": {
                "results": "results.csv",
                "plots": True
            }
        }
        
        with open(experiment_file, 'w') as f:
            if experiment_file.endswith(('.yaml', '.yml')):
                yaml.dump(template, f, default_flow_style=False)
            else:
                json.dump(template, f, indent=2)
        
        click.echo(f"Created experiment template: {experiment_file}")
        click.echo("Edit the file and run with: mlpy experiment <file> --run")